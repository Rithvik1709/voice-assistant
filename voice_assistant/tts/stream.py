from __future__ import annotations

import asyncio
import importlib.util
import itertools
import json
import logging
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from voice_assistant.benchmark import BenchmarkTracker
from voice_assistant.lang import SENTENCE_END_CHARS, base_language, language_name, word_spans
from voice_assistant.tts.normalize import normalize_for_speech
from voice_assistant.tts.queue import AudioChunk, AudioChunkQueue, safe_put
from voice_assistant.tts.voices import VoiceRouter

logger = logging.getLogger(__name__)

# ASCII marks end a sentence only before whitespace; CJK, Indic, Arabic and
# other full stops end it outright.
_WIDE_ENDS = re.escape(SENTENCE_END_CHARS.replace(".", "").replace("!", "").replace("?", ""))
_SENTENCE_SPLIT = re.compile(rf"([.!?]+(?:\s+|$)|[{_WIDE_ENDS}]+\s*)")
_WHITESPACE = re.compile(r"\s+")
_PIPER_TIMEOUT_S = 30.0
_MAX_BATCH = 4


def sentence_chunks_from_tokens(tokens: list[str], max_tokens: int = 28) -> list[str]:
    """Split streamed tokens into sentences, capping each chunk at `max_tokens` words."""
    text = "".join(tokens).strip()
    if not text:
        return []

    # Split with captures so the sentence delimiters are kept.
    parts = _SENTENCE_SPLIT.split(text)

    chunks = []
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i] + parts[i + 1]
        if sentence.strip():
            chunks.append(sentence.strip())

    if len(parts) % 2 == 1 and parts[-1].strip():
        chunks.append(parts[-1].strip())

    if not chunks:
        return [text]

    out: list[str] = []
    for chunk in chunks:
        # Chinese and Japanese have no spaces: their words are counted by characters.
        spans = word_spans(chunk)
        if not spans:
            continue

        if len(spans) > max_tokens:
            for i in range(0, len(spans), max_tokens):
                group = spans[i:i + max_tokens]
                out.append(_WHITESPACE.sub(" ", chunk[group[0][0]:group[-1][1]]))
        else:
            out.append(chunk)

    return out


def piper_python_available() -> bool:
    try:
        return importlib.util.find_spec("piper") is not None
    except (ImportError, ValueError):
        return False


def read_voice_sample_rate(voice_path: Path | str | None) -> int | None:
    """Read the output sample rate from a Piper voice's `.onnx.json` config."""
    if voice_path is None:
        return None
    config_path = Path(f"{voice_path}.json")
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return int(data["audio"]["sample_rate"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


@dataclass(slots=True)
class PiperConfig:
    voice_path: Path | None  # the default voice
    sample_rate: int = 22_050
    backend: str = "auto"  # auto | python | cli
    # One voice per language, chosen by the language of each reply.
    voices_dir: Path | None = None
    # For languages without a voice: auto | espeak | default | none
    fallback: str = "auto"


class Synthesizer(Protocol):
    def synthesize(self, text: str) -> bytes: ...

    def close(self) -> None: ...


class PiperProcess:
    """Runs the Piper CLI once per request and returns raw 16-bit PCM.

    `--output_raw` is understood by both the C++ binary and the `piper-tts`
    Python CLI, and closing stdin marks the end of the text, so the output
    length never has to be inferred from a WAV header.
    """

    def __init__(self, cmd: list[str], timeout_s: float = _PIPER_TIMEOUT_S) -> None:
        self.cmd = cmd
        self.timeout_s = timeout_s
        self.proc: subprocess.Popen[bytes] | None = None

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""

        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            pcm, stderr = self.proc.communicate(
                (text.strip() + "\n").encode("utf-8"),
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            self.proc.kill()
            self.proc.communicate()
            raise RuntimeError("Piper synthesis timed out") from exc
        finally:
            proc, self.proc = self.proc, None

        if proc.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip().splitlines()
            raise RuntimeError(
                f"Piper process exited with code {proc.returncode}"
                + (f": {detail[-1]}" if detail else "")
            )
        # Guard against a trailing odd byte so the PCM stays 16-bit aligned.
        return pcm[: len(pcm) - (len(pcm) % 2)]

    def close(self) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            proc.kill()


_VOICE_CACHE: dict[str, Any] = {}
_VOICE_CACHE_LOCK = threading.Lock()


class PiperPythonSynth:
    """In-process synthesis through the `piper-tts` package.

    Loaded voices are cached per path so concurrent gRPC streams share one
    ONNX session instead of loading the model for every connection.
    """

    def __init__(self, voice_path: Path) -> None:
        key = str(voice_path)
        with _VOICE_CACHE_LOCK:
            voice = _VOICE_CACHE.get(key)
            if voice is None:
                from piper import PiperVoice  # type: ignore

                voice = PiperVoice.load(key)
                _VOICE_CACHE[key] = voice
        self.voice = voice

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""
        # piper-tts >= 1.3 yields AudioChunk objects; older releases expose
        # synthesize_stream_raw() yielding bytes.
        stream_raw = getattr(self.voice, "synthesize_stream_raw", None)
        if stream_raw is not None:
            return b"".join(stream_raw(text))
        return b"".join(chunk.audio_int16_bytes for chunk in self.voice.synthesize(text))

    def close(self) -> None:
        pass


# espeak-ng names a few languages differently from Whisper.
_ESPEAK_VOICES = {"zh": "cmn", "jw": "jv", "no": "nb"}


def _wav_to_pcm(data: bytes) -> tuple[bytes, int]:
    """16-bit PCM and sample rate from a WAV file.

    espeak-ng writing to a pipe cannot seek back to fill in the chunk sizes,
    so the data chunk simply runs to the end of the output.
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise RuntimeError("espeak-ng did not produce WAV audio")
    pos, rate = 12, 22_050
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        size = int.from_bytes(data[pos + 4:pos + 8], "little")
        body = pos + 8
        if chunk_id == b"fmt ":
            rate = int.from_bytes(data[body + 4:body + 8], "little")
        elif chunk_id == b"data":
            pcm = data[body:body + size]
            return pcm[: len(pcm) - (len(pcm) % 2)], rate
        pos = body + size + (size & 1)
    raise RuntimeError("espeak-ng output has no audio data")


class EspeakSynth:
    """espeak-ng, for languages that have no Piper voice.

    It sounds robotic but speaks over 100 languages and is a small system
    package (`brew install espeak-ng`, `apt-get install espeak-ng`).
    """

    def __init__(self, language: str, binary: str = "espeak-ng", timeout_s: float = _PIPER_TIMEOUT_S) -> None:
        voice = _ESPEAK_VOICES.get(language, language)
        # -b 1: the text is UTF-8. "--" stops text starting with "-" being read as an option.
        self.cmd = [binary, "-v", voice, "-b", "1", "--stdout", "--"]
        self.timeout_s = timeout_s
        self.sample_rate = 22_050

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""
        try:
            proc = subprocess.run([*self.cmd, text.strip()], capture_output=True, timeout=self.timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("espeak-ng synthesis timed out") from exc
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()
            raise RuntimeError(
                f"espeak-ng exited with code {proc.returncode}" + (f": {detail[-1]}" if detail else "")
            )
        pcm, self.sample_rate = _wav_to_pcm(proc.stdout)
        return pcm

    def close(self) -> None:
        pass


def create_synthesizer(config: PiperConfig) -> Synthesizer:
    backend = config.backend
    if backend == "python" or (backend == "auto" and piper_python_available()):
        return PiperPythonSynth(Path(config.voice_path))
    return PiperProcess(["piper", "--model", str(config.voice_path), "--output_raw"])


@dataclass(slots=True)
class _FlushRequest:
    done: asyncio.Future[None]


@dataclass(slots=True)
class _Sentence:
    text: str
    language: str | None = None


class PiperStreamingTTS:
    def __init__(
        self,
        config: PiperConfig,
        playback_queue: AudioChunkQueue,
        bench: BenchmarkTracker | None = None,
    ) -> None:
        self.config = config
        self.playback_queue = playback_queue
        self.bench = bench
        self.ingest_queue: asyncio.Queue[_Sentence | str | _FlushRequest] = asyncio.Queue(maxsize=32)
        self.router = VoiceRouter(
            Path(config.voice_path) if config.voice_path is not None else None,
            config.voices_dir,
        )
        self.sample_rate = read_voice_sample_rate(self.router.default_voice) or config.sample_rate

        # The default voice, loaded at start; other languages load on first use.
        self._synth: Synthesizer | None = None
        self._voices: dict[str, tuple[Synthesizer | None, int]] = {}
        self._worker: asyncio.Task[None] | None = None
        self._running = False
        # Bumped by cancel_pending(); audio produced for an older generation is dropped.
        self._generation = 0

    async def start(self) -> None:
        if self._running:
            return
        if self._synth is None and self.router.default_voice is not None:
            self._synth = await asyncio.to_thread(
                create_synthesizer, replace(self.config, voice_path=self.router.default_voice)
            )
        self._running = True
        self._worker = asyncio.create_task(self._tts_worker())

    async def stop(self) -> None:
        if self._running:
            try:
                await asyncio.wait_for(self.flush(), timeout=_PIPER_TIMEOUT_S)
            except Exception:
                logger.exception("TTS flush failed during shutdown")
        self._running = False
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None
        if self._synth is not None:
            self._synth.close()
        for synth, _rate in self._voices.values():
            if synth is not None:
                synth.close()

    async def synthesize_sentence(self, sentence: str, language: str | None = None) -> bool:
        """Queue a sentence for synthesis in `language` (None: the default voice).

        Returns False if the queue stayed full.
        """
        if not sentence.strip():
            return True

        try:
            await asyncio.wait_for(self.ingest_queue.put(_Sentence(sentence, language)), timeout=5.0)
            return True
        except TimeoutError:
            logger.warning("TTS ingest queue full; sentence not accepted")
            return False

    def cancel_pending(self) -> None:
        """Drop queued sentences and discard audio from any in-flight synthesis."""
        self._generation += 1
        while True:
            try:
                item = self.ingest_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self.ingest_queue.task_done()
            if isinstance(item, _FlushRequest) and not item.done.done():
                item.done.set_result(None)

    async def _tts_worker(self) -> None:
        while self._running:
            items: list[_Sentence | str | _FlushRequest] = [await self.ingest_queue.get()]
            # Coalesce sentences that are already waiting, but never wait for
            # more: the first sentence of a reply should be spoken immediately.
            while not isinstance(items[-1], _FlushRequest) and len(items) < _MAX_BATCH:
                try:
                    items.append(self.ingest_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            batch = [
                item if isinstance(item, _Sentence) else _Sentence(item)
                for item in items
                if not isinstance(item, _FlushRequest)
            ]
            flushes = [item for item in items if isinstance(item, _FlushRequest)]

            try:
                # Sentences are only combined with neighbours in the same language.
                for language, group in itertools.groupby(batch, key=lambda s: s.language):
                    await self._process_batch([s.text for s in group], language)
                for req in flushes:
                    if not req.done.done():
                        req.done.set_result(None)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("TTS synthesis failed; dropping %d sentence(s): %s", len(batch), exc)
                for req in flushes:
                    if not req.done.done():
                        req.done.set_result(None)
            finally:
                for _ in items:
                    self.ingest_queue.task_done()

    async def _voice_for(self, language: str | None) -> tuple[Synthesizer | None, int]:
        """The synthesizer for `language` and its sample rate, loading it if needed."""
        voice = self.router.voice_for(language)
        if voice is not None and voice == self.router.default_voice:
            return self._synth, self.sample_rate

        key = str(voice) if voice is not None else f"fallback:{base_language(language)}"
        if key not in self._voices:
            if voice is not None:
                synth = await asyncio.to_thread(create_synthesizer, replace(self.config, voice_path=voice))
                self._voices[key] = (synth, read_voice_sample_rate(voice) or self.config.sample_rate)
            else:
                self._voices[key] = self._fallback_voice(base_language(language) or "")
        return self._voices[key]

    def _fallback_voice(self, language: str) -> tuple[Synthesizer | None, int]:
        mode = self.config.fallback
        if mode == "auto":
            mode = "espeak" if shutil.which("espeak-ng") else "default"
        name = language_name(language) or language
        if mode == "espeak":
            logger.info("No Piper voice for %s; speaking it with espeak-ng", name)
            return EspeakSynth(language), 22_050
        if mode == "default":
            logger.warning(
                "No Piper voice for %s; using the default voice. Add one to PIPER_VOICES_DIR "
                "or install espeak-ng.", name,
            )
            return self._synth, self.sample_rate
        logger.warning("No Piper voice for %s; replies in it are not spoken (TTS_FALLBACK=none)", name)
        return None, self.sample_rate

    async def _process_batch(self, batch: list[str], language: str | None = None) -> None:
        if not batch:
            return
        synth, sample_rate = await self._voice_for(language)
        if synth is None:
            return

        generation = self._generation
        combined = normalize_for_speech(" ".join(batch), language)
        if not combined:
            return
        self.playback_queue.record_batch(len(batch))
        logger.debug(
            "TTS | ingest_q=%d playback_q=%d batch=%d",
            self.ingest_queue.qsize(),
            self.playback_queue.qsize(),
            len(batch),
        )

        if self.bench and self.bench.current.tts_start_ts is None:
            self.bench.mark("tts_start_ts")

        try:
            pcm = await asyncio.wait_for(
                asyncio.to_thread(synth.synthesize, combined),
                timeout=_PIPER_TIMEOUT_S,
            )
        except TimeoutError as exc:
            logger.error("Piper synthesis timed out after %.1fs", _PIPER_TIMEOUT_S)
            synth.close()
            raise RuntimeError("Piper synthesis timed out") from exc

        if not pcm or generation != self._generation:
            return
        # espeak-ng reports its rate per call; Piper voices have a fixed one.
        sample_rate = getattr(synth, "sample_rate", None) or sample_rate

        if self.bench and self.bench.current.first_audio_ts is None:
            self.bench.mark("first_audio_ts")

        dur_sec = len(pcm) / 2 / sample_rate
        if self.bench:
            self.bench.add_synthesized_audio(dur_sec)
            self.bench.current.tts_end_ts = time.perf_counter()

        for i, chunk in enumerate(self._split_audio_chunks(pcm)):
            if generation != self._generation:
                return
            success = await safe_put(
                self.playback_queue,
                AudioChunk(
                    pcm16=chunk,
                    sample_rate=sample_rate,
                    debug_text=combined if i == 0 else "",
                ),
                timeout=5.0,
            )
            if not success:
                raise RuntimeError("TTS playback queue full; synthesis aborted")

    @staticmethod
    def _split_audio_chunks(audio_bytes: bytes, chunk_size: int = 8192):
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i:i + chunk_size]

    async def flush(self) -> None:
        """Wait until queued sentences have been synthesized into the playback queue."""
        if not self._running or self._worker is None:
            return

        loop = asyncio.get_running_loop()
        done: asyncio.Future[None] = loop.create_future()
        await self.ingest_queue.put(_FlushRequest(done))
        await done
