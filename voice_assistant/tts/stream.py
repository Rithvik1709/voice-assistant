from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from voice_assistant.benchmark import BenchmarkTracker
from voice_assistant.tts.normalize import normalize_for_speech
from voice_assistant.tts.queue import AudioChunk, AudioChunkQueue, safe_put

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"([.!?]+(?:\s+|$))")
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
        words = chunk.split()
        if not words:
            continue

        if len(words) > max_tokens:
            for i in range(0, len(words), max_tokens):
                out.append(" ".join(words[i:i + max_tokens]))
        else:
            out.append(chunk)

    return out


def piper_python_available() -> bool:
    try:
        return importlib.util.find_spec("piper") is not None
    except (ImportError, ValueError):
        return False


def read_voice_sample_rate(voice_path: Path | str) -> int | None:
    """Read the output sample rate from a Piper voice's `.onnx.json` config."""
    config_path = Path(f"{voice_path}.json")
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return int(data["audio"]["sample_rate"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


@dataclass(slots=True)
class PiperConfig:
    voice_path: Path
    sample_rate: int = 22_050
    backend: str = "auto"  # auto | python | cli


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


def create_synthesizer(config: PiperConfig) -> Synthesizer:
    backend = config.backend
    if backend == "python" or (backend == "auto" and piper_python_available()):
        return PiperPythonSynth(Path(config.voice_path))
    return PiperProcess(["piper", "--model", str(config.voice_path), "--output_raw"])


@dataclass(slots=True)
class _FlushRequest:
    done: asyncio.Future[None]


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
        self.ingest_queue: asyncio.Queue[str | _FlushRequest] = asyncio.Queue(maxsize=32)
        self.sample_rate = read_voice_sample_rate(config.voice_path) or config.sample_rate

        self._synth: Synthesizer | None = None
        self._worker: asyncio.Task[None] | None = None
        self._running = False
        # Bumped by cancel_pending(); audio produced for an older generation is dropped.
        self._generation = 0

    async def start(self) -> None:
        if self._running:
            return
        if self._synth is None:
            self._synth = await asyncio.to_thread(create_synthesizer, self.config)
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

    async def synthesize_sentence(self, sentence: str) -> bool:
        """Queue a sentence for synthesis. Returns False if the queue stayed full."""
        if not sentence.strip():
            return True

        try:
            await asyncio.wait_for(self.ingest_queue.put(sentence), timeout=5.0)
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
            items: list[str | _FlushRequest] = [await self.ingest_queue.get()]
            # Coalesce sentences that are already waiting, but never wait for
            # more: the first sentence of a reply should be spoken immediately.
            while not isinstance(items[-1], _FlushRequest) and len(items) < _MAX_BATCH:
                try:
                    items.append(self.ingest_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            batch = [item for item in items if isinstance(item, str)]
            flushes = [item for item in items if isinstance(item, _FlushRequest)]

            try:
                if batch:
                    await self._process_batch(batch)
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

    async def _process_batch(self, batch: list[str]) -> None:
        if not batch or self._synth is None:
            return

        generation = self._generation
        combined = normalize_for_speech(" ".join(batch))
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
                asyncio.to_thread(self._synth.synthesize, combined),
                timeout=_PIPER_TIMEOUT_S,
            )
        except TimeoutError as exc:
            logger.error("Piper synthesis timed out after %.1fs", _PIPER_TIMEOUT_S)
            self._synth.close()
            raise RuntimeError("Piper synthesis timed out") from exc

        if not pcm or generation != self._generation:
            return

        if self.bench and self.bench.current.first_audio_ts is None:
            self.bench.mark("first_audio_ts")

        dur_sec = len(pcm) / 2 / self.sample_rate
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
                    sample_rate=self.sample_rate,
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
