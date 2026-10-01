from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from collections.abc import AsyncIterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from vosk import KaldiRecognizer, Model, SetLogLevel  # type: ignore
    _VOSK_AVAILABLE = True
    SetLogLevel(-1)
except ImportError:
    _VOSK_AVAILABLE = False
    class Model:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
    class KaldiRecognizer:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

try:
    from whisper_cpp_python import Whisper  # type: ignore
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False
    class Whisper:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

from voice_assistant.asr.partial import PartialTranscriptStabilizer
from voice_assistant.asr.vad import VADConfig, VoiceActivityDetector
from voice_assistant.lang import AUTO

logger = logging.getLogger(__name__)

# Frames kept from before the VAD fires, so the first syllable is not clipped.
_PREROLL_FRAMES = 3

# Words that almost never end a request ("turn on the ...", "and ...", "um").
# When the transcript so far ends on one, the user is probably pausing to
# think, so the endpoint waits longer instead of cutting them off.
DANGLING_WORDS = frozenset(
    """
    a an the and or but so because if then than that which who whose what where when
    to of for with from in on at by about into onto as like is are was were be been am
    my your his her their our its this these those some any very really just also plus
    um uh er erm hmm can could would should will shall do does did not
    aur ki ka ke ko se mein toh par ya lekin kyunki jo
    """.split()
)


def looks_unfinished(text: str) -> bool:
    words = text.lower().split()
    return bool(words) and words[-1].strip(".,!?") in DANGLING_WORDS


@dataclass(slots=True)
class ASREvent:
    type: str  # speech_start | partial | final | wake
    text: str
    confidence: float
    timestamp_ms: int
    speech_end_ts: float | None = None
    # Language the user spoke (e.g. "hi"), when known.
    language: str | None = None


@dataclass(slots=True)
class _Snapshot:
    """A decode of the utterance so far, running ahead of the endpoint."""

    future: Future
    # The audio-clock time of the last speech frame it contains. If no speech
    # follows, it holds the whole utterance and the endpoint can reuse it.
    speech_audio: float
    reported: bool = False


class _VoskRecognizer:
    streaming = True

    def __init__(self, model_path: str, sample_rate: int, model: Any | None = None) -> None:
        if not _VOSK_AVAILABLE:
            raise RuntimeError(
                "vosk is not installed. Install it with `pip install 'voice-assistant[local]'`."
            )

        # A loaded Model can be shared across recognizers (one per gRPC stream).
        self.recognizer = KaldiRecognizer(model or Model(model_path), sample_rate)
        self._segments: list[str] = []

    def accept_waveform(self, audio_bytes: bytes) -> None:
        # AcceptWaveform returns True when Kaldi detects its own endpoint. The
        # finished segment is only available from Result() at that moment and
        # is lost if not collected, so keep it until the utterance ends.
        if self.recognizer.AcceptWaveform(audio_bytes):
            text = json.loads(self.recognizer.Result()).get("text", "").strip()
            if text:
                self._segments.append(text)

    def partial_result(self) -> tuple[str, float]:
        data = json.loads(self.recognizer.PartialResult())
        partial = " ".join([*self._segments, data.get("partial", "")]).strip()
        return partial, float(data.get("confidence", 0.0) or 0.0)

    def final_result(self, _utterance: bytes) -> tuple[str, float]:
        data = json.loads(self.recognizer.FinalResult())
        text = " ".join([*self._segments, data.get("text", "")]).strip()
        self._segments.clear()
        return text, float(data.get("confidence", 0.0) or 0.0)

    def reset(self) -> None:
        self._segments.clear()
        reset = getattr(self.recognizer, "Reset", None)
        if reset is not None:
            reset()


class _WhisperCppRecognizer:
    streaming = False

    def __init__(self, model_path: str, sample_rate: int) -> None:
        if not _WHISPER_AVAILABLE:
            raise RuntimeError("whisper_cpp_python is required for the whispercpp ASR backend")

        self.whisper = Whisper(model_path)
        self.sample_rate = sample_rate

    def accept_waveform(self, audio_bytes: bytes) -> None:
        pass

    def partial_result(self) -> tuple[str, float]:
        return "", 0.0

    def final_result(self, utterance: bytes) -> tuple[str, float]:
        audio = np.frombuffer(utterance, dtype=np.int16).astype("float32") / 32768.0
        segments = self.whisper.transcribe(audio, beam_size=1)
        text = " ".join(seg.text for seg in segments).strip()
        return text, 0.5

    def reset(self) -> None:
        pass


def default_whisper_model(language: str = "en") -> str:
    """Whisper model used when ASR_MODEL_PATH is not set.

    English gets the small, fast English-only model. Other languages need a
    multilingual one; large-v3-turbo is accurate across Whisper's 99 languages
    at a fraction of large-v3's cost.
    """
    return "base.en" if language == "en" else "large-v3-turbo"


class _FasterWhisperRecognizer:
    """Whisper through CTranslate2: much more accurate than Vosk small.

    Whisper decodes a whole utterance at once, so there are no partial
    transcripts; the utterance is transcribed as soon as the endpoint fires.
    `model` is a local CTranslate2 model directory or a size name such as
    "base.en" or "small" (downloaded from Hugging Face on first use).

    With language "auto" the language is detected on every utterance and
    exposed as `last_language`. `allowed_languages` limits detection to the
    languages the user is expected to speak.
    """

    streaming = False
    # Segments the model itself thinks are silence are dropped: Whisper
    # otherwise invents text ("Thank you.") for noise.
    NO_SPEECH_THRESHOLD = 0.6

    def __init__(
        self,
        model: str,
        sample_rate: int,
        language: str = "en",
        whisper_model: Any | None = None,
        allowed_languages: tuple[str, ...] = (),
        hotwords: str = "",
    ) -> None:
        if whisper_model is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is not installed. Install it with `pip install 'voice-assistant[whisper]'`."
                ) from exc
            whisper_model = WhisperModel(model or default_whisper_model(language), device="auto", compute_type="int8")
        if sample_rate != 16_000:
            raise ValueError("faster-whisper expects 16 kHz audio")
        self.model = whisper_model
        self.allowed_languages = tuple(allowed_languages)
        detect = not language or language == AUTO
        if detect and len(self.allowed_languages) == 1:
            # Nothing to choose between: skip detection.
            language, detect = self.allowed_languages[0], False
        # None asks Whisper to detect the language.
        self.language: str | None = None if detect else language
        self.last_language: str | None = self.language
        # Words Whisper should prefer when unsure, such as the wake word "Vaani".
        self.hotwords = hotwords or None

    def accept_waveform(self, audio_bytes: bytes) -> None:
        pass

    def partial_result(self) -> tuple[str, float]:
        return "", 0.0

    def final_result(self, utterance: bytes) -> tuple[str, float]:
        text, confidence, language = self.decode(utterance)
        if text:
            self.last_language = language
        return text, confidence

    def decode(self, utterance: bytes) -> tuple[str, float, str | None]:
        """Transcribe `utterance` without changing any state: (text, confidence, language).

        Safe to call from another thread while audio is still arriving, which
        is how StreamingASR decodes ahead of the endpoint.
        """
        if not utterance:
            return "", 0.0, self.language
        audio = np.frombuffer(utterance, dtype=np.int16).astype(np.float32) / 32768.0
        segments, info = self._transcribe(audio, self.language)
        language = self.language
        if language is None:
            language = getattr(info, "language", None)
            if self.allowed_languages and language not in self.allowed_languages:
                # Detected something the user does not speak (common for short
                # utterances): decode again as the likeliest allowed language.
                language = self._likeliest_allowed(info)
                segments, info = self._transcribe(audio, language)
        kept = [seg for seg in segments if seg.no_speech_prob < self.NO_SPEECH_THRESHOLD]
        text = " ".join(seg.text.strip() for seg in kept).strip()
        confidence = float(np.exp(np.mean([seg.avg_logprob for seg in kept]))) if kept else 0.0
        return text, confidence, language

    def _transcribe(self, audio: np.ndarray, language: str | None) -> tuple[Any, Any]:
        return self.model.transcribe(
            audio,
            language=language,
            beam_size=1,
            vad_filter=False,  # Vaani's VAD already cut the utterance
            condition_on_previous_text=False,
            without_timestamps=True,
            hotwords=self.hotwords,
        )

    def _likeliest_allowed(self, info: Any) -> str:
        probs = dict(getattr(info, "all_language_probs", None) or [])
        return max(self.allowed_languages, key=lambda code: probs.get(code, 0.0))

    def reset(self) -> None:
        pass


def load_recognizer(
    backend: str,
    model_path: str,
    sample_rate: int,
    language: str = "en",
    shared: Any = None,
    allowed_languages: tuple[str, ...] = (),
    hotwords: str = "",
) -> Any:
    """Create the recognizer for `backend`, optionally reusing a loaded model."""
    if backend == "vosk":
        return _VoskRecognizer(model_path, sample_rate, model=shared)
    if backend == "whisper":
        return _FasterWhisperRecognizer(
            model_path,
            sample_rate,
            language=language,
            whisper_model=shared,
            allowed_languages=allowed_languages,
            hotwords=hotwords,
        )
    if backend == "whispercpp":
        return _WhisperCppRecognizer(model_path, sample_rate)
    raise ValueError(f"Unsupported ASR backend: {backend}")


class StreamingASR:
    def __init__(
        self,
        sample_rate: int,
        chunk_size: int,
        vad: VoiceActivityDetector,
        model_path: str,
        backend: str = "vosk",
        endpoint_silence_ms: int = 400,
        speech_start_frames: int = 3,
        recognizer: Any | None = None,
        hold_silence_ms: int | None = None,
        language: str = "en",
        allowed_languages: tuple[str, ...] = (),
        early_decode_ms: int = 0,
        partial_interval_ms: int = 0,
        hotwords: str = "",
        echo_canceller: Any | None = None,
        wake_detector: Any | None = None,
        wake_gate: Any | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        # Removes the assistant's own voice from microphone frames.
        self.echo_canceller = echo_canceller
        # While `wake_gate` is asleep, audio only goes to `wake_detector`
        # (an acoustic wake word model); recognition starts once it fires.
        self.wake_detector = wake_detector if wake_gate is not None else None
        self.wake_gate = wake_gate
        self.chunk_size = chunk_size
        self.vad = vad
        self.backend = backend
        self.endpoint_silence_s = max(0.01, endpoint_silence_ms / 1000.0)
        # Longer wait used when the words so far look unfinished.
        self.hold_silence_s = max(
            self.endpoint_silence_s,
            (hold_silence_ms if hold_silence_ms is not None else endpoint_silence_ms * 2.5) / 1000.0,
        )
        self.speech_start_frames = max(1, speech_start_frames)
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=256)
        self._loop: asyncio.AbstractEventLoop | None = None
        # When muted, microphone audio is discarded (half-duplex operation
        # while the assistant is speaking).
        self.muted = False

        self._stabilizer = PartialTranscriptStabilizer()
        self._preroll: deque[bytes] = deque(maxlen=_PREROLL_FRAMES)
        self._speech_buffer = bytearray()
        self._in_speech = False
        self._speech_start_sent = False
        self._consecutive_speech = 0
        # Endpointing runs on the audio clock (seconds of audio processed), so
        # bursty network delivery or a slow consumer cannot distort it.
        self._frame_s = vad.config.frame_ms / 1000.0
        self._audio_clock = 0.0
        self._last_speech_audio = 0.0
        self._last_speech_ts = 0.0
        self._last_partial = ""
        self._latest_words = ""
        # A fixed language is reported on every transcript; with "auto" the
        # recognizer says what it detected.
        self.language = None if language == AUTO else language

        self._rec = recognizer if recognizer is not None else load_recognizer(
            backend,
            model_path,
            sample_rate,
            language=language,
            allowed_languages=allowed_languages,
            hotwords=hotwords,
        )

        # Recognizers that decode a whole utterance at once (Whisper) can be
        # run ahead of the endpoint: once the user pauses, the utterance so
        # far is decoded in the background, and if they do not speak again
        # the endpoint reuses that result instead of starting a decode.
        # During long speech, periodic decodes give live partial transcripts.
        can_decode_early = not self._rec.streaming and callable(getattr(self._rec, "decode", None))
        self.early_decode_s = early_decode_ms / 1000.0 if can_decode_early and early_decode_ms > 0 else None
        self.partial_interval_s = (
            partial_interval_ms / 1000.0 if can_decode_early and partial_interval_ms > 0 else None
        )
        self._executor: ThreadPoolExecutor | None = None
        self._snapshot: _Snapshot | None = None
        self._last_snapshot_audio = 0.0

    def _mic_callback(
        self,
        indata: Any,
        frames: int,
        _time_info: Any,
        status: Any,
    ) -> None:
        if status:
            logger.debug("Mic callback status: %s", status)
        if frames <= 0:
            return
        raw = bytes(indata)
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._put_audio_frame, raw)

    def _put_audio_frame(self, raw: bytes) -> None:
        try:
            self._audio_queue.put_nowait(raw)
        except asyncio.QueueFull:
            logger.warning("ASR audio queue full; dropping frame")

    def reset(self) -> None:
        """Forget the utterance in progress."""
        self._speech_buffer.clear()
        self._preroll.clear()
        self._in_speech = False
        self._speech_start_sent = False
        self._consecutive_speech = 0
        self._last_partial = ""
        self._latest_words = ""
        self._drop_snapshot()
        self._stabilizer = PartialTranscriptStabilizer()
        self.vad.reset()
        self._rec.reset()

    def close(self) -> None:
        """Stop the background decoder thread."""
        self._drop_snapshot()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    @property
    def in_utterance(self) -> bool:
        return self._in_speech

    def process_frame(self, frame: bytes) -> list[ASREvent]:
        """Feed one VAD-sized frame and return any events it produced."""
        if len(frame) != self.vad.frame_bytes:
            return []

        if self.echo_canceller is not None:
            # Even while muted, so its reference stays aligned with the microphone.
            frame = self.echo_canceller.process(frame)

        self._audio_clock += self._frame_s
        now_ms = int(time.time() * 1000)
        events: list[ASREvent] = []

        if self.muted:
            if self._in_speech or self._speech_buffer:
                self.reset()
            return events

        if self.wake_detector is not None and not self._in_speech and not self.wake_gate.awake:
            if self.wake_detector.detect(frame):
                self.wake_gate.wake()
                self._preroll.clear()
                self.vad.reset()
                events.append(ASREvent("wake", "", 1.0, now_ms))
            return events

        speech = self.vad.is_speech(frame)
        self._consecutive_speech = self._consecutive_speech + 1 if speech else 0

        if not self._in_speech:
            if not speech:
                self._preroll.append(frame)
                return events
            self._in_speech = True
            self._last_snapshot_audio = self._audio_clock
            for early in self._preroll:
                self._feed(early)
            self._preroll.clear()

        # Once an utterance has started, feed every frame (including short
        # pauses between words) so the recognizer sees continuous audio.
        self._feed(frame)

        if speech:
            self._last_speech_audio = self._audio_clock
            self._last_speech_ts = time.perf_counter()
            if not self._speech_start_sent and self._consecutive_speech >= self.speech_start_frames:
                self._speech_start_sent = True
                events.append(ASREvent("speech_start", "", 0.0, now_ms))

            if self._rec.streaming:
                partial, conf = self._rec.partial_result()
                stable = self._stabilizer.update(partial)
                self._latest_words = partial
                if stable and stable != self._last_partial:
                    self._last_partial = stable
                    events.append(ASREvent("partial", stable, conf, now_ms))
            elif self.partial_interval_s is not None:
                self._collect_snapshot(events, now_ms)
                idle = self._snapshot is None or self._snapshot.future.done()
                if idle and self._audio_clock - self._last_snapshot_audio >= self.partial_interval_s:
                    self._take_snapshot()
            return events

        silence = self._audio_clock - self._last_speech_audio
        if self.early_decode_s is not None or self.partial_interval_s is not None:
            # A finished decode tells the endpoint what was said so far, so the
            # longer hold for unfinished sentences works with Whisper too.
            self._collect_snapshot(events, now_ms)
        if (
            self.early_decode_s is not None
            and silence >= self.early_decode_s
            and (self._snapshot is None or self._snapshot.speech_audio != self._last_speech_audio)
        ):
            self._take_snapshot()

        # The epsilon keeps float drift in the frame clock from ending a frame early.
        if silence > self._required_silence() + 1e-6:
            events.extend(self.finalize())

        return events

    def _take_snapshot(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vaani-asr-early")
        # A decode that has not started yet is superseded by this one.
        self._drop_snapshot()
        future = self._executor.submit(self._rec.decode, bytes(self._speech_buffer))
        self._snapshot = _Snapshot(future, self._last_speech_audio)
        self._last_snapshot_audio = self._audio_clock

    def _drop_snapshot(self) -> None:
        if self._snapshot is not None:
            self._snapshot.future.cancel()
            self._snapshot = None

    def _collect_snapshot(self, events: list[ASREvent], now_ms: int) -> None:
        snap = self._snapshot
        if snap is None or snap.reported or not snap.future.done() or snap.future.cancelled():
            return
        snap.reported = True
        if snap.future.exception() is not None:
            logger.debug("Early decode failed", exc_info=snap.future.exception())
            return
        text, conf, language = snap.future.result()
        text = text.strip()
        if text and text != self._last_partial:
            self._latest_words = text
            self._last_partial = text
            events.append(ASREvent("partial", text, conf, now_ms, language=language or self.language))

    def _snapshot_result(self) -> tuple[str, float, str | None] | None:
        """The early decode, if it covers the whole utterance (no speech after it)."""
        snap = self._snapshot
        if snap is None or snap.speech_audio != self._last_speech_audio or snap.future.cancelled():
            return None
        try:
            # Usually already finished; otherwise it has a head start on a fresh decode.
            return snap.future.result()
        except Exception:
            logger.debug("Early decode failed; decoding again", exc_info=True)
            return None

    def _required_silence(self) -> float:
        if looks_unfinished(self._latest_words):
            return self.hold_silence_s
        return self.endpoint_silence_s

    def finalize(self) -> list[ASREvent]:
        """End the current utterance now (e.g. when the audio stream closes)."""
        events: list[ASREvent] = []
        if self._in_speech:
            early = self._snapshot_result()
            if early is not None:
                final_text, conf, language = early
                if final_text.strip() and language and hasattr(self._rec, "last_language"):
                    self._rec.last_language = language
            else:
                final_text, conf = self._rec.final_result(bytes(self._speech_buffer))
                language = getattr(self._rec, "last_language", None)
            if final_text.strip():
                events.append(
                    ASREvent(
                        "final",
                        final_text.strip(),
                        conf,
                        int(time.time() * 1000),
                        speech_end_ts=self._last_speech_ts,
                        language=language or self.language,
                    )
                )
        self.reset()
        return events

    def _feed(self, frame: bytes) -> None:
        self._speech_buffer.extend(frame)
        self._rec.accept_waveform(frame)

    async def stream_events(self) -> AsyncIterator[ASREvent]:
        import sounddevice as sd

        self._loop = asyncio.get_running_loop()
        frame_bytes = self.vad.frame_bytes

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_size,
            callback=self._mic_callback,
        ):
            while True:
                chunk = await self._audio_queue.get()
                for i in range(0, len(chunk) - frame_bytes + 1, frame_bytes):
                    frame = chunk[i : i + frame_bytes]
                    # Recognizers can block (Whisper decodes a whole utterance
                    # at the endpoint), so keep them off the event loop.
                    for event in await asyncio.to_thread(self.process_frame, frame):
                        yield event


def build_default_vad(sample_rate: int = 16_000, frame_ms: int = 30, aggressiveness: int = 2) -> VoiceActivityDetector:
    return VoiceActivityDetector(
        VADConfig(sample_rate=sample_rate, frame_ms=frame_ms, aggressiveness=aggressiveness, mode="webrtc")
    )
