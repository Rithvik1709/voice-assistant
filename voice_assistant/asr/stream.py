from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from collections.abc import AsyncIterator
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

logger = logging.getLogger(__name__)

# Frames kept from before the VAD fires, so the first syllable is not clipped.
_PREROLL_FRAMES = 3


@dataclass(slots=True)
class ASREvent:
    type: str  # speech_start | partial | final
    text: str
    confidence: float
    timestamp_ms: int
    speech_end_ts: float | None = None


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
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.vad = vad
        self.backend = backend
        self.endpoint_silence_s = max(0.01, endpoint_silence_ms / 1000.0)
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

        if recognizer is not None:
            self._rec = recognizer
        elif backend == "vosk":
            self._rec = _VoskRecognizer(model_path, sample_rate)
        elif backend == "whispercpp":
            self._rec = _WhisperCppRecognizer(model_path, sample_rate)
        else:
            raise ValueError(f"Unsupported ASR backend: {backend}")

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
        self._stabilizer = PartialTranscriptStabilizer()
        self.vad.reset()
        self._rec.reset()

    @property
    def in_utterance(self) -> bool:
        return self._in_speech

    def process_frame(self, frame: bytes) -> list[ASREvent]:
        """Feed one VAD-sized frame and return any events it produced."""
        if len(frame) != self.vad.frame_bytes:
            return []

        self._audio_clock += self._frame_s
        now_ms = int(time.time() * 1000)
        events: list[ASREvent] = []

        if self.muted:
            if self._in_speech or self._speech_buffer:
                self.reset()
            return events

        speech = self.vad.is_speech(frame)
        self._consecutive_speech = self._consecutive_speech + 1 if speech else 0

        if not self._in_speech:
            if not speech:
                self._preroll.append(frame)
                return events
            self._in_speech = True
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
                if stable and stable != self._last_partial:
                    self._last_partial = stable
                    events.append(ASREvent("partial", stable, conf, now_ms))
            return events

        if self._audio_clock - self._last_speech_audio > self.endpoint_silence_s:
            events.extend(self.finalize())

        return events

    def finalize(self) -> list[ASREvent]:
        """End the current utterance now (e.g. when the audio stream closes)."""
        events: list[ASREvent] = []
        if self._in_speech:
            final_text, conf = self._rec.final_result(bytes(self._speech_buffer))
            if final_text.strip():
                events.append(
                    ASREvent(
                        "final",
                        final_text.strip(),
                        conf,
                        int(time.time() * 1000),
                        speech_end_ts=self._last_speech_ts,
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
                    for event in self.process_frame(chunk[i : i + frame_bytes]):
                        yield event


def build_default_vad(sample_rate: int = 16_000, frame_ms: int = 30, aggressiveness: int = 2) -> VoiceActivityDetector:
    return VoiceActivityDetector(
        VADConfig(sample_rate=sample_rate, frame_ms=frame_ms, aggressiveness=aggressiveness, mode="webrtc")
    )
