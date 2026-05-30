from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd

try:
    from vosk import KaldiRecognizer, Model  # type: ignore
    _VOSK_AVAILABLE = True
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


@dataclass(slots=True)
class ASREvent:
    type: str
    text: str
    confidence: float
    timestamp_ms: int


class _VoskRecognizer:
    def __init__(self, model_path: str, sample_rate: int) -> None:
        if not _VOSK_AVAILABLE:
            raise RuntimeError("vosk is not installed")

        model = Model(model_path)
        self.recognizer = KaldiRecognizer(model, sample_rate)

    def accept_waveform(self, audio_bytes: bytes) -> bool:
        return bool(self.recognizer.AcceptWaveform(audio_bytes))

    def partial_result(self) -> tuple[str, float]:
        data = json.loads(self.recognizer.PartialResult())
        return data.get("partial", ""), float(data.get("confidence", 0.0) or 0.0)

    def final_result(self) -> tuple[str, float]:
        data = json.loads(self.recognizer.FinalResult())
        return data.get("text", ""), float(data.get("confidence", 0.0) or 0.0)


class _WhisperCppRecognizer:
    def __init__(self, model_path: str, sample_rate: int) -> None:
        if not _WHISPER_AVAILABLE:
            raise RuntimeError("whisper_cpp_python is required for whisper.cpp backend")

        self.whisper = Whisper(model_path)
        self.sample_rate = sample_rate

    def transcribe_chunk(self, pcm_bytes: bytes) -> tuple[str, float]:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype("float32") / 32768.0
        segments = self.whisper.transcribe(audio, beam_size=1)
        text = " ".join(seg.text for seg in segments).strip()
        return text, 0.5


class StreamingASR:
    def __init__(
        self,
        sample_rate: int,
        chunk_size: int,
        vad: VoiceActivityDetector,
        model_path: str,
        backend: str = "vosk",
        endpoint_silence_ms: int = 60,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.vad = vad
        self.backend = backend
        self.endpoint_silence_s = max(0.01, endpoint_silence_ms / 1000.0)
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=256)
        self._stabilizer = PartialTranscriptStabilizer()

        if backend == "vosk":
            self._rec = _VoskRecognizer(model_path, sample_rate)
        elif backend == "whispercpp":
            self._rec = _WhisperCppRecognizer(model_path, sample_rate)
        else:
            raise ValueError(f"Unsupported ASR backend: {backend}")

    def _mic_callback(self, indata: Any, frames: int, _time_info: Any, status: sd.CallbackFlags) -> None:
        if status:
            logger.debug("Mic callback status: %s", status)
        if frames <= 0:
            return
        raw = bytes(indata)
        try:
            self._audio_queue.put_nowait(raw)
        except asyncio.QueueFull:
            logger.warning("ASR audio queue full; dropping frame")

    async def stream_events(self) -> AsyncIterator[ASREvent]:
        frame_bytes = self.vad.frame_bytes
        speech_buffer = bytearray()
        in_speech = False
        last_speech_ts = 0.0

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_size,
            callback=self._mic_callback,
        ):
            while True:
                chunk = await self._audio_queue.get()
                if len(chunk) < frame_bytes:
                    continue

                now_ms = int(time.time() * 1000)
                for i in range(0, len(chunk), frame_bytes):
                    frame = chunk[i : i + frame_bytes]
                    if len(frame) != frame_bytes:
                        continue

                    speech = self.vad.is_speech(frame)
                    if speech:
                        in_speech = True
                        last_speech_ts = time.perf_counter()
                        speech_buffer.extend(frame)
                    else:
                        if in_speech:
                            tail = time.perf_counter() - last_speech_ts
                            if tail > self.endpoint_silence_s:
                                final_text, conf = self._flush_final(bytes(speech_buffer))
                                if final_text:
                                    yield ASREvent("final", final_text, conf, now_ms)
                                speech_buffer.clear()
                                self.vad.reset()
                                in_speech = False

    def _flush_final(self, utterance: bytes) -> tuple[str, float]:
        if not utterance:
            return "", 0.0
        if self.backend == "vosk":
            self._rec.accept_waveform(utterance)
            return self._rec.final_result()
        return self._rec.transcribe_chunk(utterance)


def build_default_vad(sample_rate: int = 16_000, frame_ms: int = 30, aggressiveness: int = 2) -> VoiceActivityDetector:
    return VoiceActivityDetector(
        VADConfig(sample_rate=sample_rate, frame_ms=frame_ms, aggressiveness=aggressiveness, mode="webrtc")
    )
