from __future__ import annotations

import audioop
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

try:
    import webrtcvad  # type: ignore
except Exception:  # pragma: no cover - optional runtime import
    webrtcvad = None

try:
    from silero_vad import VADIterator, load_silero_vad
except Exception:  # pragma: no cover - optional runtime import
    VADIterator = None  # type: ignore[assignment]
    load_silero_vad = None  # type: ignore[assignment]


@dataclass(slots=True)
class VADConfig:
    sample_rate: int = 16_000
    frame_ms: int = 30
    aggressiveness: int = 2
    speech_frames_trigger: int = 3
    threshold: float = 0.3  # <-- ADD THIS LINE (Lower threshold catches more speech)
    mode: str = "webrtc"  # webrtc | silero | energy


class VoiceActivityDetector:
    def __init__(self, config: VADConfig) -> None:
        self.config = config
        self.frame_bytes = int(config.sample_rate * config.frame_ms / 1000) * 2
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._vad = None
        self._silero_iter = None

        if config.mode == "webrtc":
            if webrtcvad is None:
                raise RuntimeError("webrtcvad is not installed")
            self._vad = webrtcvad.Vad(config.aggressiveness)
        elif config.mode == "silero":
            if load_silero_vad is None or VADIterator is None:
                raise RuntimeError("silero-vad is not installed")
            model = load_silero_vad()
            self._silero_iter = VADIterator(model, threshold=config.threshold, sampling_rate=config.sample_rate)

    def is_speech(self, pcm16: bytes) -> bool:
        if len(pcm16) != self.frame_bytes:
            return False

        if self.config.mode == "energy":
            rms = audioop.rms(pcm16, 2)
            return rms > 450

        if self.config.mode == "silero":
            assert self._silero_iter is not None
            audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
            return self._silero_iter(audio) is not None

        assert self._vad is not None
        return bool(self._vad.is_speech(pcm16, self.config.sample_rate))

    def detect_barge_in(self, frames: Iterable[bytes]) -> bool:
        for frame in frames:
            if self.is_speech(frame):
                self._consecutive_speech += 1
                self._consecutive_silence = 0
                if self._consecutive_speech >= self.config.speech_frames_trigger:
                    return True
            else:
                self._consecutive_silence += 1
                if self._consecutive_silence > 1:
                    self._consecutive_speech = 0
        return False

    def reset(self) -> None:
        self._consecutive_speech = 0
        self._consecutive_silence = 0
