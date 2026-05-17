from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
import numpy as np

# Move all imports to the top
try:
    import torch
except ImportError:
    torch = None

try:
    import webrtcvad  # type: ignore
except Exception:  # pragma: no cover
    webrtcvad = None

try:
    from silero_vad import VADIterator, load_silero_vad
except Exception:  # pragma: no cover
    VADIterator = None  
    load_silero_vad = None  


@dataclass(slots=True)
class VADConfig:
    sample_rate: int = 16_000
    frame_ms: int = 30  # Default kept at 30ms for WebRTC compatibility
    aggressiveness: int = 2
    speech_frames_trigger: int = 3
    threshold: float = 0.015  # Safe production default
    mode: str = "webrtc"  # webrtc | silero | energy


class VoiceActivityDetector:
    def __init__(self, config: VADConfig) -> None:
        self.config = config
        self.frame_bytes = int(config.sample_rate * config.frame_ms / 1000) * 2
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        
        self._vad = None
        self._model = None
        self._silero_iter = None
        self._silero_speech_active = False

        if config.mode == "webrtc":
            if webrtcvad is None:
                raise RuntimeError("webrtcvad is not installed")
            if config.frame_ms not in [10, 20, 30]:
                raise ValueError("WebRTC VAD only supports 10, 20, or 30ms frames.")
            self._vad = webrtcvad.Vad(config.aggressiveness)
            
        elif config.mode == "silero":
            if load_silero_vad is None or VADIterator is None or torch is None:
                raise RuntimeError("silero-vad or torch is not installed")
            if config.frame_ms not in [32, 64, 96] and config.sample_rate == 16000:
                # Silero can handle others, but 32ms (512 samples) is standard for 16kHz
                pass
            
            self._model = load_silero_vad()
            self._silero_iter = VADIterator(
                self._model, 
                threshold=config.threshold, 
                sampling_rate=config.sample_rate
            )

    def is_speech(self, pcm16: bytes) -> bool:
        if len(pcm16) != self.frame_bytes:
            return False

        if self.config.mode == "energy":
            audio_array = np.frombuffer(pcm16, dtype=np.int16)
            rms = np.sqrt(np.mean(np.square(audio_array, dtype=np.float32)))
            return bool(rms > 450)

        if self.config.mode == "silero":
            assert self._silero_iter is not None
            # Normalize to float32
            audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Use the stateful iterator ONCE per frame (fixes double-call & state issues)
            res = self._silero_iter(audio)
            if res is not None:
                if "start" in res:
                    self._silero_speech_active = True
                if "end" in res:
                    self._silero_speech_active = False
            
            return self._silero_speech_active

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
        """Consolidated single reset method."""
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._silero_speech_active = False
        if self._silero_iter is not None:
            self._silero_iter.reset_states()