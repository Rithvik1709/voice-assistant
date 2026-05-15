from __future__ import annotations

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
    frame_ms: int = 32  # Recommended to keep at 32 for Silero compatibility
    aggressiveness: int = 2
    speech_frames_trigger: int = 3
    threshold: float = 0.3  
    mode: str = "webrtc"  # webrtc | silero | energy


class VoiceActivityDetector:
    def __init__(self, config: VADConfig) -> None:
        self.config = config
        self.frame_bytes = int(config.sample_rate * config.frame_ms / 1000) * 2
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._vad = None
        self._silero_iter = None
        self._silero_speech_active = False
        self._model = None  # Store the loaded model resource here

        if config.mode == "webrtc":
            if webrtcvad is None:
                raise RuntimeError("webrtcvad is not installed")
            self._vad = webrtcvad.Vad(config.aggressiveness)
        elif config.mode == "silero":
            if load_silero_vad is None or VADIterator is None:
                raise RuntimeError("silero-vad is not installed")
            # Load the model once during initialization
            self._model = load_silero_vad()
            # Create the stateful iterator wrapper
            self._silero_iter = VADIterator(self._model, threshold=config.threshold, sampling_rate=config.sample_rate)

    def reset(self) -> None:
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._silero_speech_active = False
        
        # Completely re-create the iterator to guarantee a 100% fresh state
        if self.config.mode == "silero" and self._model is not None:
            from silero_vad import VADIterator
            self._silero_iter = VADIterator(self._model, threshold=self.config.threshold, sampling_rate=self.config.sample_rate)

    def is_speech(self, pcm16: bytes) -> bool:
        if len(pcm16) != self.frame_bytes:
            return False

        if self.config.mode == "energy":
            audio_array = np.frombuffer(pcm16, dtype=np.int16)
            rms = np.sqrt(np.mean(np.square(audio_array, dtype=np.float32)))
            return bool(rms > 450)

        if self.config.mode == "silero":
            assert self._silero_iter is not None
            audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
            
            import torch
            # Convert numpy array to torch tensor
            tensor_input = torch.from_numpy(audio)
            
            # 1. Get the raw probability directly from the model
            # This bypasses the strict consecutive frame rules of VADIterator
            with torch.no_grad():
                speech_prob = self._model(tensor_input, self.config.sample_rate).item()
            
            # 2. Feed it to the iterator anyway to keep stream tracking alive
            res = self._silero_iter(audio)
            if res is not None:
                if "start" in res:
                    self._silero_speech_active = True
                if "end" in res:
                    self._silero_speech_active = False
            
            # 3. For your frame check, return True if the raw probability crosses your threshold
            return speech_prob >= self.config.threshold

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
        self._silero_speech_active = False
        if self._silero_iter is not None:
            # VADIterator stores the model in self._silero_iter.model
            self._silero_iter.model.reset_states()