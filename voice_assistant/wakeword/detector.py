"""Offline wake-word detection using openWakeWord."""

from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

logger = logging.getLogger(__name__)

# Audio capture constants (openWakeWord expects 16 kHz mono audio).
_SAMPLE_RATE: int = 16_000
_CHUNK_SECONDS: float = 0.8  # duration of each recorded chunk
_CHUNK_FRAMES: int = int(_CHUNK_SECONDS * _SAMPLE_RATE)
_DETECTION_THRESHOLD: float = 0.5  # minimum confidence score to accept


class WakeWordDetector:
    """Listens on the default microphone and returns when the wake word is heard.

    Args:
        model_name: Name of the openWakeWord model to load (e.g. ``"alexa"``).
                    The model is downloaded automatically on first use.
    """

    def __init__(self, model_name: str = "alexa") -> None:
        self.model_name = model_name
        logger.info("Loading wake-word model: %s", model_name)
        self.model = Model(wakeword_models=[model_name])

    def listen(self) -> bool:
        """Block until the wake word is detected.

        Records audio in short chunks and runs inference after each chunk.
        Returns ``True`` once the model's confidence exceeds the threshold.
        """
        logger.info("Listening for wake word: %s", self.model_name)

        while True:
            # Record a fixed-length audio chunk as signed 16-bit integers.
            raw = sd.rec(
                _CHUNK_FRAMES,
                samplerate=_SAMPLE_RATE,
                channels=1,
                dtype="int16",
            )
            sd.wait()

            # openWakeWord expects a 1-D float32 array normalised to [-1, 1].
            audio = np.squeeze(raw).astype(np.float32) / 32768.0

            prediction = self.model.predict(audio)
            score: float = prediction.get(self.model_name, 0)

            if score > _DETECTION_THRESHOLD:
                logger.info("Wake word detected! (score=%.3f)", score)
                return True
