"""Offline wake-word detection using openWakeWord."""

from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

logger = logging.getLogger(__name__)

# Audio capture constants (openWakeWord expects 16 kHz mono audio).
_SAMPLE_RATE: int = 16_000
_CHUNK_SECONDS: float = 0.08  # duration of each recorded chunk
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
        try:
            self.model = Model(
                wakeword_models=[model_name],
                inference_framework="onnx"
            )
        except Exception as e:
            import openwakeword
            if model_name in openwakeword.MODELS:
                logger.info("Built-in model '%s' not found locally. Attempting to download...", model_name)
                try:
                    import openwakeword.utils
                    openwakeword.utils.download_models([model_name])
                    self.model = Model(
                        wakeword_models=[model_name],
                        inference_framework="onnx"
                    )
                    return
                except Exception as download_error:
                    raise RuntimeError(f"Failed to download built-in model '{model_name}': {download_error}") from download_error
            
            raise ValueError(
                f"Failed to load wake-word model '{model_name}'. "
                "If this is a custom model, ensure the file path is correct. "
                "If you want to use a built-in model, choose from: "
                f"{list(openwakeword.MODELS.keys())}"
            ) from e

    def listen(self) -> bool:
        """Block until the wake word is detected.

        Records audio continuously and runs inference on each 80ms window.
        Returns ``True`` once the model's confidence exceeds the threshold.
        """
        logger.info("Listening for wake word: %s", self.model_name)

        with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="int16") as stream:
            while True:
                # Read a chunk of audio (1280 samples = 80ms)
                raw, _ = stream.read(_CHUNK_FRAMES)
                
                # Normalize to float32 in range [-1, 1]
                audio = raw.flatten().astype(np.float32) / 32768.0

                prediction = self.model.predict(audio)
                # Get the score for the loaded model (robust to custom paths)
                score: float = next(iter(prediction.values()), 0)

                if score > _DETECTION_THRESHOLD:
                    logger.info("Wake word detected! (score=%.3f)", score)
                    return True
