"""Wake word: Vaani only answers when addressed ("Hey Vaani, ...").

Two ways to listen for it, which can be combined:

- **Phrase** (`WAKE_WORD="hey vaani"`): every utterance is transcribed and
  answered only if it starts with the phrase. Needs no extra model and works
  for any name, but runs speech recognition on everything said nearby.
- **Acoustic model** (`WAKE_WORD_MODEL`): an openWakeWord model listens for the
  wake word itself and speech recognition only runs after it fires, which
  uses far less CPU. Pretrained models exist for "alexa", "hey jarvis",
  "hey mycroft" and "hey rhasspy"; a custom "hey vaani" model can be trained
  with openWakeWord and given as a path.

After the wake word, and after every reply, Vaani keeps listening for a
follow-up window so a conversation does not need the wake word every turn.
"""
from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Callable
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# How closely the start of an utterance must match the wake phrase. Whisper
# spells names inconsistently ("Hey Vani", "Hey, Vaani."), so this is fuzzy.
MATCH_THRESHOLD = 0.8
# How many words may come before the wake phrase ("okay hey vaani ...").
_MAX_LEADING_WORDS = 2

_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?")


def _norm(token: str) -> str:
    return "".join(_WORD.findall(token.lower()))


def parse_wake_phrases(raw: str) -> tuple[str, ...]:
    """Comma-separated phrases: "hey vaani, hey vani, ok vaani"."""
    phrases = (" ".join(_norm(t) for t in part.split() if _norm(t)) for part in raw.split(","))
    return tuple(dict.fromkeys(p for p in phrases if p))


class WakeWordGate:
    """Decides which utterances are meant for the assistant.

    Asleep, only utterances starting with a wake phrase get through (minus the
    phrase). Awake, everything gets through. `wake()` opens the follow-up
    window and `hold()` keeps the gate open until the next `wake()`.
    """

    def __init__(
        self,
        phrases: tuple[str, ...] = (),
        follow_up_s: float = 8.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.phrases = tuple(p.split() for p in phrases)
        self.follow_up_s = follow_up_s
        self._clock = clock
        self._awake_until = -math.inf

    @property
    def awake(self) -> bool:
        return self._clock() < self._awake_until

    def wake(self) -> None:
        """Listen without the wake word for the follow-up window."""
        self._awake_until = self._clock() + self.follow_up_s

    def hold(self) -> None:
        """Stay awake until the next `wake()` or `sleep()` (while replying)."""
        self._awake_until = math.inf

    def sleep(self) -> None:
        self._awake_until = -math.inf

    def match(self, text: str) -> str | None:
        """The words after the wake phrase if `text` starts with one, else None.

        Returns "" when the utterance is only the wake phrase.
        """
        tokens = text.split()
        # Positions of real words: punctuation-only tokens such as "-" are skipped.
        words = [(i, _norm(t)) for i, t in enumerate(tokens) if _norm(t)]
        best: tuple[float, int] | None = None
        for phrase in self.phrases:
            target = " ".join(phrase)
            for start in range(min(_MAX_LEADING_WORDS + 1, len(words))):
                for length in {max(1, len(phrase) - 1), len(phrase), len(phrase) + 1}:
                    if start + length > len(words):
                        continue
                    candidate = " ".join(w for _, w in words[start : start + length])
                    score = SequenceMatcher(None, candidate, target).ratio()
                    if score >= MATCH_THRESHOLD and (best is None or score > best[0]):
                        best = (score, start + length)
        if best is None:
            return None
        end = best[1]
        if end >= len(words):
            return ""
        rest = " ".join(tokens[words[end][0] :])
        return rest.lstrip(",.!?;: ").strip()

    def hotwords(self) -> str:
        """Words for the recognizer to prefer, so it spells the name the same way every time."""
        return " ".join(dict.fromkeys(w for phrase in self.phrases for w in phrase))


class OpenWakeWordDetector:
    """Spots a wake word in raw audio with openWakeWord (16 kHz, 16-bit mono)."""

    # openWakeWord scores audio in 80 ms steps.
    STEP_SAMPLES = 1280

    def __init__(self, model: str, threshold: float = 0.5, engine=None) -> None:
        self.threshold = threshold
        self.name = Path(model).stem if model.endswith((".onnx", ".tflite")) else model
        self._model = engine if engine is not None else _load_openwakeword(model)
        self._buffer = np.zeros(0, dtype=np.int16)

    def detect(self, frame: bytes) -> bool:
        """Feed one frame; True when the wake word was just heard."""
        self._buffer = np.concatenate([self._buffer, np.frombuffer(frame, dtype=np.int16)])
        heard = False
        while len(self._buffer) >= self.STEP_SAMPLES:
            step, self._buffer = self._buffer[: self.STEP_SAMPLES], self._buffer[self.STEP_SAMPLES :]
            scores = self._model.predict(step)
            if not heard and max(scores.values(), default=0.0) >= self.threshold:
                heard = True
        if heard:
            # Forget the audio that triggered it, so it does not fire twice.
            self._model.reset()
            self._buffer = np.zeros(0, dtype=np.int16)
        return heard


def _load_openwakeword(model: str):
    try:
        from openwakeword import utils  # type: ignore
        from openwakeword.model import Model  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "WAKE_WORD_MODEL needs openWakeWord. Install it with `pip install 'voice-assistant[wakeword]'`."
        ) from exc
    is_path = model.endswith((".onnx", ".tflite"))
    if is_path and not Path(model).expanduser().exists():
        raise RuntimeError(f"WAKE_WORD_MODEL file not found: {model}")
    framework = "tflite" if model.endswith(".tflite") else "onnx"
    # Fetches the shared feature models, and the named model if it is a pretrained one.
    utils.download_models(model_names=[] if is_path else [model])
    path = str(Path(model).expanduser()) if is_path else model
    return Model(wakeword_models=[path], inference_framework=framework)
