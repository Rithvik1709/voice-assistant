"""Lightweight intent classifier scaffold for multilingual / code-mixed NLU.

This implements a minimal rule-based `SimpleIntentClassifier` that can be
used as a starting point for intent classification. It intentionally has
no runtime dependencies so it can be iterated on quickly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol


class IntentClassifier(Protocol):
    def classify(self, text: str) -> Dict[str, object]:
        """Return a small dict with at least `intent` and `confidence` keys."""


@dataclass
class _IntentResult:
    intent: str
    confidence: float
    extras: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {"intent": self.intent, "confidence": self.confidence, **self.extras}


class SimpleIntentClassifier:
    """A very small rule-based classifier supporting English + Hindi heuristics.

    - Detects presence of Devanagari characters to bias toward Hindi intents.
    - Uses keyword matching for a few demo intents: `greeting`, `play_music`, `stop`, `weather`.
    - Returns a dict with `intent` and `confidence`.
    This is intentionally simple to enable quick debugging and unit tests.
    """

    INTENT_KEYWORDS = {
        "greeting": ["hello", "hi", "hey", "namaste", "namaskar"],
        "play_music": ["play", "play music", "song", "gaana", "play the song"],
        "stop": ["stop", "pause", " ruk "],
        "weather": ["weather", "kaa mausam", "mausam", "how's the weather", "mosam"],
    }

    def _contains_devanagari(self, text: str) -> bool:
        for ch in text:
            if "\u0900" <= ch <= "\u097F":
                return True
        return False

    def classify(self, text: str) -> Dict[str, object]:
        if not text or not text.strip():
            return {"intent": "none", "confidence": 0.0}

        lowered = text.lower()
        devanagari = self._contains_devanagari(text)

        # simple keyword scoring
        scores: dict[str, int] = {k: 0 for k in self.INTENT_KEYWORDS}
        for intent, keys in self.INTENT_KEYWORDS.items():
            for kw in keys:
                if kw in lowered:
                    scores[intent] += 1

        best_intent = max(scores, key=lambda k: scores[k])
        best_score = scores[best_intent]

        if best_score == 0:
            # fallback: greet if devanagari present and contains hi keywords
            if devanagari:
                return _IntentResult("greeting", 0.5, {"lang": "hi"}).to_dict()
            return _IntentResult("unknown", 0.2, {"lang": "und"}).to_dict()

        # confidence scales with count; clamp to [0.2, 0.95]
        confidence = min(0.95, 0.2 + 0.3 * best_score)
        extras = {"lang": "hi" if devanagari else "en"}
        return _IntentResult(best_intent, confidence, extras).to_dict()
