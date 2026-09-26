"""Lightweight intent classifier scaffold for multilingual / code-mixed NLU.

This implements a minimal rule-based `SimpleIntentClassifier` that can be
used as a starting point for intent classification. It intentionally has
no runtime dependencies so it can be iterated on quickly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class IntentClassifier(Protocol):
    def classify(self, text: str) -> dict[str, object]:
        """Return a small dict with at least `intent` and `confidence` keys."""


@dataclass
class _IntentResult:
    intent: str
    confidence: float
    extras: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            **self.extras,
        }


class SimpleIntentClassifier:
    """A lightweight rule-based classifier supporting English + Hindi heuristics.

    Features:
    - Detects presence of Devanagari characters.
    - Handles simple Hindi-English code-mixed queries.
    - Uses keyword matching for demo intents.
    - Returns a dict with `intent`, `confidence`, and `lang`.

    This intentionally stays dependency-free for easy iteration and testing.
    """

    INTENT_KEYWORDS = {
        "greeting": [
            "hello",
            "hi",
            "hey",
            "namaste",
            "namaskar",
        ],
        "play_music": [
            "play",
            "play music",
            "song",
            "gaana",
            "play the song",
            "gaana chala do",
            "music chala do",
            "song baja do",
        ],
        "stop": [
            "stop",
            "pause",
            "ruk",
            "band karo",
            "music band karo",
        ],
        "weather": [
            "weather",
            "kaa mausam",
            "mausam",
            "mosam",
            "weather batao",
            "mausam batao",
        ],
        "time": [
            "what time is it",
            "whats the time",
            "what is the time",
            "tell me the time",
            "current time",
            "time kya hai",
            "kitne baje",
            "kitne baje hai",
        ],
        "date": [
            "whats the date",
            "what is the date",
            "todays date",
            "what day is it",
            "what day is today",
            "aaj ki date",
            "aaj kya date hai",
        ],
    }

    # Keywords too generic to trigger an intent on their own ("how do I play
    # chess" is not a music request).
    WEAK_KEYWORDS = {"play"}

    def _normalize(self, text: str) -> str:
        """Normalize text for lightweight matching."""
        text = text.lower()
        text = re.sub(r"['’]", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        return " ".join(text.split())

    def _contains_devanagari(self, text: str) -> bool:
        for ch in text:
            if "\u0900" <= ch <= "\u097F":
                return True
        return False

    def classify(self, text: str) -> dict[str, object]:
        if not text or not text.strip():
            return {"intent": "none", "confidence": 0.0}

        lowered = self._normalize(text)
        devanagari = self._contains_devanagari(text)

        # simple keyword scoring
        scores: dict[str, int] = {
            k: 0 for k in self.INTENT_KEYWORDS
        }
        strong: dict[str, bool] = {k: False for k in self.INTENT_KEYWORDS}

        padded = f" {lowered} "
        for intent, keys in self.INTENT_KEYWORDS.items():
            for kw in keys:
                if f" {kw} " in padded:
                    scores[intent] += 1
                    if kw not in self.WEAK_KEYWORDS:
                        strong[intent] = True

        best_intent = max(scores, key=lambda k: scores[k])
        best_score = scores[best_intent]
        lang = "hi" if devanagari else "en"
        extras: dict[str, object] = {"lang": lang, "word_count": len(lowered.split())}

        if best_score == 0:
            # No keyword matched. Devanagari text is still only "unknown":
            # it must reach the LLM rather than a canned reply.
            extras["lang"] = lang if devanagari else "und"
            return _IntentResult("unknown", 0.2, extras).to_dict()

        # confidence scales with count; clamp to [0.2, 0.95]
        confidence = min(0.95, 0.2 + 0.3 * best_score)
        if not strong[best_intent]:
            confidence = min(confidence, 0.35)

        return _IntentResult(
            best_intent,
            confidence,
            extras,
        ).to_dict()
