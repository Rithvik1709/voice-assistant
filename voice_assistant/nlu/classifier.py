from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Protocol

from voice_assistant.skills.registry import get_intent_keywords, dispatch as skill_dispatch


class IntentClassifier(Protocol):
    def classify(self, text: str) -> Dict[str, object]:
        ...


@dataclass
class _IntentResult:
    intent: str
    confidence: float
    extras: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            **self.extras,
        }


class SimpleIntentClassifier:
    def __init__(self) -> None:
        self._keywords: dict[str, list[str]] = {}

    def _load_keywords(self) -> dict[str, list[str]]:
        return get_intent_keywords()

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return " ".join(text.split())

    def _contains_devanagari(self, text: str) -> bool:
        for ch in text:
            if "\u0900" <= ch <= "\u097F":
                return True
        return False

    def classify(self, text: str) -> Dict[str, object]:
        if not text or not text.strip():
            return {"intent": "none", "confidence": 0.0}

        if not self._keywords:
            self._keywords = self._load_keywords()

        lowered = self._normalize(text)
        devanagari = self._contains_devanagari(text)

        scores: dict[str, int] = {k: 0 for k in self._keywords}

        for intent, keys in self._keywords.items():
            for kw in keys:
                if f" {kw} " in f" {lowered} ":
                    scores[intent] += 1

        best_intent = max(scores, key=lambda k: scores[k])
        best_score = scores[best_intent]

        if best_score == 0:
            if devanagari:
                return _IntentResult(
                    "greeting",
                    0.5,
                    {"lang": "hi"},
                ).to_dict()

            return _IntentResult(
                "unknown",
                0.2,
                {"lang": "und"},
            ).to_dict()

        confidence = min(0.95, 0.2 + 0.3 * best_score)

        extras = {
            "lang": "hi" if devanagari else "en",
        }

        return _IntentResult(
            best_intent,
            confidence,
            extras,
        ).to_dict()

    def dispatch(self, intent: str, text: str, entities: dict | None = None) -> dict:
        return skill_dispatch(intent, text, entities)
