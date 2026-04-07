from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class KVCacheManager:
    similarity_threshold: float = 0.55
    _cached_state: bytes | None = None
    _topic_vector: dict[str, float] = field(default_factory=dict)

    def save(self, llm: Any) -> None:
        self._cached_state = llm.save_state()

    def load(self, llm: Any) -> bool:
        if not self._cached_state:
            return False
        llm.load_state(self._cached_state)
        return True

    def clear(self) -> None:
        self._cached_state = None
        self._topic_vector = {}

    def update_topic(self, text: str) -> None:
        self._topic_vector = self._to_bow(text)

    def should_invalidate(self, new_text: str) -> bool:
        if not self._topic_vector:
            return False
        score = self._cosine(self._topic_vector, self._to_bow(new_text))
        return score < self.similarity_threshold

    @staticmethod
    def _to_bow(text: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for raw in text.lower().split():
            token = "".join(ch for ch in raw if ch.isalnum())
            if not token:
                continue
            key = hashlib.md5(token.encode("utf-8")).hexdigest()[:8]
            out[key] = out.get(key, 0.0) + 1.0
        return out

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        keys = set(a) | set(b)
        dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)
