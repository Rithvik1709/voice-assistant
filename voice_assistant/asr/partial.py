from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PartialTranscriptStabilizer:
    stable_prefix_words: int = 2

    _prev_partial: str = field(default="", init=False)
    _stable: str = field(default="", init=False)

    def reset(self) -> None:
        self._prev_partial = ""
        self._stable = ""

    def update(self, partial: str) -> str:
        partial = partial.strip()

        prev_words = self._prev_partial.split()
        cur_words = partial.split()

        stable_count = 0

        for pw, cw in zip(prev_words, cur_words):
            if pw == cw:
                stable_count += 1
            else:
                break

        if stable_count >= self.stable_prefix_words:
            self._stable = " ".join(cur_words[:stable_count])

        self._prev_partial = partial

        out = []

        if self._stable:
            out.append(self._stable)

        if stable_count < len(cur_words):
            out.extend(cur_words[stable_count:])

        return " ".join(out)