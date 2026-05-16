from dataclasses import dataclass, field


@dataclass(slots=True)
class PartialTranscriptStabilizer:
    min_stable_chars: int = 3
    _prev_partial: str = field(default="", init=False)
    _stable: str = field(default="", init=False)

    def update(self, partial: str) -> str:
        partial = partial.strip()

        if not partial:
            return self._stable

        common = self._common_prefix(self._prev_partial, partial)

        if len(common) >= self.min_stable_chars and len(common) >= len(self._stable):
            self._stable = common

        self._prev_partial = partial

        return self._stable

    @staticmethod
    def _common_prefix(a: str, b: str) -> str:
        out = []

        for ca, cb in zip(a, b, strict=False):
            if ca != cb:
                break

            out.append(ca)

        return "".join(out).strip()