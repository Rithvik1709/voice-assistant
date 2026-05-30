from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class NextTokenModel(Protocol):
    def generate_k(self, prompt_tokens: list[int], k: int, temperature: float) -> list[int]: ...

    def logprob_next(self, prompt_tokens: list[int], token: int, temperature: float) -> float: ...


@dataclass(slots=True)
class SpeculativeConfig:
    k: int = 4
    acceptance_floor: float = 0.35
    temperature: float = 0.7


@dataclass(slots=True)
class SpeculativeStats:
    accepted: int = 0
    rejected: int = 0
    fallback_steps: int = 0
    accepted_ratio: float = 0.0


@dataclass(slots=True)
class SpeculativeDecoder:
    draft: NextTokenModel
    target: NextTokenModel
    config: SpeculativeConfig = field(default_factory=SpeculativeConfig)

    def decode(self, prompt_tokens: list[int], max_new_tokens: int) -> tuple[list[int], SpeculativeStats]:
        produced: list[int] = []
        stats = SpeculativeStats()

        while len(produced) < max_new_tokens:
            current_prompt = prompt_tokens + produced
            proposal = self.draft.generate_k(current_prompt, self.config.k, self.config.temperature)
            if not proposal:
                break

            accepted_now = 0
            for token in proposal:
                if len(produced) >= max_new_tokens:
                    break
                ratio = self._acceptance_ratio(current_prompt + produced, token)
                if ratio >= 1.0:
                    produced.append(token)
                    accepted_now += 1
                    continue

                if ratio >= self.config.acceptance_floor:
                    produced.append(token)
                    accepted_now += 1
                else:
                    stats.rejected += 1
                    break

            stats.accepted += accepted_now

            block = len(proposal)
            if block > 0 and (accepted_now / block) < self.config.acceptance_floor:
                stats.fallback_steps += 1
                greedy = self.target.generate_k(prompt_tokens + produced, 1, temperature=0.0)
                if not greedy:
                    break
                produced.extend(greedy)

            if accepted_now == 0 and not proposal:
                break

        total = stats.accepted + stats.rejected
        stats.accepted_ratio = (stats.accepted / total) if total else 1.0
        return produced, stats

    def _acceptance_ratio(self, prompt_tokens: list[int], token: int) -> float:
        draft_lp = self.draft.logprob_next(prompt_tokens, token, self.config.temperature)
        target_lp = self.target.logprob_next(prompt_tokens, token, self.config.temperature)
        ratio = min(1.0, pow(2.718281828, target_lp - draft_lp))
        return max(0.0, ratio)

