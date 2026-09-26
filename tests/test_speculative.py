from __future__ import annotations

from voice_assistant.llm.speculative import SpeculativeConfig, SpeculativeDecoder


class MockModel:
    def __init__(self, seq: list[int], favor: set[int]) -> None:
        self.seq = seq
        self.favor = favor

    def generate_k(self, prompt_tokens: list[int], k: int, temperature: float) -> list[int]:
        _ = (prompt_tokens, temperature)
        return self.seq[:k]

    def logprob_next(self, prompt_tokens: list[int], token: int, temperature: float) -> float:
        _ = (prompt_tokens, temperature)
        return -0.1 if token in self.favor else -3.0


def test_speculative_acceptance_rate_and_fallback() -> None:
    draft = MockModel([1, 2, 3, 4], favor={1, 2, 3, 4})
    target = MockModel([1, 2, 8, 9], favor={1, 2})

    dec = SpeculativeDecoder(draft=draft, target=target, config=SpeculativeConfig(k=4, acceptance_floor=0.4))
    out, stats = dec.decode(prompt_tokens=[10], max_new_tokens=6)

    assert len(out) > 0
    assert 0.0 <= stats.accepted_ratio <= 1.0
    assert stats.fallback_steps >= 0


class RecordingModel(MockModel):
    def __init__(self, seq: list[int], favor: set[int]) -> None:
        super().__init__(seq, favor)
        self.contexts: list[list[int]] = []

    def logprob_next(self, prompt_tokens: list[int], token: int, temperature: float) -> float:
        self.contexts.append(list(prompt_tokens))
        return super().logprob_next(prompt_tokens, token, temperature)


def test_speculative_scores_tokens_against_correct_context() -> None:
    draft = MockModel([1, 2, 3], favor={1, 2, 3})
    target = RecordingModel([1, 2, 3], favor={1, 2, 3})

    out, _ = SpeculativeDecoder(draft=draft, target=target).decode(prompt_tokens=[10], max_new_tokens=6)

    assert out == [1, 2, 3, 1, 2, 3]
    # Each scored context is the prompt plus exactly the tokens accepted so far.
    for ctx, expected_len in zip(target.contexts, range(1, 7), strict=True):
        assert ctx == ([10] + out)[:expected_len]


def test_speculative_never_exceeds_max_new_tokens() -> None:
    draft = MockModel([5, 6, 7, 8], favor={5})
    target = MockModel([9], favor=set())

    out, _ = SpeculativeDecoder(draft=draft, target=target).decode(prompt_tokens=[1], max_new_tokens=3)

    assert len(out) == 3
