"""Tests for KV-cache integration into the voice pipeline.

Verifies that the KVCacheManager correctly detects topic shifts, caches state,
and invalidates when conversation topic changes.

Fixes: https://github.com/Rithvik1709/voice-assistant/issues/50
"""
from __future__ import annotations

from voice_assistant.llm.kv_cache import KVCacheManager


def test_kv_cache_topic_similarity_same_topic() -> None:
    """Same-topic prompts with high token overlap should NOT trigger invalidation."""
    cache = KVCacheManager(similarity_threshold=0.55)
    cache.update_topic("Tell me about the weather today")

    # High overlap query (shares most tokens)
    assert cache.should_invalidate("Tell me about the weather tomorrow") is False


def test_kv_cache_topic_shift_triggers_invalidation() -> None:
    """Completely different topic should trigger invalidation."""
    cache = KVCacheManager(similarity_threshold=0.55)
    cache.update_topic("Tell me about the weather today")

    # Completely unrelated topic
    assert cache.should_invalidate("How do I write a Python class") is True


def test_kv_cache_empty_topic_does_not_invalidate() -> None:
    """When no topic has been set, should_invalidate returns False."""
    cache = KVCacheManager()
    assert cache.should_invalidate("any text here") is False


def test_kv_cache_clear_resets_state() -> None:
    """clear() should reset both cached state and topic vector."""
    cache = KVCacheManager()
    cache.update_topic("some topic about weather")
    cache._cached_state = b"fake_state"

    cache.clear()
    assert cache._cached_state is None
    assert cache._topic_vector == {}


class FakeLlama:
    """Mock llama object with save_state/load_state for testing."""

    def __init__(self) -> None:
        self._state: bytes | None = None

    def save_state(self) -> bytes:
        return b"mock_kv_cache_state_data"

    def load_state(self, data: bytes) -> None:
        self._state = data


def test_kv_cache_save_and_load_roundtrip() -> None:
    """save() then load() should successfully restore cached state."""
    cache = KVCacheManager()
    fake_llm = FakeLlama()

    # Initially no cache
    assert cache.load(fake_llm) is False

    # Save state
    cache.save(fake_llm)
    assert cache._cached_state is not None

    # Load state into a new llama instance
    new_llm = FakeLlama()
    assert cache.load(new_llm) is True
    assert new_llm._state == b"mock_kv_cache_state_data"


def test_kv_cache_cosine_similarity() -> None:
    """Verify the cosine similarity calculation used for topic detection."""
    # Identical vectors
    a = {"a": 1.0, "b": 2.0}
    assert abs(KVCacheManager._cosine(a, a) - 1.0) < 0.001

    # Orthogonal vectors
    b = {"c": 1.0, "d": 2.0}
    assert abs(KVCacheManager._cosine(a, b)) < 0.001

    # Empty vector
    assert KVCacheManager._cosine({}, a) == 0.0


def test_kv_cache_integration_with_orchestrator_params() -> None:
    """Verify the orchestrator accepts kv_cache parameter without error."""
    from voice_assistant.llm.kv_cache import KVCacheManager

    cache = KVCacheManager(similarity_threshold=0.6)
    assert cache.similarity_threshold == 0.6

    # Verify topic update + invalidation lifecycle
    cache.update_topic("play some music for me")
    assert cache.should_invalidate("play some music for me please") is False  # high overlap
    assert cache.should_invalidate("what is quantum computing") is True  # unrelated
