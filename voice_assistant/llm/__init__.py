"""LLM components."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from voice_assistant.benchmark import BenchmarkTracker
    from voice_assistant.config import Settings


def create_llm(settings: Settings, bench: BenchmarkTracker | None = None) -> Any:
    """Build the LLM client selected by the settings (mock, llama.cpp, or OpenAI-compatible)."""
    from voice_assistant.config import mock_models_enabled

    if mock_models_enabled():
        from voice_assistant.mocks import MockLLMClient

        return MockLLMClient()

    if settings.llm_backend == "openai":
        from voice_assistant.llm.openai_compat import OpenAICompatConfig, OpenAICompatLLMClient

        return OpenAICompatLLMClient(
            OpenAICompatConfig(
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            ),
            bench=bench,
        )

    from voice_assistant.llm.client import LLMConfig, StreamingLLMClient

    return StreamingLLMClient(
        LLMConfig(
            model_path=settings.model_path,
            n_ctx=settings.llm_context_size,
            n_gpu_layers=settings.n_gpu_layers,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        ),
        bench=bench,
    )
