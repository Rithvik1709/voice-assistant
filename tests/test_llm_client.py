from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Mock llama_cpp module since it may not be installed in the environment
mock_llama_module = MagicMock()
sys.modules['llama_cpp'] = mock_llama_module

import asyncio
import pytest
from voice_assistant.llm.client import LLMConfig, StreamingLLMClient
from voice_assistant.benchmark import BenchmarkTracker

@pytest.mark.asyncio
async def test_stream_tokens_with_messages_list() -> None:
    config = LLMConfig(model_path="fake_model.gguf")

    # Mock Llama class
    mock_llama_instance = MagicMock()
    mock_llama_module.Llama.return_value = mock_llama_instance

    # Mock stream response
    mock_stream = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
        {"choices": [{"delta": {"content": "!"}}]},
    ]
    mock_llama_instance.create_chat_completion.return_value = mock_stream

    client = StreamingLLMClient(config)
    out_queue = asyncio.Queue()
    messages = [{"role": "user", "content": "Hi"}]

    # Call stream_tokens
    resp = await client.stream_tokens(messages, out_queue)

    # Assert results
    assert resp == "Hello world!"
    assert out_queue.qsize() == 3
    assert await out_queue.get() == "Hello"
    assert await out_queue.get() == " world"
    assert await out_queue.get() == "!"

    mock_llama_instance.create_chat_completion.assert_called_once_with(
        messages=messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        stream=True,
    )

@pytest.mark.asyncio
async def test_stream_tokens_with_string_prompt() -> None:
    config = LLMConfig(model_path="fake_model.gguf")

    mock_llama_instance = MagicMock()
    mock_llama_module.Llama.return_value = mock_llama_instance
    mock_stream = [
        {"choices": [{"delta": {"content": "Test"}}]},
    ]
    mock_llama_instance.create_chat_completion.return_value = mock_stream

    client = StreamingLLMClient(config)
    out_queue = asyncio.Queue()

    resp = await client.stream_tokens("Hi", out_queue)

    assert resp == "Test"
    mock_llama_instance.create_chat_completion.assert_called_once_with(
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        stream=True,
    )

@pytest.mark.asyncio
async def test_stream_tokens_with_benchmark() -> None:
    config = LLMConfig(model_path="fake_model.gguf")
    bench = BenchmarkTracker()

    mock_llama_instance = MagicMock()
    mock_llama_module.Llama.return_value = mock_llama_instance
    mock_stream = [
        {"choices": [{"delta": {"content": "Bench"}}]},
    ]
    mock_llama_instance.create_chat_completion.return_value = mock_stream

    client = StreamingLLMClient(config, bench=bench)
    out_queue = asyncio.Queue()

    resp = await client.stream_tokens("Hi", out_queue)

    assert resp == "Bench"
    # Verify benchmarks marked
    assert bench.current.prompt_sent_ts is not None
    assert bench.current.first_token_ts is not None
