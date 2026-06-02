import asyncio
import pytest
from unittest.mock import MagicMock, patch
from voice_assistant.llm.client import LLMConfig, StreamingLLMClient
from voice_assistant.benchmark import BenchmarkTracker


@pytest.mark.asyncio
async def test_stream_tokens_puts_tokens_in_queue():
    """Regression test for issue #65 — stub returned None, no tokens ever queued."""
    mock_llama = MagicMock()
    mock_llama.create_chat_completion.return_value = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " World"}}]},
    ]

    with patch("voice_assistant.llm.client.Llama", return_value=mock_llama), \
         patch("voice_assistant.llm.client._LLAMA_AVAILABLE", True):

        client = StreamingLLMClient(LLMConfig(model_path="dummy"))
        q = asyncio.Queue()

        result = await client.stream_tokens(
            [{"role": "user", "content": "hi"}], q
        )

        assert result == "Hello World"
        assert q.get_nowait() == "Hello"
        assert q.get_nowait() == " World"
        assert q.empty()


@pytest.mark.asyncio
async def test_stream_tokens_marks_bench_timestamps():
    """prompt_sent_ts and first_token_ts must be recorded when bench is provided."""
    mock_llama = MagicMock()
    mock_llama.create_chat_completion.return_value = [
        {"choices": [{"delta": {"content": "token"}}]},
    ]

    with patch("voice_assistant.llm.client.Llama", return_value=mock_llama), \
         patch("voice_assistant.llm.client._LLAMA_AVAILABLE", True):

        bench = BenchmarkTracker()
        client = StreamingLLMClient(LLMConfig(model_path="dummy"), bench=bench)
        q = asyncio.Queue()

        await client.stream_tokens([{"role": "user", "content": "hi"}], q)

        assert bench.current.prompt_sent_ts is not None
        assert bench.current.first_token_ts is not None


@pytest.mark.asyncio
async def test_stream_tokens_works_without_bench():
    """Generation must run even when bench=None — catches the if active_bench indentation bug."""
    mock_llama = MagicMock()
    mock_llama.create_chat_completion.return_value = [
        {"choices": [{"delta": {"content": "ok"}}]},
    ]

    with patch("voice_assistant.llm.client.Llama", return_value=mock_llama), \
         patch("voice_assistant.llm.client._LLAMA_AVAILABLE", True):

        client = StreamingLLMClient(LLMConfig(model_path="dummy"))
        q = asyncio.Queue()

        result = await client.stream_tokens([{"role": "user", "content": "hi"}], q)

        assert result == "ok"
        assert q.get_nowait() == "ok"
        assert q.empty()