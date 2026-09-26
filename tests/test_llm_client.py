from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from voice_assistant.llm.client import LLMConfig, StreamingLLMClient


class SlowStream:
    """A llama.cpp-style blocking token stream."""

    def __init__(self, tokens: list[str], delay: float) -> None:
        self.tokens = tokens
        self.delay = delay
        self.yielded = 0

    def __iter__(self):
        for tok in self.tokens:
            time.sleep(self.delay)  # blocking, like real inference
            self.yielded += 1
            yield {"choices": [{"delta": {"content": tok}}]}


def make_client(stream) -> StreamingLLMClient:
    llama = MagicMock()
    llama.create_chat_completion.return_value = stream
    with patch("voice_assistant.llm.client.Llama", return_value=llama), \
         patch("voice_assistant.llm.client._LLAMA_AVAILABLE", True):
        return StreamingLLMClient(LLMConfig(model_path="dummy"))


async def test_generation_does_not_block_event_loop() -> None:
    client = make_client(SlowStream(["a", "b", "c", "d", "e"], delay=0.02))
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.005)

    tick_task = asyncio.create_task(ticker())
    out: asyncio.Queue[str] = asyncio.Queue()
    reply = await client.stream_tokens([{"role": "user", "content": "hi"}], out)
    tick_task.cancel()

    assert reply == "abcde"
    assert out.qsize() == 5
    # ~100ms of generation; a blocked loop would manage at most a tick or two.
    assert ticks >= 8


async def test_cancellation_stops_generation_early() -> None:
    stream = SlowStream([str(i) for i in range(100)], delay=0.01)
    client = make_client(stream)
    out: asyncio.Queue[str] = asyncio.Queue()

    task = asyncio.create_task(client.stream_tokens([], out))
    while out.qsize() < 3:
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.yielded < 20
    # The model lock is released, so the next request can run.
    assert client._lock.acquire(timeout=1)
    client._lock.release()


async def test_generation_error_is_raised() -> None:
    def boom():
        raise ValueError("bad gguf")
        yield  # pragma: no cover

    client = make_client(boom())
    with pytest.raises(RuntimeError, match="bad gguf"):
        await client.stream_tokens([], asyncio.Queue())


def test_token_logprobs_uses_one_completion() -> None:
    llama = MagicMock()
    llama.create_completion.return_value = {
        "choices": [{"logprobs": {"top_logprobs": [{" yes": -0.1, " no": -2.0}]}}]
    }
    with patch("voice_assistant.llm.client.Llama", return_value=llama), \
         patch("voice_assistant.llm.client._LLAMA_AVAILABLE", True):
        client = StreamingLLMClient(LLMConfig(model_path="dummy"))

    result = client.token_logprobs("Q?", [" yes", " no", " maybe"])

    assert result == {" yes": -0.1, " no": -2.0, " maybe": -100.0}
    assert llama.create_completion.call_count == 1


def test_warmup_processes_system_prompt_with_one_token() -> None:
    client = make_client(SlowStream([], delay=0))

    seconds = client.warmup("Be brief.")

    call = client._llama.create_chat_completion.call_args.kwargs
    assert call["max_tokens"] == 1
    assert call["messages"][0] == {"role": "system", "content": "Be brief."}
    assert seconds >= 0
