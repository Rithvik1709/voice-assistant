from __future__ import annotations

import asyncio
import io

from voice_assistant.chat import build_chat
from voice_assistant.config import Settings
from voice_assistant.mocks import MockLLMClient


def scripted(lines: list[str]):
    it = iter(lines)

    def read_line(_prompt: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return read_line


async def chat(lines: list[str], monkeypatch) -> str:
    monkeypatch.delenv("CONVERSATION_MEMORY_PATH", raising=False)
    monkeypatch.delenv("ENABLE_WEATHER", raising=False)
    out = io.StringIO()
    orchestrator, _ = build_chat(Settings(), MockLLMClient(), read_line=scripted(lines), out=out)
    await asyncio.wait_for(orchestrator.run(), 5)
    return out.getvalue()


async def test_chat_prints_llm_and_action_replies_then_exits(monkeypatch) -> None:
    output = await chat(["tell me something", "what time is it", "", "/quit", "never read"], monkeypatch)

    lines = output.strip().splitlines()
    assert lines[0] == "Vaani: Hello this is a mock response from the assistant."
    assert lines[1].startswith("Vaani: It is ")
    assert lines[1].endswith(".")
    assert len(lines) == 2


async def test_chat_reset_and_help_commands(monkeypatch) -> None:
    output = await chat(["hello", "/reset", "/help"], monkeypatch)

    assert "Vaani: Hi, I am listening." in output
    assert "(conversation cleared)" in output
    assert "/quit exits" in output
