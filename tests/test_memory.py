from pathlib import Path

from voice_assistant.memory import SessionMemory


def test_session_memory_round_trips_recent_messages(tmp_path: Path) -> None:
    memory = SessionMemory(tmp_path / "session.jsonl")
    memory.append("user", "hello")
    memory.append("assistant", "hi")
    memory.append("user", "what now")

    assert memory.load_recent(2) == [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "what now"},
    ]


def test_session_memory_skips_invalid_lines(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text(
        'not-json\n{"role": "user", "content": "hello"}\n',
        encoding="utf-8",
    )

    assert SessionMemory(path).load_recent(5) == [{"role": "user", "content": "hello"}]
