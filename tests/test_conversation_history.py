"""Unit tests for ConversationHistory (bounded sliding-window context manager)."""

from __future__ import annotations

import pytest

from voice_assistant.llm.history import ConversationHistory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill(history: ConversationHistory, n: int) -> None:
    """Append *n* complete turns (user_i / assistant_i) into *history*."""
    for i in range(n):
        history.append_user(f"user_{i}")
        history.append_assistant(f"assistant_{i}")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestInit:
    def test_defaults(self):
        h = ConversationHistory()
        assert h.max_turns == 10
        assert h.turn_count == 0

    def test_custom_max_turns(self):
        h = ConversationHistory(max_turns=3)
        assert h.max_turns == 3

    def test_invalid_max_turns(self):
        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            ConversationHistory(max_turns=0)

    def test_with_system_prompt(self):
        h = ConversationHistory(system_prompt="You are helpful.")
        msgs = h.messages()
        assert len(msgs) == 1
        assert msgs[0] == {"role": "system", "content": "You are helpful."}


# ---------------------------------------------------------------------------
# Appending turns
# ---------------------------------------------------------------------------

class TestAppend:
    def test_single_turn(self):
        h = ConversationHistory(max_turns=5)
        h.append_user("hello")
        h.append_assistant("hi there")

        msgs = h.messages()
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "hello"}
        assert msgs[1] == {"role": "assistant", "content": "hi there"}

    def test_turn_count_increments(self):
        h = ConversationHistory(max_turns=5)
        assert h.turn_count == 0
        _fill(h, 3)
        assert h.turn_count == 3

    def test_pending_user_appears_at_end(self):
        """append_user without append_assistant still surfaces the message."""
        h = ConversationHistory(max_turns=5)
        h.append_user("pending question")
        msgs = h.messages()
        assert msgs[-1] == {"role": "user", "content": "pending question"}

    def test_assistant_without_user_creates_orphan(self):
        """Defensive: assistant reply without preceding user message is stored."""
        h = ConversationHistory(max_turns=5)
        h.append_assistant("orphan reply")
        msgs = h.messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1] == {"role": "assistant", "content": "orphan reply"}


# ---------------------------------------------------------------------------
# Eviction / sliding window
# ---------------------------------------------------------------------------

class TestEviction:
    def test_no_eviction_at_cap(self):
        h = ConversationHistory(max_turns=3)
        _fill(h, 3)
        assert h.turn_count == 3

    def test_eviction_one_over_cap(self):
        h = ConversationHistory(max_turns=3)
        _fill(h, 4)
        assert h.turn_count == 3

    def test_oldest_turn_evicted_first(self):
        h = ConversationHistory(max_turns=2)
        _fill(h, 3)  # turns: user_0/assistant_0, user_1/assistant_1, user_2/assistant_2
        # user_0 / assistant_0 should be gone
        msgs = h.messages()
        contents = [m["content"] for m in msgs]
        assert "user_0" not in contents
        assert "assistant_0" not in contents
        assert "user_1" in contents
        assert "user_2" in contents

    def test_system_prompt_never_evicted(self):
        h = ConversationHistory(max_turns=2, system_prompt="Be concise.")
        _fill(h, 10)
        msgs = h.messages()
        assert msgs[0] == {"role": "system", "content": "Be concise."}
        assert h.turn_count == 2

    def test_message_count_with_system_prompt(self):
        h = ConversationHistory(max_turns=3, system_prompt="sys")
        _fill(h, 3)
        # system(1) + 3 turns * 2 messages(6) = 7
        assert len(h) == 7

    def test_message_count_without_system_prompt(self):
        h = ConversationHistory(max_turns=3)
        _fill(h, 3)
        # 3 turns * 2 = 6
        assert len(h) == 6

    def test_large_overflow(self):
        h = ConversationHistory(max_turns=5)
        _fill(h, 100)
        assert h.turn_count == 5

    def test_max_turns_1(self):
        """Edge case: only one turn retained."""
        h = ConversationHistory(max_turns=1)
        _fill(h, 5)
        assert h.turn_count == 1
        msgs = h.messages()
        assert msgs[0]["content"] == "user_4"
        assert msgs[1]["content"] == "assistant_4"


# ---------------------------------------------------------------------------
# clear() and set_system_prompt()
# ---------------------------------------------------------------------------

class TestMutations:
    def test_clear_removes_turns(self):
        h = ConversationHistory(max_turns=5)
        _fill(h, 3)
        h.clear()
        assert h.turn_count == 0
        assert h.messages() == []

    def test_clear_removes_pending_user(self):
        h = ConversationHistory(max_turns=5)
        h.append_user("dangling")
        h.clear()
        assert h.messages() == []

    def test_clear_keeps_system_prompt(self):
        h = ConversationHistory(max_turns=5, system_prompt="Always be helpful.")
        _fill(h, 3)
        h.clear()
        msgs = h.messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_set_system_prompt(self):
        h = ConversationHistory(max_turns=5)
        _fill(h, 2)
        h.set_system_prompt("New instructions.")
        msgs = h.messages()
        assert msgs[0] == {"role": "system", "content": "New instructions."}
        assert h.turn_count == 2

    def test_set_system_prompt_none_removes_it(self):
        h = ConversationHistory(max_turns=5, system_prompt="old")
        h.set_system_prompt(None)
        msgs = h.messages()
        assert all(m["role"] != "system" for m in msgs)


# ---------------------------------------------------------------------------
# repr / __len__
# ---------------------------------------------------------------------------

class TestDunderMethods:
    def test_repr(self):
        h = ConversationHistory(max_turns=5, system_prompt="hi")
        r = repr(h)
        assert "max_turns=5" in r
        assert "has_system_prompt=True" in r

    def test_len_empty(self):
        h = ConversationHistory()
        assert len(h) == 0

    def test_len_with_pending(self):
        h = ConversationHistory(max_turns=5)
        h.append_user("pending")
        assert len(h) == 1
