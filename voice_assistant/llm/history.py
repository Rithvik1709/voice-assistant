"""Bounded conversation history with sliding-window context management.

Long-running voice sessions accumulate unbounded conversation_history lists that
inflate LLM prompt payloads, increase memory usage, and degrade streaming latency.
This module provides ``ConversationHistory``, a drop-in replacement that enforces
a configurable cap on the number of retained dialogue *turns* (one turn = one
user message + one assistant reply).  The system prompt, if any, is always kept
at position 0 and is never evicted.
"""

from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)

# A single chat message understood by llama-cpp and OpenAI-compatible APIs.
Message = dict[str, str]


class ConversationHistory:
    """Sliding-window conversation history with a hard cap on retained turns.

    Parameters
    ----------
    max_turns:
        Maximum number of *dialogue turns* to keep.  One turn consists of one
        user message followed by one assistant message.  When the cap is
        exceeded the oldest turn-pair is evicted.  Must be >= 1.
    system_prompt:
        Optional system/instruction message prepended to every outgoing
        message list.  It is never counted toward ``max_turns`` and is
        never evicted.
    """

    def __init__(self, max_turns: int = 10, system_prompt: str | None = None) -> None:
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")

        self._max_turns = max_turns
        self._system_prompt = system_prompt
        # Stores only user/assistant pairs – system prompt stored separately.
        self._turns: list[tuple[Message, Message]] = []
        # Pending user message waiting for its paired assistant reply.
        self._pending_user: Message | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def max_turns(self) -> int:
        return self._max_turns

    @property
    def turn_count(self) -> int:
        """Number of complete (user + assistant) turns currently retained."""
        return len(self._turns)

    def append_user(self, content: str) -> None:
        """Record a user utterance.  Must be followed by :meth:`append_assistant`."""
        self._pending_user = {"role": "user", "content": content}

    def append_assistant(self, content: str) -> None:
        """Pair the pending user utterance with the assistant reply and store the turn."""
        if self._pending_user is None:
            # Defensive: assistant reply without a preceding user message.
            logger.warning(
                "ConversationHistory: received assistant reply without a preceding "
                "user message – appending as orphan turn."
            )
            user_msg: Message = {"role": "user", "content": ""}
        else:
            user_msg = self._pending_user
            self._pending_user = None

        assistant_msg: Message = {"role": "assistant", "content": content}
        self._turns.append((user_msg, assistant_msg))
        self._evict_if_needed()

    def messages(self) -> list[Message]:
        """Return the full message list ready to pass to the LLM.

        The returned list always starts with the system prompt (if set),
        followed by at most ``max_turns`` turn-pairs in chronological order,
        and finally the pending (unanswered) user message if one exists.
        """
        out: list[Message] = []
        if self._system_prompt:
            out.append({"role": "system", "content": self._system_prompt})
        for user_msg, assistant_msg in self._turns:
            out.append(user_msg)
            out.append(assistant_msg)
        if self._pending_user is not None:
            out.append(self._pending_user)
        return out

    def clear(self) -> None:
        """Discard all turns (and any pending user message)."""
        self._turns.clear()
        self._pending_user = None
        logger.debug("ConversationHistory: history cleared")

    def set_system_prompt(self, system_prompt: str | None) -> None:
        """Update the pinned system prompt without touching the turn history."""
        self._system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        evicted = 0
        while len(self._turns) > self._max_turns:
            self._turns.pop(0)
            evicted += 1
        if evicted:
            logger.debug(
                "ConversationHistory: evicted %d oldest turn(s) to stay within "
                "max_turns=%d cap",
                evicted,
                self._max_turns,
            )

    # ------------------------------------------------------------------
    # Dunder helpers (useful for debugging / logging)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of *messages* (not turns) currently in history."""
        system_offset = 1 if self._system_prompt else 0
        pending_offset = 1 if self._pending_user else 0
        return system_offset + len(self._turns) * 2 + pending_offset

    def __repr__(self) -> str:
        return (
            f"ConversationHistory("
            f"max_turns={self._max_turns}, "
            f"turn_count={self.turn_count}, "
            f"has_system_prompt={self._system_prompt is not None})"
        )
