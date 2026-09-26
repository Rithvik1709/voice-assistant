from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SessionMemory:
    """Conversation history persisted as JSON lines.

    The file is compacted to its most recent half once it holds more than
    `max_messages` entries, so long-running assistants do not grow it forever.
    """

    path: Path
    max_messages: int = 2000
    _count: int | None = field(default=None, init=False, repr=False)

    def load_recent(self, max_messages: int) -> list[dict[str, str]]:
        if max_messages <= 0 or not self.path.exists():
            return []

        messages: list[dict[str, str]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant", "system"} and isinstance(content, str):
                messages.append({"role": role, "content": content})

        return messages[-max_messages:]

    def append(self, role: str, content: str) -> None:
        if role not in {"user", "assistant", "system"} or not content.strip():
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"role": role, "content": content}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

        if self._count is None:
            self._count = len(self.path.read_text(encoding="utf-8").splitlines())
        else:
            self._count += 1
        if self._count > self.max_messages:
            self._compact(self.max_messages // 2)

    def _compact(self, keep: int) -> None:
        lines = self.path.read_text(encoding="utf-8").splitlines()[-keep:]
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        tmp.replace(self.path)
        self._count = len(lines)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
        self._count = 0
