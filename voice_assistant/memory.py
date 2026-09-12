from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SessionMemory:
    path: Path

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

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
