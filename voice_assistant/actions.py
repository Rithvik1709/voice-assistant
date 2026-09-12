from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class ActionResult:
    handled: bool
    response: str = ""


class ActionHandler(Protocol):
    def handle(self, text: str, intent: dict[str, object]) -> ActionResult:
        """Return a response when an intent can be handled without the LLM."""


class BasicIntentActions:
    def __init__(self, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence

    def handle(self, text: str, intent: dict[str, object]) -> ActionResult:
        name = str(intent.get("intent", "unknown"))
        confidence = float(intent.get("confidence", 0.0))

        if confidence < self.min_confidence:
            return ActionResult(False)

        if name == "greeting":
            return ActionResult(True, "Hi, I am listening.")

        if name == "stop":
            return ActionResult(True, "Okay, stopping.")

        if name == "weather":
            return ActionResult(
                True,
                "I can detect weather requests now. Connect a weather action to fetch live conditions.",
            )

        if name == "play_music":
            return ActionResult(
                True,
                "I can detect music requests now. Connect a music action to control playback.",
            )

        return ActionResult(False)
