from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from voice_assistant.lang import base_language
from voice_assistant.weather import WeatherProvider, extract_city

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ActionResult:
    handled: bool
    response: str = ""


class ActionHandler(Protocol):
    def handle(self, text: str, intent: dict[str, object]) -> ActionResult:
        """Return a response when an intent can be handled without the LLM."""


class BasicIntentActions:
    """Deterministic replies for simple commands, answered before the LLM.

    Greetings are only short-circuited for short utterances so that
    "hi, can you explain black holes" still reaches the LLM.

    The replies are English, and the keywords English and Hindi, so requests
    spoken in any other language (intent["spoken_language"]) go to the LLM,
    which answers in that language.
    """

    LANGUAGES = frozenset({"en", "hi"})

    def __init__(
        self,
        min_confidence: float = 0.5,
        max_greeting_words: int = 4,
        clock: Callable[[], datetime] = datetime.now,
        weather: WeatherProvider | None = None,
        default_city: str = "",
    ) -> None:
        self.min_confidence = min_confidence
        self.max_greeting_words = max_greeting_words
        self.clock = clock
        self.weather = weather
        self.default_city = default_city.strip()

    def handle(self, text: str, intent: dict[str, object]) -> ActionResult:
        name = str(intent.get("intent", "unknown"))
        confidence = float(intent.get("confidence", 0.0))  # type: ignore[arg-type]

        if confidence < self.min_confidence:
            return ActionResult(False)

        spoken = base_language(str(intent.get("spoken_language") or ""))
        if spoken is not None and spoken not in self.LANGUAGES:
            return ActionResult(False)

        if name == "greeting":
            if len(text.split()) > self.max_greeting_words:
                return ActionResult(False)
            return ActionResult(True, "Hi, I am listening.")

        if name == "stop":
            return ActionResult(True, "Okay, stopping.")

        if name == "time":
            now = self.clock()
            return ActionResult(True, f"It is {now.strftime('%I:%M %p').lstrip('0')}.")

        if name == "date":
            now = self.clock()
            return ActionResult(True, f"Today is {now.strftime('%A, %B')} {now.day}, {now.year}.")

        if name == "weather":
            return self._weather(text)

        if name == "play_music":
            return ActionResult(
                True,
                "I can detect music requests now. Connect a music action to control playback.",
            )

        return ActionResult(False)

    def _weather(self, text: str) -> ActionResult:
        if self.weather is None:
            return ActionResult(
                True,
                "Live weather is turned off. Set ENABLE_WEATHER to 1 to let me look it up.",
            )
        city = extract_city(text) or self.default_city
        if not city:
            return ActionResult(True, "Which city would you like the weather for?")
        try:
            report = self.weather.current(city)
        except Exception as exc:  # network errors, timeouts, bad responses
            logger.warning("Weather lookup for %r failed: %s", city, exc)
            return ActionResult(True, "Sorry, I could not reach the weather service right now.")
        if report is None:
            return ActionResult(True, f"I could not find a place called {city}.")
        return ActionResult(True, report.spoken())
