"""Live weather through Open-Meteo (free, no API key).

Only used when ENABLE_WEATHER=1: it sends the requested city name to
open-meteo.com, which a local-first assistant should not do silently.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes, as used by Open-Meteo.
WMO_CODES = {
    0: "clear skies",
    1: "mostly clear skies",
    2: "partly cloudy skies",
    3: "overcast skies",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "heavy freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "a thunderstorm",
    96: "a thunderstorm with hail",
    99: "a severe thunderstorm with hail",
}

_CITY_AFTER = re.compile(r"\b(?:in|at|for|of)\s+([a-z][a-z .'-]*)", re.I)
_CITY_BEFORE_HINDI = re.compile(r"\b([a-z][a-z .'-]*?)\s+(?:ka|kaa|ki|mein|me)\s+(?:mausam|mosam|weather)\b", re.I)
_TRAILING = re.compile(
    r"\b(?:today|tomorrow|tonight|now|right now|currently|like|please|batao|kya hai|kaisa hai|this week)\b.*$",
    re.I,
)
_NOT_CITIES = {"the", "my", "here", "there", "outside", "today", "tomorrow", "weather", "mausam"}


def extract_city(text: str) -> str | None:
    """Pull a place name out of a weather request, e.g. "weather in New York today"."""
    cleaned = re.sub(r"[?!,.]", " ", text).strip()
    for pattern in (_CITY_AFTER, _CITY_BEFORE_HINDI):
        match = pattern.search(cleaned)
        if not match:
            continue
        city = _TRAILING.sub("", match.group(1)).strip(" .'-")
        city = re.sub(r"^(?:the)\s+", "", city, flags=re.I)
        if city and city.lower() not in _NOT_CITIES:
            return city
    return None


@dataclass(slots=True)
class WeatherReport:
    place: str
    temperature: float
    high: float | None
    low: float | None
    condition: str
    unit: str  # "celsius" | "fahrenheit"

    def spoken(self) -> str:
        degrees = f"{round(self.temperature)} degrees"
        sentence = f"In {self.place} it is {degrees} with {self.condition}."
        if self.high is not None and self.low is not None:
            sentence += f" Today's high is {round(self.high)} and the low is {round(self.low)}."
        return sentence


class WeatherProvider(Protocol):
    def current(self, city: str) -> WeatherReport | None: ...


def _get_json(url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": "vaani-voice-assistant"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class OpenMeteoWeather:
    def __init__(self, unit: str = "celsius", timeout_s: float = 4.0, fetch=_get_json) -> None:
        self.unit = "fahrenheit" if unit.lower().startswith("f") else "celsius"
        self.timeout_s = timeout_s
        self._fetch = fetch

    def current(self, city: str) -> WeatherReport | None:
        """Return current conditions, or None if the city is unknown."""
        places = self._fetch(
            GEOCODE_URL,
            {"name": city, "count": 1, "language": "en", "format": "json"},
            self.timeout_s,
        ).get("results") or []
        if not places:
            return None
        place = places[0]

        data = self._fetch(
            FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min",
                "timezone": "auto",
                "forecast_days": 1,
                "temperature_unit": self.unit,
            },
            self.timeout_s,
        )
        current = data["current"]
        daily = data.get("daily") or {}
        highs, lows = daily.get("temperature_2m_max") or [None], daily.get("temperature_2m_min") or [None]
        return WeatherReport(
            place=place.get("name", city),
            temperature=float(current["temperature_2m"]),
            high=highs[0],
            low=lows[0],
            condition=WMO_CODES.get(int(current.get("weather_code", -1)), "mixed conditions"),
            unit=self.unit,
        )
