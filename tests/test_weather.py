from __future__ import annotations

import pytest

from voice_assistant.actions import BasicIntentActions
from voice_assistant.weather import OpenMeteoWeather, WeatherReport, extract_city


@pytest.mark.parametrize(
    ("text", "city"),
    [
        ("what is the weather in New York today", "New York"),
        ("weather in paris?", "paris"),
        ("whats the weather like in San Francisco right now", "San Francisco"),
        ("weather for tokyo tomorrow", "tokyo"),
        ("delhi ka mausam batao", "delhi"),
        ("how is the weather", None),
        ("what is the weather like outside", None),
    ],
)
def test_extract_city(text: str, city: str | None) -> None:
    assert extract_city(text) == city


def fake_fetch(places, forecast):
    calls = []

    def fetch(url, params, timeout):
        calls.append((url, params))
        return {"results": places} if "geocoding" in url else forecast

    return fetch, calls


FORECAST = {
    "current": {"temperature_2m": 18.4, "weather_code": 2},
    "daily": {"temperature_2m_max": [21.2], "temperature_2m_min": [11.6]},
}


def test_open_meteo_report() -> None:
    fetch, calls = fake_fetch([{"name": "Paris", "latitude": 48.8, "longitude": 2.3}], FORECAST)

    report = OpenMeteoWeather(unit="fahrenheit", fetch=fetch).current("paris")

    assert report is not None
    assert report.spoken() == (
        "In Paris it is 18 degrees with partly cloudy skies. Today's high is 21 and the low is 12."
    )
    assert calls[1][1]["temperature_unit"] == "fahrenheit"
    assert calls[1][1]["latitude"] == 48.8


def test_open_meteo_unknown_city() -> None:
    fetch, calls = fake_fetch([], FORECAST)

    assert OpenMeteoWeather(fetch=fetch).current("nowhere") is None
    assert len(calls) == 1


class StubWeather:
    def __init__(self, report=None, error: Exception | None = None) -> None:
        self.report = report
        self.error = error
        self.cities: list[str] = []

    def current(self, city: str):
        self.cities.append(city)
        if self.error:
            raise self.error
        return self.report


WEATHER = {"intent": "weather", "confidence": 0.8}
REPORT = WeatherReport("Pune", 27.0, 30.0, 21.0, "light rain", "celsius")


def test_weather_action_disabled_explains_how_to_enable() -> None:
    result = BasicIntentActions().handle("weather in pune", WEATHER)
    assert result.handled and "ENABLE_WEATHER" in result.response


def test_weather_action_uses_city_from_request() -> None:
    weather = StubWeather(REPORT)
    result = BasicIntentActions(weather=weather).handle("weather in pune", WEATHER)
    assert weather.cities == ["pune"]
    assert result.response.startswith("In Pune it is 27 degrees with light rain.")


def test_weather_action_falls_back_to_default_city_or_asks() -> None:
    weather = StubWeather(REPORT)
    assert BasicIntentActions(weather=weather, default_city="Pune").handle("weather", WEATHER).handled
    assert weather.cities == ["Pune"]
    assert "Which city" in BasicIntentActions(weather=weather).handle("weather", WEATHER).response


def test_weather_action_handles_errors_and_unknown_places() -> None:
    down = BasicIntentActions(weather=StubWeather(error=TimeoutError()))
    assert "could not reach" in down.handle("weather in pune", WEATHER).response

    unknown = BasicIntentActions(weather=StubWeather(None))
    assert unknown.handle("weather in atlantis", WEATHER).response == "I could not find a place called atlantis."
