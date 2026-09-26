import pytest

from voice_assistant.tts.normalize import normalize_for_speech as speak


@pytest.mark.parametrize(
    ("raw", "spoken"),
    [
        ("**Paris** is the capital of *France*! 🇫🇷😀", "Paris is the capital of France!"),
        ("Tips:\n1. Drink water\n2. Sleep **8 hours**\n- Walk daily", "Tips: Drink water. Sleep 8 hours. Walk daily"),
        ("It's 21°C with 40% humidity & light wind.", "It's 21 degrees Celsius with 40 percent humidity and light wind."),
        ("See [the docs](https://x.io) or https://example.com/a?b=1", "See the docs or a link"),
        ("```python\nprint('hi')\n```\nThat prints hi.", "That prints hi."),
        ("## Summary\nIt costs $1,200, or €5.50.", "Summary. It costs 1,200 dollars, or 5.50 euros."),
        ("Use `pip install vaani`.", "Use pip install vaani."),
        ("snake_case_name stays", "snake_case_name stays"),
        ("**", ""),
        ("", ""),
    ],
)
def test_normalize_for_speech(raw: str, spoken: str) -> None:
    assert speak(raw) == spoken


def test_plain_sentences_are_unchanged() -> None:
    text = "The meeting is at three, in room four."
    assert speak(text) == text
