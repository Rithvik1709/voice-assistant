from voice_assistant.actions import BasicIntentActions


def test_basic_actions_handle_known_intent() -> None:
    actions = BasicIntentActions()

    result = actions.handle("hello", {"intent": "greeting", "confidence": 0.8})

    assert result.handled
    assert "listening" in result.response


def test_basic_actions_ignore_low_confidence_intent() -> None:
    actions = BasicIntentActions(min_confidence=0.5)

    result = actions.handle("maybe music", {"intent": "play_music", "confidence": 0.2})

    assert not result.handled


def test_greeting_inside_a_real_question_goes_to_llm() -> None:
    actions = BasicIntentActions()

    result = actions.handle(
        "hi can you explain how black holes form",
        {"intent": "greeting", "confidence": 0.5},
    )

    assert not result.handled


def test_time_and_date_actions_use_clock() -> None:
    from datetime import datetime

    actions = BasicIntentActions(clock=lambda: datetime(2026, 9, 6, 9, 5))

    assert actions.handle("time", {"intent": "time", "confidence": 0.8}).response == "It is 9:05 AM."
    assert (
        actions.handle("date", {"intent": "date", "confidence": 0.8}).response
        == "Today is Sunday, September 6, 2026."
    )
