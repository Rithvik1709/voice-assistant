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
