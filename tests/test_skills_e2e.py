import sys
from pathlib import Path

from voice_assistant.nlu import SimpleIntentClassifier
from voice_assistant.skills.registry import (
    reset,
    get_intent_keywords,
    dispatch,
    get_skill_name,
    _discover,
)


def test_e2e_add_new_skill_no_core_changes():
    skills_dir = Path(__file__).resolve().parent.parent / "voice_assistant" / "skills"

    new_file = skills_dir / "alarm.py"
    assert not new_file.exists(), "Precondition: alarm.py should not exist yet"

    try:
        new_file.write_text("""
INTENTS = {
    "set_alarm": {
        "keywords": ["alarm", "set alarm", "alarm set karo", "alarm laga do"],
    },
    "timer": {
        "keywords": ["timer", "set timer", "timer laga do"],
    },
}

def handle(intent, text, entities=None):
    return {"response": f"{intent} created for: {text}", "intent": intent}
""")

        reset()

        keywords = get_intent_keywords()
        assert "set_alarm" in keywords
        assert "timer" in keywords
        assert "alarm" in keywords["set_alarm"]
        assert "timer laga do" in keywords["timer"]

        assert get_skill_name("set_alarm") == "alarm"
        assert get_skill_name("timer") == "alarm"

        result = dispatch("set_alarm", "set alarm for 7am")
        assert result["intent"] == "set_alarm"
        assert "7am" in result["response"]

        result = dispatch("timer", "set timer for 10 minutes")
        assert result["intent"] == "timer"
        assert "timer" in result["response"]

        c = SimpleIntentClassifier()
        assert c.classify("set an alarm for 7am")["intent"] == "set_alarm"
        assert c.classify("timer laga do")["intent"] == "timer"

        assert c.classify("hello")["intent"] == "greeting"
        assert c.classify("play music")["intent"] == "play_music"
        assert c.classify("stop")["intent"] == "stop"
        assert c.classify("weather")["intent"] == "weather"

    finally:
        if new_file.exists():
            new_file.unlink()
        reset()


def test_existing_skills_untouched_after_cleanup():
    reset()
    c = SimpleIntentClassifier()
    assert c.classify("hello")["intent"] == "greeting"
    assert c.classify("play the song")["intent"] == "play_music"
    assert c.classify("band karo")["intent"] == "stop"
    assert c.classify("mausam batao")["intent"] == "weather"

    keywords = get_intent_keywords()
    assert "set_alarm" not in keywords
    assert "timer" not in keywords


def test_duplicate_intent_warning():
    skills_dir = Path(__file__).resolve().parent.parent / "voice_assistant" / "skills"
    dup_file = skills_dir / "dup_skill.py"
    dup_code = """
INTENTS = {
    "greeting": {
        "keywords": ["howdy"],
    },
}
"""
    try:
        dup_file.write_text(dup_code)
        reset()

        keywords = get_intent_keywords()
        assert "greeting" in keywords
        assert "howdy" in keywords["greeting"]

    finally:
        if dup_file.exists():
            dup_file.unlink()
        reset()


def test_skill_with_errors_graceful():
    skills_dir = Path(__file__).resolve().parent.parent / "voice_assistant" / "skills"
    bad_file = skills_dir / "_test_broken.py"
    bad_code = """
import nonexistent_module_that_does_not_exist_xyz
INTENTS = {"broken": {"keywords": ["test"]}}
"""
    try:
        bad_file.write_text(bad_code)
        reset()

        keywords = get_intent_keywords()
        assert "broken" not in keywords
        assert "greeting" in keywords

    finally:
        if bad_file.exists():
            bad_file.unlink()
        reset()


def test_classifier_dispatch_integration():
    reset()
    c = SimpleIntentClassifier()
    result = c.dispatch("greeting", "hello")
    assert result["intent"] == "greeting"
    assert result["response"] == "Hello! How can I help you?"

    result = c.dispatch("weather", "weather in delhi")
    assert result["intent"] == "weather"
    assert "delhi" in result["response"]

    result = c.dispatch("nonexistent", "test")
    assert result["response"] is None
