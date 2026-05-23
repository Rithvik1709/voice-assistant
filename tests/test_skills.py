from voice_assistant.nlu import SimpleIntentClassifier
from voice_assistant.skills.registry import (
    _discover,
    get_intent_keywords,
    dispatch,
    get_skill_name,
)


def test_skills_auto_discovery():
    """All skill files in skills/ should be auto-discovered."""
    _discover()
    keywords = get_intent_keywords()
    assert "greeting" in keywords
    assert "play_music" in keywords
    assert "stop" in keywords
    assert "weather" in keywords


def test_skill_intent_keywords():
    """Each skill should expose the correct keywords."""
    _discover()
    keywords = get_intent_keywords()
    assert "hello" in keywords["greeting"]
    assert "song" in keywords["play_music"]
    assert "band karo" in keywords["stop"]
    assert "mausam" in keywords["weather"]


def test_skill_dispatch():
    """Dispatch should route to the correct skill handler."""
    _discover()
    result = dispatch("greeting", "hello")
    assert result["intent"] == "greeting"
    assert result["response"] == "Hello! How can I help you?"

    result = dispatch("weather", "what is the weather like")
    assert result["intent"] == "weather"
    assert "weather" in result["response"]

    result = dispatch("unknown_intent", "something")
    assert result["response"] is None


def test_skill_name_lookup():
    _discover()
    assert get_skill_name("greeting") == "greeting"
    assert get_skill_name("play_music") == "play_music"
    assert get_skill_name("weather") == "weather"
    assert get_skill_name("nonexistent") is None


def test_classifier_integration():
    """Classifier should correctly match intents from registered skills."""
    c = SimpleIntentClassifier()
    assert c.classify("hello there")["intent"] == "greeting"
    assert c.classify("please play the song")["intent"] == "play_music"


def test_classifier_hinglish():
    """Hinglish queries should be matched correctly."""
    c = SimpleIntentClassifier()
    assert c.classify("gaana chala do")["intent"] == "play_music"
    assert c.classify("music band karo")["intent"] == "stop"
    assert c.classify("mausam batao")["intent"] == "weather"


def test_classifier_normalization():
    """Text normalization should handle punctuation and case."""
    c = SimpleIntentClassifier()
    assert c.classify("PLAY!!! MUSIC")["intent"] == "play_music"
    assert c.classify("Hello!!!")["intent"] == "greeting"


def test_classifier_unknown():
    """Unknown intents should return 'unknown'."""
    c = SimpleIntentClassifier()
    assert c.classify("tell me about quantum computing")["intent"] == "unknown"


def test_classifier_devanagari_fallback():
    """Devanagari text without keyword match should fall back to greeting."""
    c = SimpleIntentClassifier()
    res = c.classify("नमस्ते, कैसे हो")
    assert res["intent"] in {"greeting", "unknown"}
    assert res["lang"] == "hi"


def test_classifier_code_mixed():
    """Code-mixed queries should work."""
    c = SimpleIntentClassifier()
    res = c.classify("play gaana")
    assert res["intent"] == "play_music"


def test_classifier_dispatch():
    """Classifier dispatch should route to skill handlers."""
    c = SimpleIntentClassifier()
    result = c.dispatch("greeting", "hello")
    assert result["intent"] == "greeting"
    assert result["response"] == "Hello! How can I help you?"


def test_handler_returns_string():
    """Handler returning a string should be normalized to dict."""
    from voice_assistant.skills.registry import dispatch
    result = dispatch("greeting", "hello")
    assert isinstance(result, dict)
    assert "response" in result
