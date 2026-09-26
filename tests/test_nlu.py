from voice_assistant.nlu import SimpleIntentClassifier


def test_simple_english_intents():
    c = SimpleIntentClassifier()

    assert c.classify("hello there")["intent"] == "greeting"
    assert c.classify("please play the song")["intent"] == "play_music"


def test_code_mixed_hindi_english():
    c = SimpleIntentClassifier()

    # Devanagari greeting
    res = c.classify("नमस्ते, कैसे हो")
    assert res["intent"] in {"greeting", "unknown"}
    assert res["lang"] == "hi"

    # Code-mixed
    res2 = c.classify("play gaana")
    assert res2["intent"] == "play_music"


def test_additional_hinglish_queries():
    c = SimpleIntentClassifier()

    assert c.classify("gaana chala do")["intent"] == "play_music"

    assert c.classify("music band karo")["intent"] == "stop"

    assert c.classify("mausam batao")["intent"] == "weather"


def test_text_normalization():
    c = SimpleIntentClassifier()

    assert c.classify("PLAY!!! MUSIC")["intent"] == "play_music"

    assert c.classify("Hello!!!")["intent"] == "greeting"


def test_unknown_intent():
    c = SimpleIntentClassifier()

    assert c.classify(
        "tell me about quantum computing"
    )["intent"] == "unknown"

def test_devanagari_without_keywords_is_not_a_greeting():
    c = SimpleIntentClassifier()

    res = c.classify("मुझे एक कहानी सुनाओ")
    assert res["intent"] == "unknown"
    assert res["lang"] == "hi"


def test_generic_play_is_low_confidence():
    c = SimpleIntentClassifier()

    res = c.classify("how do I play chess")
    assert res["intent"] == "play_music"
    assert res["confidence"] < 0.5


def test_time_and_date_intents():
    c = SimpleIntentClassifier()

    assert c.classify("What's the time?")["intent"] == "time"
    assert c.classify("kitne baje hai")["intent"] == "time"
    assert c.classify("what is the date today")["intent"] == "date"
