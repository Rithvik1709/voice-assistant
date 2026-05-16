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

    # Code-mixed: English verb + Hindi word in Devanagari
    res2 = c.classify("play gaana")
    assert res2["intent"] == "play_music"
