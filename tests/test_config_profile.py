from voice_assistant.config import Settings


def test_low_latency_profile_caps_expensive_defaults(monkeypatch) -> None:
    monkeypatch.setenv("VAANI_PROFILE", "low_latency")
    monkeypatch.setenv("CHUNK_MS", "30")
    monkeypatch.setenv("ASR_ENDPOINT_SILENCE_MS", "80")

    settings = Settings()

    assert settings.chunk_ms == 20
    assert settings.asr_endpoint_silence_ms == 50
    assert settings.llm_max_tokens == 128
    assert settings.sentence_max_tokens <= 6
    assert settings.tts_eager_min_words <= 2
