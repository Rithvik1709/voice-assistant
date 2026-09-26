from voice_assistant.config import Settings


def test_low_latency_profile_caps_expensive_defaults(monkeypatch) -> None:
    monkeypatch.setenv("VAANI_PROFILE", "low_latency")
    monkeypatch.setenv("CHUNK_MS", "30")
    monkeypatch.setenv("ASR_ENDPOINT_SILENCE_MS", "600")

    settings = Settings()

    assert settings.chunk_ms == 20
    assert settings.asr_endpoint_silence_ms == 300
    assert settings.llm_max_tokens == 128
    assert settings.sentence_max_tokens <= 6
    assert settings.tts_eager_min_words <= 2


def test_invalid_env_values_raise_config_error(monkeypatch) -> None:
    import pytest

    from voice_assistant.config import ConfigError

    monkeypatch.setenv("LLM_MAX_TOKENS", "lots")
    with pytest.raises(ConfigError, match="LLM_MAX_TOKENS must be an integer"):
        Settings()


def test_check_ranges_reports_bad_values(monkeypatch) -> None:
    monkeypatch.setenv("CHUNK_MS", "25")
    monkeypatch.setenv("VAD_AGGRESSIVENESS", "7")
    monkeypatch.delenv("VAANI_PROFILE", raising=False)

    problems = Settings().check_ranges()

    assert any("CHUNK_MS" in p for p in problems)
    assert any("VAD_AGGRESSIVENESS" in p for p in problems)


def test_default_settings_are_valid(monkeypatch) -> None:
    for name in ("CHUNK_MS", "VAD_AGGRESSIVENESS", "ASR_ENDPOINT_SILENCE_MS", "VAANI_PROFILE", "BARGE_IN_MS"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings()

    assert settings.check_ranges() == []
    assert settings.barge_in_frames == settings.barge_in_ms // settings.chunk_ms


def test_chat_validation_only_needs_llm(monkeypatch, tmp_path) -> None:
    import pytest

    from voice_assistant.config import ConfigError

    model = tmp_path / "model.gguf"
    model.write_bytes(b"")
    monkeypatch.delenv("MOCK_MODELS", raising=False)
    monkeypatch.setenv("MODEL_PATH", str(model))
    monkeypatch.setenv("PIPER_VOICE", "")
    monkeypatch.setenv("ASR_MODEL_PATH", "")

    Settings().validate(need_asr=False, need_tts=False)
    with pytest.raises(ConfigError, match="PIPER_VOICE"):
        Settings().validate(need_asr=False, need_tts=True)
