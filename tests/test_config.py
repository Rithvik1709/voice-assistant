import pytest

from voice_assistant.config import Settings


def test_chunk_size_is_calculated_from_sample_rate_and_chunk_ms():
    settings = Settings(sample_rate=16000, chunk_ms=20)

    assert settings.chunk_size == 320


def test_validate_raises_error_when_required_model_paths_are_missing():
    settings = Settings(model_path="", piper_voice="")

    with pytest.raises(ValueError, match="MODEL_PATH, PIPER_VOICE"):
        settings.validate()


def test_validate_raises_error_when_model_path_is_missing():
    settings = Settings(model_path="", piper_voice="models/en_US-lessac-medium.onnx")

    with pytest.raises(ValueError, match="MODEL_PATH"):
        settings.validate()


def test_validate_raises_error_when_piper_voice_is_missing():
    settings = Settings(model_path="models/model.gguf", piper_voice="")

    with pytest.raises(ValueError, match="PIPER_VOICE"):
        settings.validate()


def test_validate_passes_when_required_model_paths_are_present():
    settings = Settings(
        model_path="models/model.gguf",
        piper_voice="models/en_US-lessac-medium.onnx",
    )

    settings.validate()


def test_piper_voice_path_expands_user_directory_to_absolute_path():
    settings = Settings(piper_voice="~/models/en_US-lessac-medium.onnx")

    assert settings.piper_voice_path.is_absolute()
    assert "~" not in str(settings.piper_voice_path)