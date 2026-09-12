from pathlib import Path

from voice_assistant.model_setup import env_template, format_model_plan, write_env_file


def test_env_template_uses_local_open_model_paths() -> None:
    text = env_template(Path("models"))

    assert 'MODEL_PATH="models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"' in text
    assert 'PIPER_VOICE="models/en_US-lessac-medium.onnx"' in text
    assert 'ASR_MODEL_PATH="models/vosk-model-small-en-us-0.15"' in text
    assert 'CONVERSATION_MEMORY_PATH="data/session.jsonl"' in text


def test_write_env_file_does_not_overwrite_by_default(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("KEEP=1\n", encoding="utf-8")

    wrote = write_env_file(env_file, Path("models"))

    assert not wrote
    assert env_file.read_text(encoding="utf-8") == "KEEP=1\n"


def test_model_plan_mentions_open_model_licenses() -> None:
    plan = format_model_plan(Path("models"))

    assert "Apache-2.0" in plan
    assert "MIT" in plan
