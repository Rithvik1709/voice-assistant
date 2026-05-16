import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from voice_assistant.tts.queue import AudioChunkQueue
from voice_assistant.tts.stream import PiperConfig, PiperStreamingTTS


def _make_tts() -> PiperStreamingTTS:
    return PiperStreamingTTS(
        PiperConfig(voice_path=Path("fake.onnx")),
        queue=AudioChunkQueue(maxsize=4),
    )


def test_timeout_returns_empty_bytes():
    tts = _make_tts()
    with patch("voice_assistant.tts.stream.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="piper", timeout=30)):
        assert tts._run_piper("Hello world.") == b""


def test_process_error_returns_empty_bytes():
    tts = _make_tts()
    with patch("voice_assistant.tts.stream.subprocess.run", side_effect=subprocess.CalledProcessError(1, "piper", stderr=b"model not found")):
        assert tts._run_piper("Hello world.") == b""


def test_success_returns_pcm():
    tts = _make_tts()
    mock_proc = MagicMock()
    mock_proc.stdout = b"\x00\x01" * 100
    with patch("voice_assistant.tts.stream.subprocess.run", return_value=mock_proc):
        assert tts._run_piper("Hello world.") == b"\x00\x01" * 100
