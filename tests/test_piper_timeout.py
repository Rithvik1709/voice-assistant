import io
import struct
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from voice_assistant.tts.stream import PiperConfig, PiperProcess, PiperStreamingTTS
from voice_assistant.tts.queue import AudioChunkQueue


class _ReadStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def read(self, _size: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


def test_piper_process_success(monkeypatch):
    stdin = io.BytesIO()

    data_size = 100
    wav_header = b"RIFF" + b"\x00" * 36 + struct.pack("<I", data_size)
    pcm_data = b"\x00\x01" * 50
    mock_proc = SimpleNamespace(
        stdin=stdin,
        stdout=_ReadStream([wav_header, pcm_data]),
        poll=Mock(return_value=None),
    )
    monkeypatch.setattr(
        "voice_assistant.tts.stream.subprocess.Popen",
        lambda *args, **kwargs: mock_proc,
    )

    proc = PiperProcess(["fake_cmd"])
    res = proc.synthesize("hello")

    assert res == pcm_data
    assert stdin.getvalue() == b'{"text": "hello"}\n'


def test_piper_process_short_header(monkeypatch):
    mock_proc = SimpleNamespace(
        stdin=io.BytesIO(),
        stdout=_ReadStream([b"short"]),
        poll=Mock(return_value=None),
    )
    monkeypatch.setattr(
        "voice_assistant.tts.stream.subprocess.Popen",
        lambda *args, **kwargs: mock_proc,
    )

    proc = PiperProcess(["fake_cmd"])
    res = proc.synthesize("hello")
    assert res == b""


def test_piper_process_crashed(monkeypatch):
    mock_proc = SimpleNamespace(
        stdin=io.BytesIO(),
        stdout=_ReadStream([]),
        poll=Mock(return_value=1),
    )
    monkeypatch.setattr(
        "voice_assistant.tts.stream.subprocess.Popen",
        lambda *args, **kwargs: mock_proc,
    )

    proc = PiperProcess(["fake_cmd"])
    with pytest.raises(RuntimeError, match="Piper process exited unexpectedly"):
        proc.synthesize("hello")


def test_piper_streaming_tts_init():
    q = AudioChunkQueue(maxsize=1)
    config = PiperConfig(voice_path="dummy", sample_rate=22050)
    tts = PiperStreamingTTS(config=config, playback_queue=q)
    assert tts.playback_queue is q
