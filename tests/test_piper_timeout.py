import json
import subprocess
from pathlib import Path

import pytest

from voice_assistant.tts.queue import AudioChunkQueue
from voice_assistant.tts.stream import (
    PiperConfig,
    PiperProcess,
    PiperStreamingTTS,
    read_voice_sample_rate,
)


class _FakeProc:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0, hang: bool = False) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._hang = hang
        self.stdin_data: bytes | None = None
        self.killed = False

    def communicate(self, data: bytes | None = None, timeout: float | None = None):
        if self._hang and not self.killed:
            raise subprocess.TimeoutExpired("piper", timeout)
        self.stdin_data = data
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    def poll(self):
        return self.returncode


def _patch_popen(monkeypatch, proc: _FakeProc) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        calls.append(cmd)
        return proc

    monkeypatch.setattr("voice_assistant.tts.stream.subprocess.Popen", fake_popen)
    return calls


def test_piper_process_success(monkeypatch):
    pcm_data = b"\x00\x01" * 50
    proc = _FakeProc(stdout=pcm_data)
    calls = _patch_popen(monkeypatch, proc)

    res = PiperProcess(["piper", "--output_raw"]).synthesize("hello")

    assert res == pcm_data
    assert proc.stdin_data == b"hello\n"
    assert calls == [["piper", "--output_raw"]]


def test_piper_process_trims_odd_trailing_byte(monkeypatch):
    _patch_popen(monkeypatch, _FakeProc(stdout=b"\x00\x01\x02"))

    assert PiperProcess(["piper"]).synthesize("hello") == b"\x00\x01"


def test_piper_process_empty_text_skips_subprocess(monkeypatch):
    calls = _patch_popen(monkeypatch, _FakeProc())

    assert PiperProcess(["piper"]).synthesize("   ") == b""
    assert calls == []


def test_piper_process_crashed(monkeypatch):
    _patch_popen(monkeypatch, _FakeProc(stderr=b"boom: bad model\n", returncode=1))

    with pytest.raises(RuntimeError, match="exited with code 1: boom: bad model"):
        PiperProcess(["piper"]).synthesize("hello")


def test_piper_process_timeout_kills_process(monkeypatch):
    proc = _FakeProc(hang=True)
    _patch_popen(monkeypatch, proc)

    with pytest.raises(RuntimeError, match="timed out"):
        PiperProcess(["piper"], timeout_s=0.01).synthesize("hello")
    assert proc.killed


def test_piper_streaming_tts_init():
    q = AudioChunkQueue(maxsize=1)
    config = PiperConfig(voice_path="dummy", sample_rate=22050)
    tts = PiperStreamingTTS(config=config, playback_queue=q)
    assert tts.playback_queue is q
    assert tts.sample_rate == 22050


def test_sample_rate_is_read_from_voice_config(tmp_path: Path):
    voice = tmp_path / "voice.onnx"
    Path(f"{voice}.json").write_text(json.dumps({"audio": {"sample_rate": 16000}}), encoding="utf-8")

    assert read_voice_sample_rate(voice) == 16000
    tts = PiperStreamingTTS(PiperConfig(voice_path=voice), playback_queue=AudioChunkQueue())
    assert tts.sample_rate == 16000


def test_sample_rate_missing_config_returns_none(tmp_path: Path):
    assert read_voice_sample_rate(tmp_path / "missing.onnx") is None
