import struct
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from voice_assistant.tts.stream import PiperConfig, PiperProcess, PiperStreamingTTS
from voice_assistant.tts.queue import AudioChunkQueue


@patch("voice_assistant.tts.stream.subprocess.Popen")
def test_piper_process_success(mock_popen):
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc
    mock_proc.poll.return_value = None

    data_size = 100
    wav_header = b"RIFF" + b"\x00" * 36 + struct.pack("<I", data_size)
    pcm_data = b"\x00\x01" * 50

    mock_proc.stdout.read.side_effect = [wav_header, pcm_data]

    proc = PiperProcess(["fake_cmd"])
    res = proc.synthesize("hello")

    assert res == pcm_data
    mock_proc.stdin.write.assert_called_once_with(b'{"text": "hello"}\n')


@patch("voice_assistant.tts.stream.subprocess.Popen")
def test_piper_process_short_header(mock_popen):
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc
    mock_proc.poll.return_value = None
    mock_proc.stdout.read.return_value = b"short"

    proc = PiperProcess(["fake_cmd"])
    res = proc.synthesize("hello")
    assert res == b""


@patch("voice_assistant.tts.stream.subprocess.Popen")
def test_piper_process_crashed(mock_popen):
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc
    mock_proc.poll.return_value = 1  # process exited

    proc = PiperProcess(["fake_cmd"])
    with pytest.raises(RuntimeError, match="Piper process exited unexpectedly"):
        proc.synthesize("hello")

def test_piper_streaming_tts_init():
    q = AudioChunkQueue(maxsize=1)
    config = PiperConfig(voice_path="dummy", sample_rate=22050)
    tts = PiperStreamingTTS(config=config, playback_queue=q)
    assert tts.playback_queue is q
