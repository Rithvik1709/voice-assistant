import struct
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from voice_assistant.tts.stream import PiperConfig, PiperProcess, PiperStreamingTTS
from voice_assistant.tts.queue import AudioChunkQueue


def test_piper_process_success():
    # Construct a valid WAV header with 200 bytes of PCM data
    # Subchunk2Size is at offset 40-43
    header = bytearray(44)
    struct.pack_into('<I', header, 40, 200)
    mock_pcm = b"\x00\x01" * 100
    
    mock_proc = MagicMock()
    mock_proc.stdin = MagicMock()
    mock_proc.stdout = MagicMock()
    mock_proc.stdout.read.side_effect = [header, mock_pcm]
    mock_proc.poll.return_value = None

    with patch("voice_assistant.tts.stream.subprocess.Popen", return_value=mock_proc):
        proc = PiperProcess(["piper"])
        res = proc.synthesize("Hello world.")
        assert res == mock_pcm


def test_piper_process_exited():
    mock_proc = MagicMock()
    mock_proc.stdin = MagicMock()
    mock_proc.stdout = MagicMock()
    mock_proc.poll.return_value = 1 # process exited

    with patch("voice_assistant.tts.stream.subprocess.Popen", return_value=mock_proc):
        proc = PiperProcess(["piper"])
        try:
            proc.synthesize("Hello world.")
        except RuntimeError as e:
            assert "exited unexpectedly" in str(e)
