"""Tests for barge-in interrupt recovery — verifying that AudioPlayer.resume()
is correctly called so audio output is not permanently silenced after a
barge-in event.

Fixes: https://github.com/Rithvik1709/voice-assistant/issues/28
"""
from __future__ import annotations

import struct

import numpy as np

from voice_assistant.tts.player import AudioPlayer, PlaybackState
from voice_assistant.tts.queue import AudioChunk


def _pcm16_chunk(value: int = 1000, samples: int = 256, sample_rate: int = 22_050) -> AudioChunk:
    pcm = struct.pack("<" + "h" * samples, *([value] * samples))
    return AudioChunk(pcm16=pcm, sample_rate=sample_rate)


def test_player_resume_clears_interrupted_flag() -> None:
    """After interrupt(), the interrupted flag is True and audio is silenced.
    Calling resume() must clear it so playback can continue."""
    player = AudioPlayer(sample_rate=22_050, blocksize=128)
    assert player.state.interrupted is False

    player.interrupt()
    assert player.state.interrupted is True

    player.resume()
    assert player.state.interrupted is False


def test_player_interrupt_drains_pending_and_queue() -> None:
    """interrupt() should clear both the internal _pending buffer and the
    playback queue so stale audio is not played after resume."""
    player = AudioPlayer(sample_rate=22_050, blocksize=128)

    # Manually populate the internal queue
    audio = np.array([0.5] * 128, dtype=np.float32)
    player._queue.put_nowait(audio)
    assert not player._queue.empty()

    player._pending = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    player.interrupt()
    assert player._queue.empty()
    assert len(player._pending) == 0
    assert player.state.interrupted is True


def test_player_callback_silences_output_when_interrupted() -> None:
    """When state.interrupted is True, the audio callback must output zeros."""
    player = AudioPlayer(sample_rate=22_050, blocksize=64)
    player.state.interrupted = True

    outdata = np.zeros((64, 1), dtype=np.float32)
    player._callback(outdata, 64, None, None)

    assert np.all(outdata == 0.0)


def test_player_callback_outputs_audio_after_resume() -> None:
    """After resume(), the callback should output actual audio data again."""
    player = AudioPlayer(sample_rate=22_050, blocksize=64)

    # Queue some audio
    audio = np.ones(64, dtype=np.float32) * 0.5
    player._queue.put_nowait(audio)

    # Interrupt, then resume
    player.interrupt()
    assert player.state.interrupted is True

    player.resume()
    assert player.state.interrupted is False

    # Queue fresh audio after resume
    fresh_audio = np.ones(64, dtype=np.float32) * 0.3
    player._queue.put_nowait(fresh_audio)

    outdata = np.zeros((64, 1), dtype=np.float32)
    player._callback(outdata, 64, None, None)

    # Output should not be all zeros
    assert np.any(outdata != 0.0)
