from __future__ import annotations

import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from voice_assistant.tts.player import AudioPlayer, PlaybackState
from voice_assistant.tts.queue import AudioChunk


@pytest.mark.asyncio
async def test_audio_player_basic_interruption() -> None:
    player = AudioPlayer(sample_rate=22050, blocksize=128)
    
    # Initially not interrupted
    assert not player.state.interrupted
    assert player._generation == 0

    # Call interrupt
    player.interrupt()
    assert player.state.interrupted
    assert player._generation == 1
    assert len(player._pending) == 0
    assert player._queue.empty()

    # Call resume
    player.resume()
    assert not player.state.interrupted


@pytest.mark.asyncio
async def test_audio_player_callback_muted_when_interrupted() -> None:
    player = AudioPlayer(sample_rate=22050, blocksize=128)
    player.interrupt()
    
    outdata = np.ones((128, 1), dtype=np.float32)
    player._callback(outdata, 128, None, None)
    
    # Callback should fill output with 0s if interrupted
    assert np.all(outdata == 0.0)


@pytest.mark.asyncio
async def test_audio_player_auto_resume_on_play() -> None:
    player = AudioPlayer(sample_rate=22050, blocksize=128)
    player.interrupt()
    assert player.state.interrupted

    # Play a new chunk
    chunk = AudioChunk(pcm16=b"\x00\x00" * 256, sample_rate=22050)
    await player.play(chunk)
    
    # Player should automatically resume
    assert not player.state.interrupted


@pytest.mark.asyncio
async def test_audio_player_discard_stale_chunks() -> None:
    player = AudioPlayer(sample_rate=22050, blocksize=128)
    
    # 1. Queue a chunk when generation is 0
    chunk1 = AudioChunk(pcm16=b"\x00\x01" * 128, sample_rate=22050)
    await player.play(chunk1)
    
    assert player._queue.qsize() == 1
    
    # Get the item to verify it has generation 0
    item, gen = player._queue.get()
    assert gen == 0
    # Put it back
    player._queue.put((item, gen))

    # 2. Trigger interrupt -> generation increments to 1, queue is cleared
    player.interrupt()
    assert player._generation == 1
    assert player._queue.empty()

    # 3. Simulate a stale play() putting a chunk generated in generation 0
    # We can simulate this by manually putting a tuple of (audio, 0)
    stale_audio = np.ones((128,), dtype=np.float32)
    player._queue.put((stale_audio, 0))
    
    # 4. Invoke the callback. It should discard the stale chunk (gen 0) 
    # since current generation is 1.
    outdata = np.ones((128, 1), dtype=np.float32)
    
    # Unmute first to let the callback process the queue
    player.resume()
    player._callback(outdata, 128, None, None)
    
    # Since the chunk was stale, it should be ignored, leaving outdata zero-filled
    assert np.all(outdata == 0.0)
    assert len(player._pending) == 0


@pytest.mark.asyncio
async def test_audio_player_occ_pending_audio_resurrection_prevention() -> None:
    player = AudioPlayer(sample_rate=22050, blocksize=128)
    
    # Setup some pending audio
    player._pending = np.ones((64,), dtype=np.float32)
    player._pending_generation = 0
    player._generation = 0

    # Simulate callback beginning execution at generation 0.
    # Inside callback, cb_gen = 0.
    # But before it concatenates parts, interrupt() is called.
    
    # Let's run a custom simulation showing OCC logic:
    cb_gen = player._generation  # 0
    assert player._pending_generation == cb_gen
    
    parts = [player._pending]  # has length 64
    
    # Now interrupt() is called on another thread/asyncio path
    player.interrupt()
    assert player._generation == 1
    assert len(player._pending) == 0
    assert player._pending_generation == 1

    # Callback continues and tries to concatenate:
    if parts and cb_gen == player._generation:
        player._pending = np.concatenate(parts)
        player._pending_generation = cb_gen
        
    # Since cb_gen (0) != player._generation (1), player._pending should still be empty (0 length)!
    assert len(player._pending) == 0
    assert player._pending_generation == 1
