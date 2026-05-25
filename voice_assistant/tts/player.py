from __future__ import annotations
 
import asyncio
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any
 
import numpy as np
import sounddevice as sd
 
from voice_assistant.tts.queue import AudioChunk

logger = logging.getLogger(__name__)
 
 
@dataclass(slots=True)
class PlaybackState:
    interrupted: bool = False
 
 
class AudioPlayer:
    def __init__(self, sample_rate: int = 22_050, blocksize: int = 128) -> None:
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self._queue: queue.Queue[tuple[np.ndarray, int]] = queue.Queue(maxsize=64)
        self._pending = np.array([], dtype=np.float32)
        self._pending_generation = 0
        self._generation = 0
        self._lock = threading.Lock()
        self.state = PlaybackState()
        self._active = False
 
    def _callback(self, outdata: Any, frames: int, _time: Any, _status: sd.CallbackFlags) -> None:
        if self.state.interrupted:
            outdata.fill(0)
            return
 
        cb_gen = self._generation
 
        # If pending generation is stale, discard it
        if self._pending_generation != cb_gen:
            self._pending = np.array([], dtype=np.float32)
            self._pending_generation = cb_gen
 
        if len(self._pending) < frames:
            parts = [self._pending]
            needed = frames - len(self._pending)
            while needed > 0:
                try:
                    nxt = self._queue.get_nowait()
                    nxt_audio, nxt_gen = nxt
                    if nxt_gen == cb_gen:
                        parts.append(nxt_audio)
                        needed -= len(nxt_audio)
                except queue.Empty:
                    break
            if parts and cb_gen == self._generation:
                self._pending = np.concatenate(parts)
                self._pending_generation = cb_gen
 
        if cb_gen != self._generation:
            outdata.fill(0)
            return
 
        if len(self._pending) == 0:
            outdata.fill(0)
            return
 
        out = np.zeros((frames,), dtype=np.float32)
        take = min(frames, len(self._pending))
        out[:take] = self._pending[:take]
        
        if cb_gen == self._generation:
            self._pending = self._pending[take:]
            self._pending_generation = cb_gen
            outdata[:, 0] = out
        else:
            outdata.fill(0)
 
    async def start(self) -> None:
        if self._active:
            return
        self._active = True
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            blocksize=self.blocksize,
        )
        self._stream.start()
 
    async def play(self, chunk: AudioChunk) -> None:
        audio = np.frombuffer(chunk.pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        
        with self._lock:
            start_gen = self._generation
            # Auto-resume when playing new audio of the current active generation
            if self.state.interrupted:
                self.state.interrupted = False
                logger.info("Auto-resumed/unmuted AudioPlayer on play()")
 
        while True:
            with self._lock:
                if self._generation != start_gen:
                    logger.warning(
                        f"Discarded audio chunk playback due to generation change "
                        f"({start_gen} -> {self._generation})"
                    )
                    return
            try:
                self._queue.put_nowait((audio, start_gen))
                break
            except queue.Full:
                await asyncio.sleep(0.005)
 
    async def stop(self) -> None:
        self._active = False
        if hasattr(self, "_stream"):
            self._stream.stop()
            self._stream.close()
 
    def interrupt(self) -> None:
        with self._lock:
            self.state.interrupted = True
            self._generation += 1
            self._pending = np.array([], dtype=np.float32)
            self._pending_generation = self._generation
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            logger.info(f"AudioPlayer interrupted. Transitioned to generation {self._generation}")
 
    def resume(self) -> None:
        with self._lock:
            self.state.interrupted = False
            logger.info("AudioPlayer manually resumed")
