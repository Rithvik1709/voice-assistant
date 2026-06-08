from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd

from voice_assistant.tts.queue import AudioChunk


@dataclass(slots=True)
class PlaybackState:
    interrupted: bool = False


class AudioPlayer:
    def __init__(self, sample_rate: int = 22_050, blocksize: int = 128) -> None:
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._pending = np.array([], dtype=np.float32)
        self.state = PlaybackState()
        self._state_lock = threading.Lock()
        self._active = False

    def _callback(
        self,
        outdata: Any,
        frames: int,
        _time: Any,
        _status: sd.CallbackFlags,
    ) -> None:
        with self._state_lock:
            interrupted = self.state.interrupted

        if interrupted:
            outdata.fill(0)
            return

        with self._state_lock:
            pending = self._pending

        if len(pending) < frames:
            parts = [pending]
            needed = frames - len(pending)

            while needed > 0:
                try:
                    nxt = self._queue.get_nowait()
                    parts.append(nxt)
                    needed -= len(nxt)
                except queue.Empty:
                    break

            pending = np.concatenate(parts) if parts else pending

        if len(pending) == 0:
            outdata.fill(0)
            return

        out = np.zeros((frames,), dtype=np.float32)
        take = min(frames, len(pending))

        out[:take] = pending[:take]
        remaining = pending[take:]

        with self._state_lock:
            self._pending = remaining

        outdata[:, 0] = out

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
        audio = np.frombuffer(
            chunk.pcm16,
            dtype=np.int16,
        ).astype(np.float32) / 32768.0

        while True:
            try:
                self._queue.put_nowait(audio)
                break
            except queue.Full:
                await asyncio.sleep(0.005)

    async def stop(self) -> None:
        self._active = False

        if hasattr(self, "_stream"):
            self._stream.stop()
            self._stream.close()

    def interrupt(self) -> None:
        with self._state_lock:
            self.state.interrupted = True
            self._pending = np.array([], dtype=np.float32)

        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def resume(self) -> None:
        with self._state_lock:
            self.state.interrupted = False
            