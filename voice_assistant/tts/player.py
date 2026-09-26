from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any

import numpy as np
import sounddevice as sd

from voice_assistant.tts.queue import AudioChunk


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or len(audio) == 0:
        return audio
    n_out = max(1, int(round(len(audio) * dst_rate / src_rate)))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


class AudioPlayer:
    def __init__(self, sample_rate: int = 22_050, blocksize: int = 128) -> None:
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._pending = np.array([], dtype=np.float32)
        self._state_lock = threading.Lock()
        self._stream: sd.OutputStream | None = None

    def _callback(
        self,
        outdata: Any,
        frames: int,
        _time: Any,
        _status: sd.CallbackFlags,
    ) -> None:
        with self._state_lock:
            pending = self._pending

            if len(pending) < frames:
                parts = [pending]
                needed = frames - len(pending)

                while needed > 0:
                    try:
                        nxt = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    parts.append(nxt)
                    needed -= len(nxt)

                pending = np.concatenate(parts)

            take = min(frames, len(pending))
            outdata.fill(0)
            if take:
                outdata[:take, 0] = pending[:take]
            self._pending = pending[take:]

    @property
    def is_playing(self) -> bool:
        """True while audio is buffered or waiting to be played."""
        with self._state_lock:
            return len(self._pending) > 0 or not self._queue.empty()

    async def start(self) -> None:
        if self._stream is not None:
            return

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
        audio = resample_linear(audio, chunk.sample_rate, self.sample_rate)

        while True:
            try:
                self._queue.put_nowait(audio)
                break
            except queue.Full:
                await asyncio.sleep(0.005)

    async def wait_until_drained(self, poll_s: float = 0.02) -> None:
        while self.is_playing:
            await asyncio.sleep(poll_s)

    async def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()

    def interrupt(self) -> None:
        """Immediately silence playback by discarding everything buffered.

        Playback continues normally with the next chunk passed to `play()`.
        """
        with self._state_lock:
            self._pending = np.array([], dtype=np.float32)
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    def resume(self) -> None:
        """Kept for API compatibility; `interrupt()` no longer latches silence."""
