from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(slots=True)
class AudioChunk:
    pcm16: bytes
    sample_rate: int


class AudioChunkQueue:
    def __init__(self, maxsize: int = 6) -> None:
        self._q: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=maxsize)

    async def put(self, chunk: AudioChunk) -> None:
        await self._q.put(chunk)

    async def get(self) -> AudioChunk:
        return await self._q.get()

    def qsize(self) -> int:
        return self._q.qsize()

    def empty(self) -> bool:
        return self._q.empty()

    def clear(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except asyncio.QueueEmpty:
                break