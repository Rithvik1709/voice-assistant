from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AudioChunk:
    pcm16: bytes
    sample_rate: int
    debug_text: str = ""


class AudioChunkQueue:
    def __init__(self, maxsize: int = 32) -> None:
        self._q: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=maxsize)
        self._put_count = 0
        self._get_count = 0
        self._total_put_wait = 0.0
        self._total_get_wait = 0.0
        self._batch_sizes: list[int] = []
        self._dropped_count = 0

    async def put(self, chunk: AudioChunk) -> None:
        start = time.perf_counter()
        await self._q.put(chunk)
        self._put_count += 1
        self._total_put_wait += (time.perf_counter() - start)

    async def get(self) -> AudioChunk:
        start = time.perf_counter()
        chunk = await self._q.get()
        self._get_count += 1
        self._total_get_wait += (time.perf_counter() - start)
        return chunk

    def record_batch(self, size: int) -> None:
        self._batch_sizes.append(size)
        if len(self._batch_sizes) > 100:
            self._batch_sizes.pop(0)

    def record_drop(self) -> None:
        self._dropped_count += 1

    def qsize(self) -> int:
        return self._q.qsize()

    def empty(self) -> bool:
        return self._q.empty()

    def clear(self) -> None:
        while not self._q.empty():
            self._q.get_nowait()
        self._batch_sizes.clear()

    def get_metrics(self) -> dict:
        avg_batch = sum(self._batch_sizes) / len(self._batch_sizes) if self._batch_sizes else 0
        return {
            "qsize": self.qsize(),
            "put_count": self._put_count,
            "get_count": self._get_count,
            "dropped_count": self._dropped_count,
            "avg_put_latency": self._total_put_wait / self._put_count if self._put_count > 0 else 0,
            "avg_get_latency": self._total_get_wait / self._get_count if self._get_count > 0 else 0,
            "avg_batch_size": avg_batch,
            "max_batch_size": max(self._batch_sizes) if self._batch_sizes else 0
        }

async def safe_put(queue: AudioChunkQueue | asyncio.Queue, item: any, timeout: float = 0.1) -> bool:
    try:
        await asyncio.wait_for(queue.put(item), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        logger.warning("Queue full, dropping audio chunk")
        if isinstance(queue, AudioChunkQueue):
            queue.record_drop()
        return False
