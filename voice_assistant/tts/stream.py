from __future__ import annotations

import asyncio
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from voice_assistant.benchmark import BenchmarkTracker
from voice_assistant.tts.queue import AudioChunk, AudioChunkQueue

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentence_chunks_from_tokens(tokens: list[str], max_tokens: int = 28) -> list[str]:
    text = "".join(tokens).strip()
    if not text:
        return []

    words = text.split()
    out: list[str] = []
    buf: list[str] = []
    
    for word in words:
        buf.append(word)
        # Check if the word ends with sentence-ending punctuation
        is_complete = word[-1] in {".", "!", "?"}
        
        if len(buf) >= max_tokens or is_complete:
            out.append(" ".join(buf))
            buf = []
            
    if buf:
        out.append(" ".join(buf))
    return out


@dataclass(slots=True)
class PiperConfig:
    voice_path: Path
    sample_rate: int = 22_050


class PiperStreamingTTS:
    def __init__(self, config: PiperConfig, queue: AudioChunkQueue, bench: BenchmarkTracker | None = None) -> None:
        self.config = config
        self.queue = queue
        self.bench = bench

    async def synthesize_sentence(self, sentence: str) -> None:
        if not sentence.strip():
            return

        if self.bench and self.bench.current.tts_start_ts is None:
            self.bench.mark("tts_start_ts")

        start = time.perf_counter()
        pcm = await asyncio.to_thread(self._run_piper, sentence)

        if self.bench and self.bench.current.first_audio_ts is None:
            self.bench.mark("first_audio_ts")

        dur_sec = len(pcm) / 2 / self.config.sample_rate
        if self.bench:
            self.bench.add_synthesized_audio(dur_sec)
            self.bench.current.tts_end_ts = time.perf_counter()

        await self.queue.put(AudioChunk(pcm16=pcm, sample_rate=self.config.sample_rate))
        _ = time.perf_counter() - start

    def _run_piper(self, sentence: str) -> bytes:
        cmd = [
            "piper",
            "--model",
            str(self.config.voice_path),
            "--output_raw",
        ]
        proc = subprocess.run(cmd, input=sentence.encode("utf-8"), capture_output=True, check=True)
        return proc.stdout
