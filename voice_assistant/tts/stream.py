from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from voice_assistant.benchmark import BenchmarkTracker
from voice_assistant.tts.queue import AudioChunk, AudioChunkQueue

_SENTENCE_SPLIT = re.compile(r"([.!?]+(?:\s+|$))")
_logger = logging.getLogger(__name__)
_PIPER_TIMEOUT_S = 30.0


def sentence_chunks_from_tokens(tokens: list[str], max_tokens: int = 28) -> list[str]:
    text = "".join(tokens).strip()
    if not text:
        return []

    # split with captures so we keep the delimiters
    parts = _SENTENCE_SPLIT.split(text)
    
    chunks = []
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i] + parts[i+1]
        if sentence.strip():
            chunks.append(sentence.strip())
    
    if len(parts) % 2 == 1 and parts[-1].strip():
        chunks.append(parts[-1].strip())

    if not chunks:
        return [text]

    out: list[str] = []
    # If we want to split into sentences EVEN IF they fit in max_tokens, 
    # we need to change the logic. The current logic joins them if they fit.
    # The test expectation is ["Hello there.", "How are you?", "I am fine!"]
    # which means it wants EXACTLY one sentence per chunk if possible.
    
    for chunk in chunks:
        words = chunk.split()
        if not words:
            continue
            
        if len(words) > max_tokens:
            for i in range(0, len(words), max_tokens):
                out.append(" ".join(words[i:i + max_tokens]))
        else:
            out.append(chunk)

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

        if pcm:
            await self.queue.put(AudioChunk(pcm16=pcm, sample_rate=self.config.sample_rate))
        _ = time.perf_counter() - start

    def _run_piper(self, sentence: str) -> bytes:
        cmd = ["piper", "--model", str(self.config.voice_path), "--output_raw"]
        try:
            proc = subprocess.run(
                cmd,
                input=sentence.encode("utf-8"),
                capture_output=True,
                check=True,
                timeout=_PIPER_TIMEOUT_S,
            )
            return proc.stdout
        except subprocess.TimeoutExpired:
            _logger.warning("Piper timed out after %.1fs for sentence: %r", _PIPER_TIMEOUT_S, sentence[:60])
            return b""
        except subprocess.CalledProcessError as exc:
            _logger.error("Piper failed (exit %d): %s", exc.returncode, exc.stderr.decode(errors="replace") if exc.stderr else "")
            return b""
        except OSError as exc:
            _logger.error("Piper executable error: %s", exc)
            return b""
