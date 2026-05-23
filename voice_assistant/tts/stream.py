from __future__ import annotations

import asyncio
import logging
import re
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
import json

from voice_assistant.benchmark import BenchmarkTracker
from voice_assistant.tts.queue import AudioChunk, AudioChunkQueue, safe_put

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"([.!?]+(?:\s+|$))")
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

class PiperProcess:
    def __init__(self, cmd: list[str]):
        # We must NOT use --output_raw because we need the WAV header to know the length
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0
        )

    def synthesize(self, text: str) -> bytes:
        if not self.proc.stdin or not self.proc.stdout:
            return b""
            
        payload = json.dumps({"text": text}) + "\n"
        self.proc.stdin.write(payload.encode("utf-8"))
        self.proc.stdin.flush()

        # Read WAV header to determine chunk size (44 bytes)
        # Note: Assumes standard PCM WAV output; parsing validated against current Piper behavior.
        # If Piper changes header format (e.g., adds extra chunks), this could break.
        if self.proc.poll() is not None:
            raise RuntimeError("Piper process exited unexpectedly")
        header = self.proc.stdout.read(44)
        if len(header) < 44:
            return b""
            
        # Parse Subchunk2Size (bytes 40-43, little endian)
        data_size = struct.unpack('<I', header[40:44])[0]
        
        # Read the exact amount of PCM data
        if self.proc.poll() is not None:
            raise RuntimeError("Piper process exited unexpectedly")
        pcm = self.proc.stdout.read(data_size)
        return pcm

class SentenceBatcher:
    def __init__(self, max_batch_size: int = 10, max_wait_ms: int = 50):
        self.buffer: list[str] = []
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000.0
        self.last_flush = time.monotonic()

    def add(self, sentence: str) -> list[str] | None:
        self.buffer.append(sentence)

        if len(self.buffer) >= self.max_batch_size:
            return self.flush()

        return None

    def flush(self) -> list[str] | None:
        if not self.buffer:
            return None

        batch = self.buffer
        self.buffer = []
        self.last_flush = time.monotonic()
        return batch

class PiperStreamingTTS:
    def __init__(self, config: PiperConfig, playback_queue: AudioChunkQueue, bench: BenchmarkTracker | None = None) -> None:
        self.config = config
        self.playback_queue = playback_queue
        self.bench = bench
        self.ingest_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=32)
        
        self._cmd = [
            "piper",
            "--model", str(self.config.voice_path),
            "--json-input"
        ]
        
        self._batcher = SentenceBatcher(max_batch_size=10, max_wait_ms=50)
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._num_workers = 1 # Keep at 1 unless we need parallel synthesis
        self._flush_event = asyncio.Event()

    async def start(self):
        if self._running:
            return
        self._running = True
        self._workers = [asyncio.create_task(self._tts_worker()) for _ in range(self._num_workers)]

    async def stop(self):
        self._running = False
        for w in self._workers:
            w.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

    async def synthesize_sentence(self, sentence: str) -> bool:
        """
        If the system is under heavy load, synthesize_sentence() may raise RuntimeError.
        """
        if not sentence.strip():
            return True
        
        try:
            await asyncio.wait_for(self.ingest_queue.put(sentence), timeout=5.0)
            return True
        except asyncio.TimeoutError as exc:
            logger.error("TTS ingest queue full backpressure; aborting synthesis task")
            raise RuntimeError("TTS ingest queue full; synthesis aborted") from exc

    async def _tts_worker(self):
        while self._running:
            # 1 Piper process per worker to avoid stdout/stdin interleaving
            piper = PiperProcess(self._cmd)
            
            try:
                while self._running:
                    try:
                        # Wait for sentence or timeout to flush batch
                        timeout = max(0.01, self._batcher.max_wait - (time.monotonic() - self._batcher.last_flush))
                        
                        # If we have a manual flush event, don't wait
                        if self._flush_event.is_set():
                            sentence = self.ingest_queue.get_nowait()
                        else:
                            sentence = await asyncio.wait_for(self.ingest_queue.get(), timeout=timeout)
                        
                        try:
                            batch = self._batcher.add(sentence)
                            if batch:
                                await self._process_batch(batch, piper)
                        finally:
                            self.ingest_queue.task_done()
                        
                    except asyncio.TimeoutError:
                        # Time to flush if buffer isn't empty
                        batch = self._batcher.flush()
                        if batch:
                            await self._process_batch(batch, piper)
                    except asyncio.QueueEmpty:
                        # Manual flush triggered but queue was empty
                        batch = self._batcher.flush()
                        if batch:
                            await self._process_batch(batch, piper)
                        self._flush_event.clear()
                    except Exception as e:
                        # Catch any synthesis errors (BrokenPipeError, etc.)
                        logger.error(f"TTS worker encountered error: {e}", exc_info=True)
                        # Attempt to flush any pending batch
                        batch = self._batcher.flush()
                        if batch:
                            logger.warning(f"Dropping {len(batch)} sentences due to worker error")
                        # Break inner loop to restart the piper process
                        break
            finally:
                # Ensure Piper subprocess is always terminated before potentially restarting
                if piper.proc and piper.proc.stdin:
                    try:
                        piper.proc.stdin.close()
                    except Exception:
                        pass
                if piper.proc:
                    try:
                        piper.proc.terminate()
                        piper.proc.wait(timeout=2)
                    except Exception:
                        try:
                            piper.proc.kill()
                        except Exception:
                            pass
                logger.info("Piper process cleaned up")
                
        logger.info("TTS worker stopped completely")

    def _split_audio_chunks(self, audio_bytes: bytes, chunk_size: int = 32768):
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i:i+chunk_size]

    async def _process_batch(self, batch: list[str], piper: PiperProcess) -> None:
        if not batch:
            return
            
        combined = " ".join(batch)
        self.playback_queue.record_batch(len(batch))
        
        metrics = self.playback_queue.get_metrics()
        logger.info(
            f"TTS | ingest_q={self.ingest_queue.qsize()} "
            f"playback_q={self.playback_queue.qsize()} "
            f"drops={metrics['dropped_count']} "
            f"batch={len(batch)}"
        )

        if self.bench and self.bench.current.tts_start_ts is None:
            self.bench.mark("tts_start_ts")

        pcm = await asyncio.to_thread(piper.synthesize, combined)

        if not pcm:
            return

        if self.bench and self.bench.current.first_audio_ts is None:
            self.bench.mark("first_audio_ts")

        dur_sec = len(pcm) / 2 / self.config.sample_rate
        if self.bench:
            self.bench.add_synthesized_audio(dur_sec)
            self.bench.current.tts_end_ts = time.perf_counter()

        chunks = list(self._split_audio_chunks(pcm))
        for i, chunk in enumerate(chunks):
            text_metadata = combined if i == 0 else ""
            success = await safe_put(
                self.playback_queue, 
                AudioChunk(pcm16=chunk, sample_rate=self.config.sample_rate, debug_text=text_metadata),
                timeout=5.0
            )
            if not success:
                logger.error("TTS playback queue full backpressure; aborting synthesis task")
                raise RuntimeError("TTS playback queue full; synthesis aborted")

    async def flush(self) -> None:
        """Wait until all queued sentences have been processed."""
        # Signal workers to flush immediately
        self._flush_event.set()
        # Wait for all items currently in queue to be marked as done
        await self.ingest_queue.join()
        # Clear the flush event for next time
        self._flush_event.clear()

