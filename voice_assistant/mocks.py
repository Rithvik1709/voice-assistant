"""Stand-ins for the ASR, LLM and TTS models (MOCK_MODELS=1).

They let the gRPC server, chat mode, CI and benchmarks run end to end without
downloading any model.
"""
from __future__ import annotations

import asyncio
import logging
import time

from voice_assistant.benchmark import BenchmarkTracker
from voice_assistant.tts.queue import AudioChunk, AudioChunkQueue
from voice_assistant.tts.stream import PiperConfig

logger = logging.getLogger(__name__)



class MockRecognizer:
    """Stands in for Vosk: every utterance decodes to the same question."""

    TRANSCRIPT = "tell me something interesting"

    streaming = True

    def accept_waveform(self, frame: bytes) -> None:
        # Simulate short CPU decoding time.
        time.sleep(0.001)

    def partial_result(self) -> tuple[str, float]:
        return "tell me", 0.9

    def final_result(self, _utterance: bytes) -> tuple[str, float]:
        return self.TRANSCRIPT, 0.9

    def reset(self) -> None:
        pass


class MockLLMClient:
    REPLY = ["Hello", " this", " is", " a", " mock", " response", " from", " the", " assistant", "."]

    async def stream_tokens(
        self,
        messages: list[dict[str, str]] | str,
        out_queue: asyncio.Queue[str],
        bench: BenchmarkTracker | None = None,
    ) -> str:
        for i, tok in enumerate(self.REPLY):
            if i == 0 and bench is not None:
                bench.mark("first_token_ts")
            try:
                await asyncio.wait_for(out_queue.put(tok), timeout=10.0)
            except TimeoutError as exc:
                logger.error("LLM output queue backpressure; aborting generation task")
                raise RuntimeError("LLM output queue full; generation aborted") from exc
            await asyncio.sleep(0.005)
        return "".join(self.REPLY)


class MockPiperStreamingTTS:
    def __init__(self, config: PiperConfig | None, playback_queue: AudioChunkQueue, bench: BenchmarkTracker | None = None) -> None:
        self.playback_queue = playback_queue
        self.bench = bench
        self.sample_rate = 22050

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def synthesize_sentence(self, sentence: str) -> bool:
        # 100ms of 22050Hz 16-bit mono silence per sentence.
        pcm16 = b"\x00\x00" * 2205
        chunk = AudioChunk(pcm16=pcm16, sample_rate=self.sample_rate, debug_text=sentence)
        try:
            await self.playback_queue.put(chunk)
            return True
        except Exception:
            return False

    def cancel_pending(self) -> None:
        pass

    async def flush(self) -> None:
        pass
