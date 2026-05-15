from __future__ import annotations

import asyncio
import logging

import numpy as np

from voice_assistant.asr.stream import ASREvent, StreamingASR
from voice_assistant.benchmark import BenchmarkTracker
from voice_assistant.llm.client import StreamingLLMClient
from voice_assistant.tts.player import AudioPlayer
from voice_assistant.tts.queue import AudioChunk, AudioChunkQueue
from voice_assistant.tts.stream import PiperStreamingTTS, sentence_chunks_from_tokens

from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class VoicePipelineOrchestrator:
    def __init__(
        self,
        asr: StreamingASR,
        llm: StreamingLLMClient,
        tts: PiperStreamingTTS,
        player: AudioPlayer,
        bench: BenchmarkTracker,
        tts_backpressure_threshold: int = 3,
        tts_sentence_max_tokens: int = 8,
        tts_eager_min_words: int = 3,
        ack_tone_ms: int = 55,
    ) -> None:
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.player = player
        self.bench = bench
        self.tts_backpressure_threshold = tts_backpressure_threshold
        self.tts_sentence_max_tokens = tts_sentence_max_tokens
        self.tts_eager_min_words = tts_eager_min_words
        self.ack_tone_ms = ack_tone_ms

        self.partial_queue: asyncio.Queue[ASREvent] = asyncio.Queue(maxsize=64)
        self.prompt_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=8)
        self.token_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=256)
        self.audio_queue: AudioChunkQueue = tts.queue
        self.interrupt_event = asyncio.Event()

    async def asr_task(self) -> None:
        async for event in self.asr.stream_events():
            if event.type == "partial":
                if self.partial_queue.full():
                    _ = self.partial_queue.get_nowait()
                await self.partial_queue.put(event)
                continue

            if event.type == "final" and event.text.strip():
                self.bench.mark("final_text_ts")
                await self.prompt_queue.put(event.text)

    async def llm_task(self) -> None:
        while True:
            prompt = await self.prompt_queue.get()
            with tracer.start_as_current_span("orchestrator.process_prompt") as span:
                span.set_attribute("prompt.length", len(prompt))
                if self.interrupt_event.is_set():
                    self._drain_queue(self.token_queue)
                    self.interrupt_event.clear()
                self.bench.mark("prompt_sent_ts")
                if self.audio_queue.empty():
                    await self._enqueue_ack_tone()
                await self.llm.stream_tokens(prompt, self.token_queue)
                await self.token_queue.put("<eos>")

    async def tts_task(self) -> None:
        token_buf: list[str] = []
        while True:
            token = await self.token_queue.get()

            if self.interrupt_event.is_set():
                self._drain_queue(self.token_queue)
                token_buf.clear()
                self.audio_queue.clear()
                self.player.interrupt()
                self.interrupt_event.clear()
                continue

            if token == "<eos>":
                for sentence in sentence_chunks_from_tokens(token_buf):
                    await self.tts.synthesize_sentence(sentence)
                token_buf.clear()
                continue

            token_buf.append(token)
            if self.audio_queue.qsize() > self.tts_backpressure_threshold:
                await asyncio.sleep(0.005)
                continue

            ready = sentence_chunks_from_tokens(token_buf, max_tokens=self.tts_sentence_max_tokens)
            if ready:
                for sentence in ready[:-1]:
                    await self.tts.synthesize_sentence(sentence)
                token_buf = [ready[-1]]

            if token_buf and self._should_flush_eager(token_buf, token):
                eager_text = "".join(token_buf).strip()
                if eager_text:
                    await self.tts.synthesize_sentence(eager_text)
                    token_buf.clear()

    async def playback_task(self) -> None:
        await self.player.start()
        while True:
            chunk = await self.audio_queue.get()
            await self.player.play(chunk)

    async def run(self) -> None:
        tasks = [
            asyncio.create_task(self.asr_task()),
            asyncio.create_task(self.llm_task()),
            asyncio.create_task(self.tts_task()),
            asyncio.create_task(self.playback_task()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in pending:
            t.cancel()
        for t in done:
            exc = t.exception()
            if exc:
                raise exc

    @staticmethod
    def _drain_queue(q: asyncio.Queue[str]) -> None:
        while not q.empty():
            q.get_nowait()

    def _should_flush_eager(self, token_buf: list[str], latest_token: str) -> bool:
        text = "".join(token_buf).strip()
        if not text:
            return False
        word_count = len(text.split())
        boundary = latest_token.endswith((" ", "\n", ".", ",", "!", "?", ";", ":"))
        return word_count >= self.tts_eager_min_words and boundary

    async def _enqueue_ack_tone(self) -> None:
        duration = max(0.02, self.ack_tone_ms / 1000.0)
        sr = self.player.sample_rate
        samples = int(sr * duration)
        if samples <= 0:
            return
        t = np.arange(samples, dtype=np.float32) / float(sr)
        tone = 0.08 * np.sin(2.0 * np.pi * 880.0 * t)
        pcm16 = np.clip(tone * 32767.0, -32768.0, 32767.0).astype(np.int16).tobytes()
        if self.bench.current.first_audio_ts is None:
            self.bench.mark("first_audio_ts")
        await self.audio_queue.put(AudioChunk(pcm16=pcm16, sample_rate=sr))
