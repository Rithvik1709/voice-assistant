from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from voice_assistant.asr.stream import ASREvent, StreamingASR
    from voice_assistant.tts.player import AudioPlayer

from opentelemetry import trace

from voice_assistant.actions import ActionHandler
from voice_assistant.audio import make_ack_tone
from voice_assistant.benchmark import BenchmarkTracker
from voice_assistant.llm.client import StreamingLLMClient
from voice_assistant.memory import SessionMemory
from voice_assistant.tts.queue import AudioChunk, AudioChunkQueue, safe_put
from voice_assistant.tts.stream import PiperStreamingTTS, sentence_chunks_from_tokens

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

TTS_RETRY_DELAY_S = 0.5
FALLBACK_REPLY = "Sorry, something went wrong. Please try again."
# Room echo lingers briefly after playback; keep the mic muted a little longer.
ECHO_TAIL_S = 0.15


@dataclass(frozen=True, slots=True)
class EndOfTurn:
    """Marks the end of one turn's tokens in the token queue."""

    turn: int


class VoicePipelineOrchestrator:
    def __init__(
        self,
        asr: StreamingASR,
        llm: StreamingLLMClient,
        tts: PiperStreamingTTS,
        player: AudioPlayer,
        bench: BenchmarkTracker,
        nlu: Any | None = None,
        tts_backpressure_threshold: int = 3,
        tts_sentence_max_tokens: int = 8,
        tts_eager_min_words: int = 3,
        ack_tone_ms: int = 55,
        max_conversation_turns: int = 10,
        action_handler: ActionHandler | None = None,
        memory: SessionMemory | None = None,
        system_prompt: str = "",
        barge_in: bool = True,
        on_reply_token: Callable[[str], None] | None = None,
        on_turn_end: Callable[[], None] | None = None,
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
        self.barge_in = barge_in
        # Optional observers, used by text chat mode to print replies.
        self.on_reply_token = on_reply_token
        self.on_turn_end = on_turn_end

        self.partial_queue: asyncio.Queue[ASREvent] = asyncio.Queue(maxsize=64)
        self.prompt_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=8)
        self.token_queue: asyncio.Queue[str | EndOfTurn] = asyncio.Queue(maxsize=256)
        self.audio_queue: AudioChunkQueue = tts.playback_queue
        self.nlu = nlu
        self.action_handler = action_handler
        self.memory = memory
        self.system_prompt = system_prompt.strip()

        self.max_conversation_turns = max(1, max_conversation_turns)
        self.conversation_history = self._load_conversation_history()

        # Bumped on every interrupt so in-flight work can notice it is stale.
        self._generation = 0
        # Incremented when a turn starts; stale end-of-turn work is ignored.
        self._turn = 0
        self._responding = False
        self._response_task: asyncio.Task[None] | None = None
        self._finish_task: asyncio.Task[None] | None = None
        self._token_buf: list[str] = []
        self._reply_tokens: list[str] = []

    @property
    def is_responding(self) -> bool:
        return self._responding

    async def asr_task(self) -> None:
        async for event in self.asr.stream_events():
            if event.type == "speech_start":
                if self.barge_in and self._responding:
                    logger.info("Barge-in detected; interrupting response")
                    self.interrupt()
                continue

            if event.type == "partial":
                self._offer_partial(event)
                continue

            if event.type == "final" and event.text.strip():
                logger.info("User: %s", event.text)
                self.bench.reset()
                if event.speech_end_ts is not None:
                    self.bench.current.speech_end_ts = event.speech_end_ts
                self.bench.mark("final_text_ts")
                await self.prompt_queue.put(event.text)

    def _offer_partial(self, event: ASREvent) -> None:
        # Keep the queue non-blocking and prefer the freshest partials.
        if self.partial_queue.full():
            try:
                self.partial_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self.partial_queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    async def llm_task(self) -> None:
        while True:
            prompt = await self.prompt_queue.get()
            task = asyncio.create_task(self._respond(prompt))
            self._response_task = task
            try:
                # asyncio.wait does not raise when `task` is cancelled by an
                # interrupt, only when this loop itself is cancelled.
                await asyncio.wait({task})
            finally:
                if not task.done():
                    task.cancel()
            if not task.cancelled() and task.exception() is not None:
                logger.error("Response failed", exc_info=task.exception())

    async def _respond(self, prompt: str) -> None:
        with tracer.start_as_current_span("orchestrator.process_prompt") as span:
            span.set_attribute("prompt.length", len(prompt))
            turn = self._begin_response()

            intent: dict[str, object] | None = None
            if self.nlu is not None:
                try:
                    intent = self.nlu.classify(prompt)
                    if isinstance(intent, dict):
                        span.set_attribute("nlu.intent", str(intent.get("intent", "")))
                        span.set_attribute("nlu.confidence", float(intent.get("confidence", 0.0)))
                except Exception:
                    logger.exception("nlu classification failed")

            self.bench.mark("prompt_sent_ts")

            if self.audio_queue.empty():
                await self._enqueue_ack_tone()

            self._add_message("user", prompt)
            self._reply_tokens = []

            try:
                if intent is not None and self.action_handler is not None:
                    # Actions may do network I/O (weather), so keep them off the loop.
                    action_result = await asyncio.to_thread(self.action_handler.handle, prompt, intent)
                    if action_result.handled:
                        response = action_result.response.strip()
                        if response:
                            await self._emit_text_response(response)
                            self._add_message("assistant", response)
                        return

                reply = await self.llm.stream_tokens(self.conversation_history, self.token_queue)
                self._add_message("assistant", reply)
            except asyncio.CancelledError:
                # Interrupted: remember what the user actually heard.
                heard = "".join(self._reply_tokens).strip()
                if heard:
                    self._add_message("assistant", heard)
                raise
            except Exception:
                logger.exception("Assistant response failed")
                self._drain_queue(self.token_queue)
                await self._emit_text_response(FALLBACK_REPLY)
            finally:
                if not asyncio.current_task().cancelling():  # type: ignore[union-attr]
                    await self.token_queue.put(EndOfTurn(turn))

    async def tts_task(self) -> None:
        while True:
            token = await self.token_queue.get()
            generation = self._generation

            if isinstance(token, EndOfTurn):
                for sentence in sentence_chunks_from_tokens(self._token_buf):
                    await self._synthesize_with_retry(sentence)
                self._token_buf = []
                if generation == self._generation:
                    self._schedule_finish(token.turn)
                continue

            self._reply_tokens.append(token)
            if self.on_reply_token is not None:
                self.on_reply_token(token)
            self._token_buf.append(token)

            # Natural backpressure: synthesize_sentence awaits when the TTS
            # ingest queue is full, which slows this loop down.
            ready = sentence_chunks_from_tokens(
                self._token_buf,
                max_tokens=self.tts_sentence_max_tokens,
            )

            if len(ready) > 1:
                for sentence in ready[:-1]:
                    await self._synthesize_with_retry(sentence)
                    if generation != self._generation:
                        break
                if generation != self._generation:
                    continue
                self._token_buf = [ready[-1]]

            if self._token_buf and self._should_flush_eager(self._token_buf, token):
                eager_text = "".join(self._token_buf).strip()
                self._token_buf = []
                if eager_text:
                    await self._synthesize_with_retry(eager_text)

    async def playback_task(self) -> None:
        await self.player.start()

        try:
            while True:
                chunk = await self.audio_queue.get()
                await self.player.play(chunk)
        finally:
            await self.player.stop()

    async def run(self) -> None:
        await self.tts.start()

        tasks = [
            asyncio.create_task(self.asr_task()),
            asyncio.create_task(self.llm_task()),
            asyncio.create_task(self.tts_task()),
            asyncio.create_task(self.playback_task()),
        ]
        logger.info("Vaani is listening. Press Ctrl+C to stop.")

        try:
            # The worker loops run forever, so the first task to finish is
            # either a failure or the input source ending (e.g. /quit in chat).
            done, _pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for t in done:
                exc = t.exception()
                if exc:
                    raise exc

        except asyncio.CancelledError:
            logger.info("Orchestrator shutting down gracefully...")
            raise

        finally:
            for t in tasks:
                t.cancel()
            for t in (self._response_task, self._finish_task):
                if t is not None:
                    t.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)
            await self.tts.stop()

    def reset_conversation(self) -> None:
        """Forget the conversation (keeps the system prompt)."""
        self.conversation_history = [m for m in self.conversation_history if m.get("role") == "system"][:1]
        if self.memory is not None:
            self.memory.clear()

    def interrupt(self) -> None:
        """Stop the current response immediately (barge-in)."""
        self._generation += 1
        if self._response_task is not None and not self._response_task.done():
            self._response_task.cancel()
        if self._finish_task is not None and not self._finish_task.done():
            self._finish_task.cancel()
        self._drain_queue(self.token_queue)
        self._token_buf = []
        self.tts.cancel_pending()
        self.audio_queue.clear()
        self.player.interrupt()
        self._end_response()

    def _begin_response(self) -> int:
        self._turn += 1
        self._responding = True
        if not self.barge_in:
            self.asr.muted = True
        return self._turn

    def _end_response(self) -> None:
        self._responding = False
        self.asr.muted = False

    def _schedule_finish(self, turn: int) -> None:
        if turn != self._turn:
            return
        if self._finish_task is not None and not self._finish_task.done():
            self._finish_task.cancel()
        self._finish_task = asyncio.create_task(self._finish_response(turn))

    async def _finish_response(self, turn: int) -> None:
        """Mark the turn complete once its audio has finished playing."""
        await self.tts.flush()
        while not self.audio_queue.empty() or self.player.is_playing:
            await asyncio.sleep(0.02)
        if not self.barge_in:
            await asyncio.sleep(ECHO_TAIL_S)
        # A newer turn has started (or is queued): it owns the state now.
        if turn != self._turn or not self.prompt_queue.empty():
            return
        self._end_response()
        if self.on_turn_end is not None:
            self.on_turn_end()
        metrics = {k: v for k, v in self.bench.snapshot().items() if v is not None}
        if metrics:
            logger.info("Turn metrics: %s", metrics)

    @staticmethod
    def _drain_queue(q: asyncio.Queue) -> None:
        while not q.empty():
            q.get_nowait()

    def _add_message(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})
        self._remember(role, content)
        self._prune_conversation_history()

    def _prune_conversation_history(self) -> None:
        max_messages = self.max_conversation_turns * 2
        system_messages = [
            item for item in self.conversation_history
            if item.get("role") == "system"
        ]
        chat_messages = [
            item for item in self.conversation_history
            if item.get("role") != "system"
        ]

        if len(chat_messages) > max_messages:
            chat_messages = chat_messages[-max_messages:]

        self.conversation_history = system_messages[:1] + chat_messages

    def _load_conversation_history(self) -> list[dict[str, str]]:
        history = []
        if self.system_prompt:
            history.append({"role": "system", "content": self.system_prompt})

        if self.memory is None:
            return history

        history.extend(
            item for item in self.memory.load_recent(self.max_conversation_turns * 2)
            if item.get("role") != "system"
        )
        return history

    def _remember(self, role: str, content: str) -> None:
        if self.memory is not None:
            self.memory.append(role, content)

    async def _emit_text_response(self, response: str) -> None:
        words = response.split(" ")
        for i, word in enumerate(words):
            await self.token_queue.put(word)
            if i < len(words) - 1:
                await self.token_queue.put(" ")

    async def _synthesize_with_retry(
        self,
        sentence: str,
        retries: int = 2,
    ) -> None:
        for attempt in range(retries + 1):
            if await self.tts.synthesize_sentence(sentence):
                return

            if attempt < retries:
                logger.warning(
                    "TTS backpressure, retrying sentence (%d/%d): %s...",
                    attempt + 1, retries, sentence[:50],
                )
                await asyncio.sleep(TTS_RETRY_DELAY_S)

        logger.error(
            "Dropped sentence after %d retries due to TTS backpressure: %s...",
            retries, sentence[:50],
        )

    def _should_flush_eager(
        self,
        token_buf: list[str],
        latest_token: str,
    ) -> bool:
        text = "".join(token_buf).strip()

        if not text:
            return False

        word_count = len(text.split())
        boundary = latest_token.endswith(
            (" ", "\n", ".", ",", "!", "?", ";", ":")
        )

        return (
            word_count >= self.tts_eager_min_words
            and boundary
        )

    async def _enqueue_ack_tone(self) -> None:
        sr = self.player.sample_rate
        pcm16 = make_ack_tone(sr, self.ack_tone_ms)
        if not pcm16:
            return

        if self.bench.current.first_audio_ts is None:
            self.bench.mark("first_audio_ts")

        await safe_put(
            self.audio_queue,
            AudioChunk(
                pcm16=pcm16,
                sample_rate=sr,
            ),
        )
