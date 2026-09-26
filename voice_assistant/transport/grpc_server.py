from __future__ import annotations

import asyncio
import logging
import signal
import time
from collections.abc import AsyncIterator
from typing import Any

import grpc

from voice_assistant.asr.stream import ASREvent, StreamingASR, _VoskRecognizer
from voice_assistant.asr.vad import VADConfig, VoiceActivityDetector
from voice_assistant.audio import make_ack_tone
from voice_assistant.benchmark import BenchmarkTracker
from voice_assistant.config import Settings, mock_models_enabled
from voice_assistant.llm.client import LLMConfig, StreamingLLMClient, warm_up_llm
from voice_assistant.mocks import MockLLMClient, MockPiperStreamingTTS, MockRecognizer  # noqa: F401 (re-exported)
from voice_assistant.nlu import SimpleIntentClassifier
from voice_assistant.tts.queue import AudioChunkQueue
from voice_assistant.tts.stream import PiperConfig, PiperStreamingTTS, sentence_chunks_from_tokens

logger = logging.getLogger(__name__)

try:
    from voice_assistant.transport import voice_assistant_pb2 as pb2
    from voice_assistant.transport import voice_assistant_pb2_grpc as pb2_grpc
except Exception as exc:  # pragma: no cover - runtime setup
    raise RuntimeError(
        "Protobuf stubs are missing. Run grpc_tools.protoc using voice_assistant/transport/voice_assistant.proto"
    ) from exc

try:
    from vosk import Model  # type: ignore
    _VOSK_AVAILABLE = True
except ImportError:
    _VOSK_AVAILABLE = False

EOS = "<eos>"
FALLBACK_REPLY = "Sorry, something went wrong. Please try again."


# =====================================================================
# Per-stream session
# =====================================================================

class _VoiceSession:
    """State and tasks for one StreamVoice call."""

    def __init__(self, service: VoiceAssistantService) -> None:
        self.service = service
        self.settings = service.settings
        self.bench = BenchmarkTracker()
        self.history: list[dict[str, str]] = []
        if self.settings.assistant_system_prompt.strip():
            self.history.append({"role": "system", "content": self.settings.assistant_system_prompt.strip()})

        mock = mock_models_enabled()
        vad = VoiceActivityDetector(
            VADConfig(
                sample_rate=self.settings.sample_rate,
                frame_ms=self.settings.chunk_ms,
                aggressiveness=self.settings.vad_aggressiveness,
                # Mock mode uses energy VAD so synthetic audio is detected reliably.
                mode="energy" if mock else "webrtc",
            )
        )
        recognizer = MockRecognizer() if mock else _VoskRecognizer(
            self.settings.asr_model_path, self.settings.sample_rate, model=service.vosk_model
        )
        self.asr = StreamingASR(
            sample_rate=self.settings.sample_rate,
            chunk_size=self.settings.chunk_size,
            vad=vad,
            model_path=self.settings.asr_model_path,
            endpoint_silence_ms=self.settings.asr_endpoint_silence_ms,
            speech_start_frames=self.settings.barge_in_frames,
            recognizer=recognizer,
        )

        self.tts_queue = AudioChunkQueue(maxsize=self.settings.tts_queue_maxsize)
        if mock:
            self.tts: Any = MockPiperStreamingTTS(None, self.tts_queue, bench=self.bench)
        else:
            self.tts = PiperStreamingTTS(
                PiperConfig(self.settings.piper_voice_path, self.settings.tts_sample_rate),
                playback_queue=self.tts_queue,
                bench=self.bench,
            )

        self.token_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=256)
        self.responses: asyncio.Queue[pb2.AudioResponse | None] = asyncio.Queue()
        self.response_task: asyncio.Task[None] | None = None
        self.pump_task: asyncio.Task[None] | None = None
        # Whether audio was sent since the user last spoke; if so a barge-in
        # tells the client to drop whatever it still has buffered.
        self.sent_audio = False
        self.pending_frame = bytearray()

    # ----- lifecycle -------------------------------------------------

    async def start(self) -> None:
        await self.tts.start()
        self.pump_task = asyncio.create_task(self._pump_audio())

    async def close(self) -> None:
        self.cancel_response()
        if self.pump_task is not None:
            self.pump_task.cancel()
            await asyncio.gather(self.pump_task, return_exceptions=True)
        await self.tts.stop()

    # ----- input -----------------------------------------------------

    async def feed(self, pcm16: bytes) -> None:
        """Split incoming audio into VAD frames regardless of chunk size."""
        self.pending_frame.extend(pcm16)
        frame_bytes = self.asr.vad.frame_bytes
        while len(self.pending_frame) >= frame_bytes:
            frame = bytes(self.pending_frame[:frame_bytes])
            del self.pending_frame[:frame_bytes]
            events = await asyncio.to_thread(self.asr.process_frame, frame)
            for event in events:
                await self._handle_event(event)

    async def end_of_input(self) -> None:
        """The client half-closed: answer whatever was said, then drain."""
        for event in await asyncio.to_thread(self.asr.finalize):
            await self._handle_event(event)
        if self.response_task is not None:
            await asyncio.gather(self.response_task, return_exceptions=True)
        await self.tts.flush()
        while not self.tts_queue.empty():
            await asyncio.sleep(0.005)

    async def _handle_event(self, event: ASREvent) -> None:
        if event.type == "speech_start":
            self.barge_in()
            return
        if event.type != "final" or not event.text.strip():
            return

        self.barge_in()
        # Tell the client what was heard, so it can show the conversation.
        self.responses.put_nowait(pb2.AudioResponse(transcript=event.text.strip(), timestamp_ms=_now_ms()))
        self.bench.reset()
        if event.speech_end_ts is not None:
            self.bench.current.speech_end_ts = event.speech_end_ts
        self.bench.mark("final_text_ts")

        if self.settings.enable_ack_tone:
            ack_pcm = make_ack_tone(self.tts.sample_rate, self.settings.ack_tone_ms)
            if ack_pcm:
                self.bench.mark("first_audio_ts")
                self._send(ack_pcm, self.tts.sample_rate, "[ack]")

        self.response_task = asyncio.create_task(self._respond(event.text.strip()))

    # ----- interruption ---------------------------------------------

    def barge_in(self) -> None:
        was_active = self.cancel_response()
        if was_active or self.sent_audio:
            self.responses.put_nowait(pb2.AudioResponse(interrupt=True, timestamp_ms=_now_ms()))
        self.sent_audio = False

    def cancel_response(self) -> bool:
        active = self.response_task is not None and not self.response_task.done()
        if active:
            self.response_task.cancel()  # type: ignore[union-attr]
        self.response_task = None
        while not self.token_queue.empty():
            self.token_queue.get_nowait()
        self.tts.cancel_pending()
        self.tts_queue.clear()
        # Drop audio that was queued for the client but not yet sent.
        kept = []
        while not self.responses.empty():
            item = self.responses.get_nowait()
            if item is None or item.interrupt or item.transcript:
                kept.append(item)
        for item in kept:
            self.responses.put_nowait(item)
        return active

    # ----- response generation --------------------------------------

    async def _respond(self, text: str) -> None:
        self.bench.mark("prompt_sent_ts")
        self.history.append({"role": "user", "content": text})
        reply = ""
        try:
            reply = await asyncio.to_thread(self._action_reply, text)
            if reply:
                await self._speak(reply)
            else:
                generation = asyncio.create_task(
                    self.service.llm.stream_tokens(self.history, self.token_queue, bench=self.bench)
                )
                try:
                    await self._speak_stream(generation)
                    reply = generation.result()
                finally:
                    if not generation.done():
                        generation.cancel()
                        await asyncio.gather(generation, return_exceptions=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Assistant response failed")
            await self._speak(FALLBACK_REPLY)
        if reply:
            self.history.append({"role": "assistant", "content": reply})
            self._prune_history()
        await self.tts.flush()
        metrics = {k: v for k, v in self.bench.snapshot().items() if v is not None}
        logger.info("Turn metrics: %s", metrics)

    def _action_reply(self, text: str) -> str:
        intent = self.service.nlu.classify(text)
        result = self.service.actions.handle(text, intent)
        return result.response.strip() if result.handled else ""

    async def _speak(self, text: str) -> None:
        for sentence in sentence_chunks_from_tokens([text], max_tokens=self.settings.sentence_max_tokens):
            await self.tts.synthesize_sentence(sentence)

    async def _speak_stream(self, generation: asyncio.Task[str]) -> None:
        tokens: list[str] = []

        async def next_token() -> str:
            get = asyncio.ensure_future(self.token_queue.get())
            await asyncio.wait({get, generation}, return_when=asyncio.FIRST_COMPLETED)
            if get.done():
                return get.result()
            get.cancel()
            # Generation finished (or failed); drain what it already queued.
            if not self.token_queue.empty():
                return self.token_queue.get_nowait()
            generation.result()  # re-raise a generation failure
            return EOS

        while True:
            tok = await next_token()
            if tok == EOS:
                remaining = "".join(tokens).strip()
                if remaining:
                    await self.tts.synthesize_sentence(remaining)
                return

            tokens.append(tok)
            ready = sentence_chunks_from_tokens(tokens, max_tokens=self.settings.sentence_max_tokens)
            if ready and (len(ready) > 1 or ready[-1].endswith((".", "!", "?"))):
                for sentence in ready[:-1]:
                    await self.tts.synthesize_sentence(sentence)
                if ready[-1].endswith((".", "!", "?")):
                    await self.tts.synthesize_sentence(ready[-1])
                    tokens = []
                else:
                    tokens = [ready[-1]]

    def _prune_history(self) -> None:
        max_messages = self.settings.conversation_history_turns * 2
        system = [m for m in self.history if m.get("role") == "system"][:1]
        chat = [m for m in self.history if m.get("role") != "system"][-max_messages:]
        self.history[:] = system + chat

    # ----- output ----------------------------------------------------

    async def _pump_audio(self) -> None:
        while True:
            chunk = await self.tts_queue.get()
            if self.bench.current.first_audio_ts is None:
                self.bench.mark("first_audio_ts")
            self._send(chunk.pcm16, chunk.sample_rate, chunk.debug_text)

    def _send(self, pcm16: bytes, sample_rate: int, debug_text: str) -> None:
        self.sent_audio = True
        self.responses.put_nowait(
            pb2.AudioResponse(
                pcm16=pcm16,
                sample_rate=sample_rate,
                timestamp_ms=_now_ms(),
                debug_text=debug_text,
            )
        )


def _now_ms() -> int:
    return int(time.time() * 1000)


# =====================================================================
# Main VoiceAssistant gRPC Service
# =====================================================================

class VoiceAssistantService(pb2_grpc.VoiceAssistantServicer):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.nlu = SimpleIntentClassifier()
        self.actions = settings.build_actions()
        self.vosk_model: Any = None

        # Check if high-fidelity mock mode is enabled (for regression testing/CI)
        if mock_models_enabled():
            logger.info("Starting gRPC service in HIGH-FIDELITY MOCK MODE (MOCK_MODELS=1)")
            self.llm: Any = MockLLMClient()
        else:
            if not _VOSK_AVAILABLE:
                raise RuntimeError("vosk is not installed. Install it with `pip install 'voice-assistant[local]'`.")
            # One model shared by every stream; each stream gets its own recognizer.
            self.vosk_model = Model(settings.asr_model_path)
            self.llm = StreamingLLMClient(
                LLMConfig(
                    model_path=settings.model_path,
                    n_ctx=settings.llm_context_size,
                    n_gpu_layers=settings.n_gpu_layers,
                    max_tokens=settings.llm_max_tokens,
                    temperature=settings.llm_temperature,
                ),
            )

    async def StreamVoice(
        self, request_iterator: AsyncIterator[pb2.AudioChunk], context: grpc.aio.ServicerContext
    ) -> AsyncIterator[pb2.AudioResponse]:
        session = _VoiceSession(self)
        await session.start()

        async def read_requests() -> None:
            try:
                async for req in request_iterator:
                    if req.pcm16:
                        await session.feed(req.pcm16)
                await session.end_of_input()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in request reader")
            finally:
                session.responses.put_nowait(None)

        read_task = asyncio.create_task(read_requests())

        try:
            while True:
                response = await session.responses.get()
                if response is None:
                    break
                yield response
        finally:
            read_task.cancel()
            await asyncio.gather(read_task, return_exceptions=True)
            await session.close()


async def serve(host: str, port: int, settings: Settings) -> None:
    server = grpc.aio.server()
    service = VoiceAssistantService(settings)
    pb2_grpc.add_VoiceAssistantServicer_to_server(service, server)
    await warm_up_llm(service.llm, settings.assistant_system_prompt)
    bound = server.add_insecure_port(f"{host}:{port}")
    if bound == 0:
        raise RuntimeError(f"Could not bind gRPC server to {host}:{port}")
    await server.start()
    logger.info("gRPC server listening on %s:%s", host, port)

    # Stop gracefully on SIGTERM (docker stop, systemd, Kubernetes): in-flight
    # streams get a grace period instead of being cut off.
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGTERM, stopping.set)
    except (NotImplementedError, RuntimeError):  # Windows / non-main thread
        pass
    # Cancelling wait_for_termination() would cancel gRPC's shutdown future,
    # so it is left to finish on its own once the server has stopped.
    waiter = asyncio.create_task(server.wait_for_termination())
    stopper = asyncio.create_task(stopping.wait())
    try:
        await asyncio.wait({waiter, stopper}, return_when=asyncio.FIRST_COMPLETED)
        if stopping.is_set():
            logger.info("SIGTERM received; shutting down gRPC server")
    finally:
        stopper.cancel()
        await server.stop(grace=5)
        await asyncio.gather(waiter, stopper, return_exceptions=True)
        logger.info("gRPC server stopped")
