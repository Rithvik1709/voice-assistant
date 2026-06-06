from __future__ import annotations

import asyncio
import json
import logging
import time
import os
from collections.abc import AsyncIterator
from typing import Any

import grpc
from concurrent.futures import ThreadPoolExecutor

from voice_assistant.asr.partial import PartialTranscriptStabilizer
from voice_assistant.asr.vad import VADConfig, VoiceActivityDetector
from voice_assistant.benchmark import BenchmarkTracker
from voice_assistant.config import Settings
from voice_assistant.llm.client import LLMConfig, StreamingLLMClient
from voice_assistant.tts.queue import AudioChunk, AudioChunkQueue
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
    from vosk import KaldiRecognizer, Model  # type: ignore
    _VOSK_AVAILABLE = True
except ImportError:
    _VOSK_AVAILABLE = False
    class Model:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
    class KaldiRecognizer:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


# =====================================================================
# High-Fidelity Mock Implementations for Performance Testing & CI Jobs
# =====================================================================

class MockKaldiRecognizer:
    def __init__(self) -> None:
        pass

    def AcceptWaveform(self, frame: bytes) -> bool:
        # Simulate short CPU decoding time (5ms)
        time.sleep(0.005)
        return True

    def PartialResult(self) -> str:
        return '{"partial": "hello"}'

    def FinalResult(self) -> str:
        return '{"text": "hello world"}'

    def Reset(self) -> None:
        pass


class MockLLMClient:
    async def stream_tokens(self, prompt: str, out_queue: asyncio.Queue[str]) -> str:
        tokens = ["Hello", " this", " is", " a", " mock", " response", " from", " the", " assistant", "."]
        for tok in tokens:
            try:
                await asyncio.wait_for(out_queue.put(tok), timeout=10.0)
            except asyncio.TimeoutError as exc:
                logger.error("LLM output queue backpressure; aborting generation task")
                raise RuntimeError("LLM output queue full; generation aborted") from exc
            await asyncio.sleep(0.005)
        return "Hello this is a mock response from the assistant."


class MockPiperStreamingTTS:
    def __init__(self, config: PiperConfig | None, playback_queue: AudioChunkQueue, bench: BenchmarkTracker | None = None) -> None:
        self.playback_queue = playback_queue
        self.bench = bench

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def synthesize_sentence(self, sentence: str) -> bool:
        # Generate dummy 1 second 22050Hz 16-bit mono silent chunk
        pcm16 = b"\x00\x00" * 22050
        chunk = AudioChunk(pcm16=pcm16, sample_rate=22050, debug_text=sentence)
        try:
            await self.playback_queue.put(chunk)
            return True
        except Exception:
            return False


# =====================================================================
# Main VoiceAssistant gRPC Service
# =====================================================================

class VoiceAssistantService(pb2_grpc.VoiceAssistantServicer):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.bench = BenchmarkTracker()

        # Check if high-fidelity mock mode is enabled (for regression testing/CI)
        if os.getenv("MOCK_MODELS") == "1":
            logger.info("Starting gRPC service in HIGH-FIDELITY MOCK MODE (MOCK_MODELS=1)")
            self._vosk_model = None
            self.llm = MockLLMClient()
        else:
            if not _VOSK_AVAILABLE:
                raise RuntimeError("vosk is not installed")
            self._vosk_model = Model(settings.asr_model_path)
            self.llm = StreamingLLMClient(
                LLMConfig(
                    model_path=settings.model_path,
                    n_ctx=settings.llm_context_size,
                    n_gpu_layers=settings.n_gpu_layers,
                    max_tokens=settings.llm_max_tokens,
                    temperature=settings.llm_temperature,
                ),
                bench=self.bench,
            )

    async def StreamVoice(
        self, request_iterator: AsyncIterator[pb2.AudioChunk], context: grpc.aio.ServicerContext
    ) -> AsyncIterator[pb2.AudioResponse]:
        bench = BenchmarkTracker()
        vad = self._build_vad()
        recognizer = KaldiRecognizer(self._vosk_model, self.settings.sample_rate)
        partial_stabilizer = PartialTranscriptStabilizer()
        tts_queue = AudioChunkQueue(maxsize=self.settings.tts_queue_maxsize)
        tts = PiperStreamingTTS(PiperConfig(self.settings.piper_voice_path), tts_queue, bench=bench)

        speech_buffer = bytearray()
        
        # Mock mode uses energy VAD (no binary dependency). Production uses webrtc VAD (more accurate).
        vad_mode = "energy" if os.getenv("MOCK_MODELS") == "1" else "webrtc"
        stream_vad = VoiceActivityDetector(
            VADConfig(
                sample_rate=self.settings.sample_rate,
                frame_ms=self.settings.chunk_ms,
                aggressiveness=self.settings.vad_aggressiveness,
                mode=vad_mode,
            )
        )

        # Instantiate per-stream TTS and queue to isolate concurrent requests and enable parallel synthesis
        if os.getenv("MOCK_MODELS") == "1":
            tts_queue = AudioChunkQueue(maxsize=self.settings.tts_queue_maxsize)
            tts = MockPiperStreamingTTS(None, tts_queue, bench=self.bench)
            recognizer = MockKaldiRecognizer()
        else:
            if not _VOSK_AVAILABLE:
                raise RuntimeError("vosk is not installed")
            recognizer = KaldiRecognizer(self._vosk_model, self.settings.sample_rate)
            tts_queue = AudioChunkQueue(maxsize=self.settings.tts_queue_maxsize)
            tts = PiperStreamingTTS(
                PiperConfig(self.settings.piper_voice_path, self.settings.tts_sample_rate),
                playback_queue=tts_queue,
                bench=self.bench
            )

        # Start TTS process/workers for this stream
        await tts.start()
        
        token_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=256)
        response_queue: asyncio.Queue[pb2.AudioResponse | None] = asyncio.Queue()

        active_llm_task: asyncio.Task | None = None
        active_response_task: asyncio.Task | None = None
        active_audio_stream_task: asyncio.Task | None = None

        async def cancel_active_response(await_cleanup: bool = False):
            nonlocal active_llm_task, active_response_task, active_audio_stream_task
            if active_llm_task and not active_llm_task.done():
                active_llm_task.cancel()
                if await_cleanup:
                    try:
                        await active_llm_task
                    except asyncio.CancelledError:
                        pass
                active_llm_task = None

            if active_response_task and not active_response_task.done():
                active_response_task.cancel()
                if await_cleanup:
                    try:
                        await active_response_task
                    except asyncio.CancelledError:
                        pass
                active_response_task = None

            if active_audio_stream_task and not active_audio_stream_task.done():
                active_audio_stream_task.cancel()
                if await_cleanup:
                    try:
                        await active_audio_stream_task
                    except asyncio.CancelledError:
                        pass
                active_audio_stream_task = None

            # Drain token_queue
            while not token_queue.empty():
                try:
                    token_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # Clear tts_queue
            tts_queue.clear()

            # Drain tts ingest_queue
            if hasattr(tts, "ingest_queue"):
                while not tts.ingest_queue.empty():
                    try:
                        tts.ingest_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            # Drain response_queue
            while not response_queue.empty():
                try:
                    response_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        async def read_requests():
            nonlocal active_llm_task, active_response_task, active_audio_stream_task
            try:
                async for req in request_iterator:
                    frame = req.pcm16
                    if not frame:
                        continue

                    speech = stream_vad.is_speech(frame[: stream_vad.frame_bytes]) if len(frame) >= stream_vad.frame_bytes else False
                    if speech:
                        if not speech_buffer:
                            # New speech starting! Trigger interruption if assistant is active
                            await cancel_active_response(await_cleanup=False)
                        speech_buffer.extend(frame)
                        continue

                    if speech_buffer:
                        # Offload single-pass ASR on complete buffer to thread pool exactly once at end-of-speech
                        await asyncio.to_thread(recognizer.AcceptWaveform, bytes(speech_buffer))
                        text = self._extract_final(recognizer)
                        speech_buffer.clear()

                        if not text.strip():
                            continue

                        self.bench.mark("prompt_sent_ts")

                        # Cancel any active response first (though we likely already did when speech started)
                        await cancel_active_response(await_cleanup=False)

                        # Run LLM streaming in a background task and append "<eos>" at the end
                        async def run_llm():
                            try:
                                await self.llm.stream_tokens(text, token_queue)
                            except asyncio.CancelledError:
                                raise
                            except Exception as e:
                                logger.error("Error in LLM stream_tokens: %s", e)
                            finally:
                                await token_queue.put("<eos>")

                        active_llm_task = asyncio.create_task(run_llm())

                        # Start response processor task
                        async def process_response():
                            tokens: list[str] = []
                            try:
                                while True:
                                    tok = await token_queue.get()
                                    if tok == "<eos>":
                                        # Flush remaining tokens
                                        remaining_sentence = "".join(tokens).strip()
                                        if remaining_sentence:
                                            await tts.synthesize_sentence(remaining_sentence)
                                        break

                                    tokens.append(tok)
                                    ready = sentence_chunks_from_tokens(tokens, max_tokens=self.settings.sentence_max_tokens)
                                    if ready and (len(ready) > 1 or ready[-1].endswith(('.', '!', '?'))):
                                        for sentence in ready[:-1]:
                                            await tts.synthesize_sentence(sentence)

                                        # If the last chunk is also a complete sentence, synthesize it immediately too
                                        if ready[-1].endswith(('.', '!', '?')):
                                            await tts.synthesize_sentence(ready[-1])
                                            tokens = []
                                        else:
                                            tokens = [ready[-1]]

                                logger.info("metrics=%s", self.bench.snapshot())
                                self.bench.reset()
                            except asyncio.CancelledError:
                                logger.info("Response processor task was cancelled")
                                raise
                            except Exception as e:
                                logger.error("Error in response processor: %s", e, exc_info=True)

                        active_response_task = asyncio.create_task(process_response())

                        # Start independent audio streaming task to drain tts_queue
                        async def stream_audio_responses():
                            try:
                                while True:
                                    chunk = await tts_queue.get()
                                    await response_queue.put(pb2.AudioResponse(
                                        pcm16=chunk.pcm16,
                                        sample_rate=chunk.sample_rate,
                                        timestamp_ms=int(time.time() * 1000),
                                        debug_text=chunk.debug_text,
                                    ))
                            except asyncio.CancelledError:
                                pass
                            except Exception as e:
                                logger.error("Error in stream_audio_responses: %s", e)

                        active_audio_stream_task = asyncio.create_task(stream_audio_responses())

            except asyncio.CancelledError:
                logger.info("Request reader task was cancelled")
                raise
            except Exception as e:
                logger.error("Error in request reader: %s", e, exc_info=True)
            finally:
                # Wait for any active response task to complete before ending the stream
                if active_response_task and not active_response_task.done():
                    try:
                        await active_response_task
                    except Exception as e:
                        logger.error("Error waiting for active response task: %s", e)
                if active_audio_stream_task and not active_audio_stream_task.done():
                    try:
                        await active_audio_stream_task
                    except Exception as e:
                        logger.error("Error waiting for active audio stream task: %s", e)
                # Signal the response queue to stop yielding
                await response_queue.put(None)

        read_task = asyncio.create_task(read_requests())

        try:
            while True:
                response = await response_queue.get()
                if response is None:
                    break
                yield response
        finally:
            read_task.cancel()
            await cancel_active_response(await_cleanup=True)
            try:
                await read_task
            except asyncio.CancelledError:
                pass
            # Ensure TTS subprocess and queue resources are cleanly torn down
            await tts.stop()

    def _extract_partial(self, recognizer) -> str:
        import json

        raw = recognizer.PartialResult()
        try:
            data = json.loads(raw)
            return data.get("partial", "")
        except Exception:
            return ""

    def _extract_final(self, recognizer) -> str:
        import json
        raw = recognizer.FinalResult()
        try:
            data = json.loads(raw)
            return data.get("text", "")
        except Exception:
            return ""

async def serve(host: str, port: int, settings: Settings) -> None:
    server = grpc.aio.server()
    pb2_grpc.add_VoiceAssistantServicer_to_server(VoiceAssistantService(settings), server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    logger.info("gRPC server listening on %s:%s", host, port)
    await server.wait_for_termination()
