from __future__ import annotations

import asyncio
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
                await asyncio.wait_for(out_queue.put(tok), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("LLM output queue backpressure; dropping token")
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
        chunk = AudioChunk(pcm16=pcm16, sample_rate=22050)
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

        try:
            async for req in request_iterator:
                frame = req.pcm16
                if not frame:
                    continue

                speech = stream_vad.is_speech(frame[: stream_vad.frame_bytes]) if len(frame) >= stream_vad.frame_bytes else False
                if speech:
                    speech_buffer.extend(frame)
                    # Skip incremental ASR to avoid event loop queuing delays under concurrency
                    continue

                if speech_buffer:
                    # Offload single-pass ASR on complete buffer to thread pool exactly once at end-of-speech
                    await asyncio.to_thread(recognizer.AcceptWaveform, bytes(speech_buffer))
                    text = self._extract_final(recognizer)
                    speech_buffer.clear()
                    
                    # Reset Vosk recognizer state for the next turn
                    recognizer.Reset()

                    if not text.strip():
                        continue

                    self.bench.mark("prompt_sent_ts")

                    # Run LLM streaming in a background task and append "<eos>" at the end
                    async def run_llm():
                        try:
                            await self.llm.stream_tokens(text, token_queue)
                        finally:
                            await token_queue.put("<eos>")
                    
                    _ = asyncio.create_task(run_llm())

                    tokens: list[str] = []
                    first_audio_sent = False
                    while True:
                        tok = await token_queue.get()
                        if tok == "<eos>":
                            # Flush remaining tokens
                            remaining_sentence = "".join(tokens).strip()
                            if remaining_sentence:
                                try:
                                    await asyncio.wait_for(tts.synthesize_sentence(remaining_sentence), timeout=30.0)
                                    chunk = await asyncio.wait_for(tts_queue.get(), timeout=30.0)
                                    if not first_audio_sent:
                                        first_audio_sent = True
                                    yield pb2.AudioResponse(
                                        pcm16=chunk.pcm16,
                                        sample_rate=chunk.sample_rate,
                                        timestamp_ms=int(time.time() * 1000),
                                        debug_text=remaining_sentence,
                                    )
                                except asyncio.TimeoutError:
                                    logger.warning("TTS synthesis/queue timeout during flush for sentence: %s", remaining_sentence)
                            break

                        tokens.append(tok)
                        ready = sentence_chunks_from_tokens(tokens, max_tokens=self.settings.sentence_max_tokens)
                        if ready and (len(ready) > 1 or ready[-1].endswith(('.', '!', '?'))):
                            for sentence in ready[:-1]:
                                try:
                                    await asyncio.wait_for(tts.synthesize_sentence(sentence), timeout=30.0)
                                    chunk = await asyncio.wait_for(tts_queue.get(), timeout=30.0)
                                    if not first_audio_sent:
                                        first_audio_sent = True
                                    yield pb2.AudioResponse(
                                        pcm16=chunk.pcm16,
                                        sample_rate=chunk.sample_rate,
                                        timestamp_ms=int(time.time() * 1000),
                                        debug_text=sentence,
                                    )
                                except asyncio.TimeoutError:
                                    logger.warning("TTS synthesis/queue timeout for sentence: %s", sentence)
                            
                            # If the last chunk is also a complete sentence, synthesize it immediately too
                            if ready[-1].endswith(('.', '!', '?')):
                                try:
                                    await asyncio.wait_for(tts.synthesize_sentence(ready[-1]), timeout=30.0)
                                    chunk = await asyncio.wait_for(tts_queue.get(), timeout=30.0)
                                    if not first_audio_sent:
                                        first_audio_sent = True
                                    yield pb2.AudioResponse(
                                        pcm16=chunk.pcm16,
                                        sample_rate=chunk.sample_rate,
                                        timestamp_ms=int(time.time() * 1000),
                                        debug_text=ready[-1],
                                    )
                                except asyncio.TimeoutError:
                                    logger.warning("TTS synthesis/queue timeout for sentence: %s", ready[-1])
                                tokens = []
                            else:
                                tokens = [ready[-1]]

                    logger.info("metrics=%s", self.bench.snapshot())
                    self.bench.reset()
        finally:
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
