from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

import grpc
from vosk import KaldiRecognizer, Model

from voice_assistant.asr.partial import PartialTranscriptStabilizer
from voice_assistant.asr.vad import VADConfig, VoiceActivityDetector
from voice_assistant.benchmark import BenchmarkTracker
from voice_assistant.config import Settings
from voice_assistant.llm.client import LLMConfig, StreamingLLMClient
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


class VoiceAssistantService(pb2_grpc.VoiceAssistantServicer):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._vosk_model = Model(settings.asr_model_path)

        self.llm = StreamingLLMClient(
            LLMConfig(
                model_path=settings.model_path,
                n_ctx=settings.llm_context_size,
                n_gpu_layers=settings.n_gpu_layers,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )
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

        async for req in request_iterator:
            frame = req.pcm16
            if not frame:
                continue

            speech = vad.is_speech(frame[: vad.frame_bytes]) if len(frame) >= vad.frame_bytes else False
            if speech:
                speech_buffer.extend(frame)
                recognizer.AcceptWaveform(frame)
                partial = partial_stabilizer.update(self._extract_partial(recognizer))
                if partial:
                    logger.debug("partial=%s", partial)
                continue

            if speech_buffer:
                recognizer.AcceptWaveform(bytes(speech_buffer))
                text = self._extract_final(recognizer)
                speech_buffer.clear()
                if not text.strip():
                    vad.reset()
                    partial_stabilizer = PartialTranscriptStabilizer()
                    tts_queue.clear()
                    continue

                bench.mark("prompt_sent_ts")
                token_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=256)
                _ = asyncio.create_task(self.llm.stream_tokens(text, token_queue, bench=bench))

                tokens: list[str] = []
                first_audio_sent = False
                while True:
                    tok = await token_queue.get()
                    tokens.append(tok)
                    for sentence in sentence_chunks_from_tokens(tokens, max_tokens=self.settings.sentence_max_tokens):
                        await tts.synthesize_sentence(sentence)
                        chunk = await tts_queue.get()
                        if not first_audio_sent:
                            first_audio_sent = True
                        yield pb2.AudioResponse(
                            pcm16=chunk.pcm16,
                            sample_rate=chunk.sample_rate,
                            timestamp_ms=int(time.time() * 1000),
                            debug_text=sentence,
                        )
                        tokens = []
                    if tok.endswith("\n") or tok.endswith(".") or tok.endswith("?") or tok.endswith("!"):
                        break

                logger.info("metrics=%s", bench.snapshot())
                bench.reset()
                vad.reset()
                partial_stabilizer = PartialTranscriptStabilizer()
                tts_queue.clear()

    def _build_vad(self) -> VoiceActivityDetector:
        return VoiceActivityDetector(
            VADConfig(
                sample_rate=self.settings.sample_rate,
                frame_ms=self.settings.chunk_ms,
                aggressiveness=self.settings.vad_aggressiveness,
                mode="webrtc",
            )
        )

    @staticmethod
    def _extract_partial(recognizer: KaldiRecognizer) -> str:
        data = json.loads(recognizer.PartialResult())
        return data.get("partial", "")

    @staticmethod
    def _extract_final(recognizer: KaldiRecognizer) -> str:
        data = json.loads(recognizer.FinalResult())
        return data.get("text", "")


async def serve(host: str, port: int, settings: Settings) -> None:
    server = grpc.aio.server()
    pb2_grpc.add_VoiceAssistantServicer_to_server(VoiceAssistantService(settings), server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    logger.info("gRPC server listening on %s:%s", host, port)
    await server.wait_for_termination()
