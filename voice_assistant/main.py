from __future__ import annotations

import argparse
import asyncio
import logging

from voice_assistant.config import Settings
from voice_assistant.pipeline.orchestrator import VoicePipelineOrchestrator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time streaming voice assistant")
    p.add_argument("--mode", choices=["local", "server", "client"], default="local")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=50051)
    p.add_argument("--target", default="localhost:50051")
    return p.parse_args()


async def run_local(settings: Settings) -> None:
    # Local-only imports to avoid requiring sounddevice when running in server mode
    from voice_assistant.asr.stream import StreamingASR
    from voice_assistant.asr.vad import VADConfig, VoiceActivityDetector
    from voice_assistant.benchmark import BenchmarkTracker
    from voice_assistant.llm.client import LLMConfig, StreamingLLMClient
    from voice_assistant.tts.player import AudioPlayer
    from voice_assistant.tts.queue import AudioChunkQueue
    from voice_assistant.tts.stream import PiperConfig, PiperStreamingTTS
    from voice_assistant.nlu import SimpleIntentClassifier

    bench = BenchmarkTracker()
    vad = VoiceActivityDetector(
        VADConfig(
            sample_rate=settings.sample_rate,
            frame_ms=settings.chunk_ms,
            aggressiveness=settings.vad_aggressiveness,
            mode="webrtc",
        )
    )

    asr = StreamingASR(
        sample_rate=settings.sample_rate,
        chunk_size=settings.chunk_size,
        vad=vad,
        model_path=settings.asr_model_path,
        backend=settings.asr_backend,
        endpoint_silence_ms=settings.asr_endpoint_silence_ms,
    )

    llm = StreamingLLMClient(
        LLMConfig(
            model_path=settings.model_path,
            n_ctx=settings.llm_context_size,
            n_gpu_layers=settings.n_gpu_layers,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        ),
        bench=bench,
    )

    queue = AudioChunkQueue(maxsize=settings.tts_queue_maxsize)
    tts = PiperStreamingTTS(PiperConfig(settings.piper_voice_path, settings.tts_sample_rate), queue=queue, bench=bench)
    player = AudioPlayer(sample_rate=settings.tts_sample_rate, blocksize=settings.player_blocksize)

    orchestrator = VoicePipelineOrchestrator(
        asr=asr,
        llm=llm,
        tts=tts,
        player=player,
        nlu=SimpleIntentClassifier(),
        bench=bench,
        tts_sentence_max_tokens=settings.sentence_max_tokens,
        tts_eager_min_words=settings.tts_eager_min_words,
    )
    await orchestrator.run()


async def amain() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    from voice_assistant.telemetry import init_telemetry
    init_telemetry()
    args = parse_args()
    settings = Settings()

    if args.mode in {"local", "server"}:
        settings.validate()

    if args.mode == "local":
        await run_local(settings)
        return

    if args.mode == "server":
        from voice_assistant.transport.grpc_server import serve

        await serve(args.host, args.port or settings.grpc_port, settings)
        return

    if args.mode == "client":
        from voice_assistant.transport.grpc_client import GRPCVoiceClient

        client = GRPCVoiceClient(target=args.target, sample_rate=settings.sample_rate, chunk_size=settings.chunk_size)
        await client.run()


if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n[Voice Assistant] Shutting down...")
