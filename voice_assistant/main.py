from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from voice_assistant import __version__
from voice_assistant.config import ConfigError, Settings

MODES = ["local", "chat", "web", "server", "client", "doctor", "models"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vaani", description="Vaani: real-time streaming voice assistant")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--mode", choices=MODES, default=None, help="What to run (default: local)")
    p.add_argument("mode_arg", nargs="?", choices=MODES, metavar="MODE", help="Same as --mode: " + ", ".join(MODES))
    p.add_argument("--host", default=None, help="bind address (default: 0.0.0.0 for server, 127.0.0.1 for web)")
    p.add_argument("--port", type=int, default=None, help="port (default: GRPC_PORT or 50051 for server, 8080 for web)")
    p.add_argument("--target", default="localhost:50051", help="gRPC server address for client mode")
    p.add_argument("--speak", action="store_true", help="chat: also speak replies through Piper")
    p.add_argument("--skip-audio-check", action="store_true", help="doctor: skip the microphone check")
    p.add_argument("--models-dir", default="models", help="models: download directory")
    p.add_argument("--download", action="store_true", help="models: download the starter model set")
    p.add_argument("--write-env", action="store_true", help="models: write a .env pointing at the models")
    p.add_argument("--env-file", default=".env", help="models: path of the .env file to write")
    p.add_argument(
        "--languages",
        default="",
        help="models: languages to set up besides English, e.g. hi,ta,ja (a voice each, multilingual Whisper)",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)
    if args.mode and args.mode_arg and args.mode != args.mode_arg:
        p.error(f"conflicting modes: --mode {args.mode} and {args.mode_arg}")
    from voice_assistant.lang import LANGUAGE_NAMES, parse_language_list

    args.languages = parse_language_list(args.languages)
    unknown = [code for code in args.languages if code not in LANGUAGE_NAMES]
    if unknown:
        p.error(f"unknown language codes: {', '.join(unknown)} (use codes such as hi, ta, fr, ja)")
    args.mode = args.mode or args.mode_arg or "local"
    return args


async def run_local(settings: Settings) -> None:
    # Local-only imports to avoid requiring sounddevice when running in server mode
    from voice_assistant.asr.stream import StreamingASR
    from voice_assistant.asr.vad import VADConfig, VoiceActivityDetector
    from voice_assistant.audio.aec import create_echo_canceller
    from voice_assistant.benchmark import BenchmarkTracker
    from voice_assistant.llm import create_llm
    from voice_assistant.llm.client import warm_up_llm
    from voice_assistant.memory import SessionMemory
    from voice_assistant.nlu import SimpleIntentClassifier
    from voice_assistant.pipeline.orchestrator import VoicePipelineOrchestrator
    from voice_assistant.tts.player import AudioPlayer
    from voice_assistant.tts.queue import AudioChunkQueue
    from voice_assistant.tts.stream import PiperStreamingTTS

    bench = BenchmarkTracker()
    aec = create_echo_canceller(
        settings.echo_cancellation, settings.sample_rate, noise_suppression=settings.aec_noise_suppression
    )
    if aec is None and settings.enable_barge_in:
        logging.getLogger(__name__).info(
            "Without echo cancellation Vaani can hear itself on speakers; use headphones or ENABLE_BARGE_IN=0"
        )
    wake_gate = settings.build_wake_gate()
    wake_detector = settings.build_wake_detector()
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
        backend=settings.asr_backend,
        endpoint_silence_ms=settings.asr_endpoint_silence_ms,
        speech_start_frames=settings.barge_in_frames,
        hold_silence_ms=settings.asr_hold_silence_ms,
        early_decode_ms=settings.asr_early_decode_ms,
        partial_interval_ms=settings.asr_partial_interval_ms,
        language=settings.asr_language,
        allowed_languages=settings.asr_languages,
        model_path=settings.whisper_model() if settings.asr_backend == "whisper" else settings.asr_model_path,
        echo_canceller=aec,
        wake_detector=wake_detector,
        wake_gate=wake_gate,
        hotwords=wake_gate.hotwords() if wake_gate is not None else "",
    )

    llm = create_llm(settings, bench=bench)

    queue = AudioChunkQueue(maxsize=settings.tts_queue_maxsize)
    tts = PiperStreamingTTS(
        settings.piper_config(),
        playback_queue=queue,
        bench=bench,
    )
    player = AudioPlayer(
        sample_rate=tts.sample_rate,
        blocksize=settings.player_blocksize,
        on_output=aec.push_playback if aec is not None else None,
    )
    memory = (
        SessionMemory(Path(settings.conversation_memory_path).expanduser())  # noqa: ASYNC240
        if settings.conversation_memory_path
        else None
    )

    orchestrator = VoicePipelineOrchestrator(
        asr=asr,
        llm=llm,
        tts=tts,
        player=player,
        nlu=SimpleIntentClassifier(),
        action_handler=settings.build_actions(),
        memory=memory,
        system_prompt=settings.assistant_system_prompt,
        bench=bench,
        tts_sentence_max_tokens=settings.sentence_max_tokens,
        tts_eager_min_words=settings.tts_eager_min_words,
        ack_tone_ms=settings.ack_tone_ms if settings.enable_ack_tone else 0,
        max_conversation_turns=settings.conversation_history_turns,
        barge_in=settings.enable_barge_in,
        reply_in_user_language=settings.multilingual,
        wake_gate=wake_gate,
        facts=settings.build_facts(),
    )
    if wake_gate is not None:
        names = [p.title() for p in settings.wake_phrases] or [wake_detector.name.replace("_", " ").title()]
        logging.getLogger(__name__).info("Say \"%s\" to talk to Vaani", names[0])
    await warm_up_llm(llm, settings.assistant_system_prompt)
    await orchestrator.run()


async def amain(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    from voice_assistant.telemetry import init_telemetry
    init_telemetry()
    settings = Settings()

    if args.mode == "doctor":
        from voice_assistant.doctor import format_doctor_report, has_failures, run_doctor

        checks = run_doctor(settings, check_audio=not args.skip_audio_check)
        print(format_doctor_report(checks))
        if has_failures(checks):
            raise SystemExit(1)
        return

    if args.mode == "models":
        from voice_assistant.model_setup import (
            download_recommended_models,
            format_model_plan,
            write_env_file,
        )

        models_dir = Path(args.models_dir)
        print(format_model_plan(models_dir, args.languages))

        if args.write_env:
            wrote = write_env_file(Path(args.env_file), models_dir, languages=args.languages)
            status = "wrote" if wrote else "kept existing"
            print(f"{status}: {args.env_file}")

        if args.download:
            for message in download_recommended_models(models_dir, args.languages):
                print(message)
        return

    if args.mode in {"local", "server", "web"}:
        settings.validate()

    if args.mode == "chat":
        from voice_assistant.chat import run_chat

        settings.validate(need_asr=False, need_tts=args.speak)
        await run_chat(settings, speak=args.speak)
        return

    if args.mode == "local":
        await run_local(settings)
        return

    if args.mode == "server":
        from voice_assistant.transport.grpc_server import serve

        await serve(args.host or "0.0.0.0", args.port if args.port is not None else settings.grpc_port, settings)
        return

    if args.mode == "web":
        from voice_assistant.transport.webrtc import require_webrtc, serve_web

        require_webrtc()
        await serve_web(args.host or "127.0.0.1", args.port if args.port is not None else 8080, settings)
        return

    if args.mode == "client":
        from voice_assistant.transport.grpc_client import GRPCVoiceClient

        client = GRPCVoiceClient(target=args.target, sample_rate=settings.sample_rate, chunk_size=settings.chunk_size)
        await client.run()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\n[Vaani] Shutting down...")
    except ConfigError as exc:
        # Configuration problems: show the message, not a traceback.
        print(f"[Vaani] {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except RuntimeError as exc:
        # Missing optional packages ("pip install ...") are reported plainly.
        if "pip install" not in str(exc):
            raise
        print(f"[Vaani] {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
