"""Text chat with the assistant: type instead of speaking.

Runs the same orchestrator as voice mode (intent actions, LLM, memory), with
typed lines standing in for the microphone. Replies are printed as they
stream; with `speak=True` they are also spoken through Piper.
"""
from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from voice_assistant.asr.stream import ASREvent
from voice_assistant.benchmark import BenchmarkTracker
from voice_assistant.config import Settings, mock_models_enabled
from voice_assistant.tts.queue import AudioChunkQueue

HELP = "Type a message and press Enter. Commands: /reset forgets the conversation, /quit exits."


class TextInput:
    """ASR stand-in that yields each typed line as a final transcript."""

    def __init__(self, read_line: Callable[[str], str] = input, prompt: str = "You: ") -> None:
        self.read_line = read_line
        self.prompt = prompt
        self.muted = False
        self.ready = asyncio.Event()
        self.ready.set()
        self.on_command: Callable[[str], None] | None = None

    async def stream_events(self) -> AsyncIterator[ASREvent]:
        while True:
            # Wait until the previous reply has finished before prompting.
            await self.ready.wait()
            try:
                line = await asyncio.to_thread(self.read_line, self.prompt)
            except EOFError:
                return
            text = line.strip()
            if not text:
                continue
            if text.lower() in {"/quit", "/exit", "/q"}:
                return
            if text.startswith("/"):
                if self.on_command is not None:
                    self.on_command(text.lower())
                continue
            self.ready.clear()
            yield ASREvent("final", text, 1.0, int(time.time() * 1000))


class ReplyPrinter:
    def __init__(self, out=sys.stdout) -> None:
        self.out = out
        self.started = False

    def token(self, token: str) -> None:
        if not self.started:
            self.out.write("Vaani: ")
            self.started = True
            token = token.lstrip()
        self.out.write(token)
        self.out.flush()

    def end(self) -> None:
        if self.started:
            self.out.write("\n")
            self.out.flush()
        self.started = False


class SilentTTS:
    """TTS stand-in for text-only chat."""

    sample_rate = 22_050

    def __init__(self) -> None:
        self.playback_queue = AudioChunkQueue(maxsize=8)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def synthesize_sentence(self, sentence: str) -> bool:
        return True

    def cancel_pending(self) -> None:
        pass

    async def flush(self) -> None:
        pass


class SilentPlayer:
    sample_rate = 22_050
    is_playing = False

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def play(self, chunk) -> None:
        pass

    def interrupt(self) -> None:
        pass


def build_chat(
    settings: Settings,
    llm,
    speak: bool = False,
    read_line: Callable[[str], str] = input,
    out=sys.stdout,
    tts=None,
    player=None,
):
    """Wire an orchestrator for text chat. Returns (orchestrator, text_input)."""
    from voice_assistant.memory import SessionMemory
    from voice_assistant.nlu import SimpleIntentClassifier
    from voice_assistant.pipeline.orchestrator import VoicePipelineOrchestrator

    text_input = TextInput(read_line=read_line)
    printer = ReplyPrinter(out)

    def turn_end() -> None:
        printer.end()
        text_input.ready.set()

    orchestrator = VoicePipelineOrchestrator(
        asr=text_input,  # type: ignore[arg-type]
        llm=llm,
        tts=tts or SilentTTS(),  # type: ignore[arg-type]
        player=player or SilentPlayer(),  # type: ignore[arg-type]
        bench=BenchmarkTracker(),
        nlu=SimpleIntentClassifier(),
        action_handler=settings.build_actions(),
        memory=(
            SessionMemory(Path(settings.conversation_memory_path).expanduser())
            if settings.conversation_memory_path
            else None
        ),
        system_prompt=settings.assistant_system_prompt,
        tts_sentence_max_tokens=settings.sentence_max_tokens,
        tts_eager_min_words=settings.tts_eager_min_words,
        ack_tone_ms=0,
        max_conversation_turns=settings.conversation_history_turns,
        # Typing never interrupts; replies finish before the next prompt.
        barge_in=True,
        on_reply_token=printer.token,
        on_turn_end=turn_end,
    )

    def command(text: str) -> None:
        if text == "/reset":
            orchestrator.reset_conversation()
            out.write("(conversation cleared)\n")
        else:
            out.write(HELP + "\n")
        out.flush()

    text_input.on_command = command
    return orchestrator, text_input


async def run_chat(settings: Settings, speak: bool = False) -> None:
    from voice_assistant.llm.client import LLMConfig, StreamingLLMClient, warm_up_llm

    if mock_models_enabled():
        from voice_assistant.mocks import MockLLMClient

        llm = MockLLMClient()
    else:
        llm = StreamingLLMClient(
            LLMConfig(
                model_path=settings.model_path,
                n_ctx=settings.llm_context_size,
                n_gpu_layers=settings.n_gpu_layers,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )
        )
        await warm_up_llm(llm, settings.assistant_system_prompt)

    tts = player = None
    if speak and not mock_models_enabled():
        from voice_assistant.tts.player import AudioPlayer
        from voice_assistant.tts.stream import PiperConfig, PiperStreamingTTS

        tts = PiperStreamingTTS(
            PiperConfig(settings.piper_voice_path, settings.tts_sample_rate),
            playback_queue=AudioChunkQueue(maxsize=settings.tts_queue_maxsize),
        )
        player = AudioPlayer(sample_rate=tts.sample_rate, blocksize=settings.player_blocksize)

    orchestrator, _ = build_chat(settings, llm, speak=speak, tts=tts, player=player)
    print(HELP, flush=True)
    await orchestrator.run()
