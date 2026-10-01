from __future__ import annotations

import asyncio
from datetime import datetime

from voice_assistant.actions import BasicIntentActions
from voice_assistant.asr.stream import ASREvent
from voice_assistant.benchmark import BenchmarkTracker
from voice_assistant.nlu import SimpleIntentClassifier
from voice_assistant.pipeline.orchestrator import FALLBACK_REPLY, VoicePipelineOrchestrator
from voice_assistant.tts.queue import AudioChunkQueue


class FakeASR:
    def __init__(self) -> None:
        self.events: asyncio.Queue[ASREvent] = asyncio.Queue()
        self.muted = False
        self.muted_history: list[bool] = []

    def __setattr__(self, name, value):
        if name == "muted" and hasattr(self, "muted_history"):
            self.muted_history.append(value)
        object.__setattr__(self, name, value)

    async def stream_events(self):
        while True:
            yield await self.events.get()

    def say(self, text: str) -> None:
        self.events.put_nowait(ASREvent("final", text, 1.0, 0))

    def start_speaking(self) -> None:
        self.events.put_nowait(ASREvent("speech_start", "", 0.0, 0))


class FakeLLM:
    def __init__(self, tokens: list[str], delay: float = 0.0, fail: bool = False) -> None:
        self.tokens = tokens
        self.delay = delay
        self.fail = fail
        self.calls: list[list[dict[str, str]]] = []
        self.cancelled = 0

    async def stream_tokens(self, messages, out_queue, bench=None):
        self.calls.append([dict(m) for m in messages])
        if self.fail:
            raise RuntimeError("model crashed")
        try:
            for tok in self.tokens:
                await out_queue.put(tok)
                await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return "".join(self.tokens)


class FakeTTS:
    def __init__(self) -> None:
        self.playback_queue = AudioChunkQueue(maxsize=8)
        self.sentences: list[str] = []
        self.languages: list[str | None] = []
        self.cancelled = 0
        self.sample_rate = 22050

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def synthesize_sentence(self, sentence: str, language: str | None = None) -> bool:
        self.sentences.append(sentence)
        self.languages.append(language)
        return True

    def cancel_pending(self) -> None:
        self.cancelled += 1

    async def flush(self) -> None:
        pass


class FakePlayer:
    sample_rate = 22050

    def __init__(self) -> None:
        self.interrupts = 0
        self.is_playing = False

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def play(self, chunk) -> None:
        pass

    def interrupt(self) -> None:
        self.interrupts += 1


def make(llm: FakeLLM, barge_in: bool = True, **kwargs):
    asr, tts, player = FakeASR(), FakeTTS(), FakePlayer()
    orch = VoicePipelineOrchestrator(
        asr=asr,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        tts=tts,  # type: ignore[arg-type]
        player=player,  # type: ignore[arg-type]
        bench=BenchmarkTracker(),
        ack_tone_ms=0,
        system_prompt="Be brief.",
        barge_in=barge_in,
        **kwargs,
    )
    return orch, asr, tts, player


async def wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.005)


async def run_with(orch, body) -> None:
    runner = asyncio.create_task(orch.run())
    try:
        await body()
    finally:
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)


async def test_turn_is_spoken_and_recorded() -> None:
    llm = FakeLLM(["Paris", " is", " the", " capital", "."])
    orch, asr, tts, _ = make(llm)

    async def body():
        asr.say("what is the capital of france")
        await wait_for(lambda: not orch.is_responding and len(orch.conversation_history) == 3)

    await run_with(orch, body)

    assert "".join(tts.sentences).replace(" ", "") == "Parisisthecapital."
    assert orch.conversation_history[1] == {"role": "user", "content": "what is the capital of france"}
    assert orch.conversation_history[2]["content"] == "Paris is the capital."
    assert llm.calls[0][0] == {"role": "system", "content": "Be brief."}


async def test_barge_in_cancels_response_and_keeps_listening() -> None:
    llm = FakeLLM([f" word{i}" for i in range(200)], delay=0.01)
    orch, asr, tts, player = make(llm)

    async def body():
        asr.say("tell me a long story")
        await wait_for(lambda: len(tts.sentences) >= 1)
        asr.start_speaking()
        await wait_for(lambda: llm.cancelled == 1)
        assert not orch.is_responding

        llm.tokens, llm.delay = ["Sure", "."], 0.0
        asr.say("never mind, just say sure")
        await wait_for(lambda: len(llm.calls) == 2 and not orch.is_responding)

    await run_with(orch, body)

    assert player.interrupts == 1
    assert tts.cancelled == 1
    # What the user heard before interrupting is kept as the assistant turn.
    roles = [m["role"] for m in orch.conversation_history]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    assert orch.conversation_history[2]["content"].startswith("word0")
    assert orch.conversation_history[-1]["content"] == "Sure."


async def test_speech_while_idle_does_not_interrupt() -> None:
    orch, asr, _, player = make(FakeLLM(["Hi", "."]))

    async def body():
        asr.start_speaking()
        await asyncio.sleep(0.05)

    await run_with(orch, body)
    assert player.interrupts == 0


async def test_llm_failure_speaks_fallback_and_recovers() -> None:
    llm = FakeLLM(["ok"], fail=True)
    orch, asr, tts, _ = make(llm)

    async def body():
        asr.say("first question")
        await wait_for(lambda: any("Sorry" in s for s in tts.sentences) and not orch.is_responding)
        llm.fail = False
        asr.say("second question")
        await wait_for(lambda: len(llm.calls) == 2 and not orch.is_responding)

    await run_with(orch, body)
    assert " ".join(tts.sentences).startswith(FALLBACK_REPLY.split()[0])


async def test_half_duplex_mutes_mic_while_responding() -> None:
    orch, asr, _, _ = make(FakeLLM(["Done", "."]), barge_in=False)

    async def body():
        asr.say("hello there friend how are you")
        await wait_for(lambda: asr.muted_history[-2:] == [True, False], timeout=3.0)

    await run_with(orch, body)
    assert asr.muted is False


async def test_actions_answer_without_llm() -> None:
    llm = FakeLLM(["unused"])
    orch, asr, tts, _ = make(
        llm,
        nlu=SimpleIntentClassifier(),
        action_handler=BasicIntentActions(clock=lambda: datetime(2026, 9, 26, 15, 4)),
    )

    async def body():
        asr.say("what time is it")
        await wait_for(lambda: len(orch.conversation_history) == 3 and not orch.is_responding)

    await run_with(orch, body)
    assert llm.calls == []
    assert "3:04 PM" in " ".join(tts.sentences)


async def test_previous_turn_cleanup_does_not_end_new_turn() -> None:
    orch, _, _, player = make(FakeLLM(["A", "."]))
    player.is_playing = True  # turn 1 audio still playing

    first = orch._begin_response()
    orch._schedule_finish(first)
    second = orch._begin_response()  # user asked again before audio drained
    orch._schedule_finish(first)  # late end-of-turn marker for turn 1

    player.is_playing = False
    await asyncio.sleep(0.1)
    assert orch.is_responding  # turn 2 is still in progress

    orch._schedule_finish(second)
    await wait_for(lambda: not orch.is_responding)
