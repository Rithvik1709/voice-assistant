from __future__ import annotations

import asyncio
import struct

import numpy as np
import pytest

from tests.test_orchestrator import FakeLLM, make, run_with, wait_for
from voice_assistant.asr.stream import ASREvent, StreamingASR
from voice_assistant.asr.vad import VADConfig, VoiceActivityDetector
from voice_assistant.config import Settings
from voice_assistant.wakeword import OpenWakeWordDetector, WakeWordGate, parse_wake_phrases


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def gate(phrases: str = "hey vaani", follow_up_s: float = 8.0) -> tuple[WakeWordGate, Clock]:
    clock = Clock()
    return WakeWordGate(parse_wake_phrases(phrases), follow_up_s=follow_up_s, clock=clock), clock


def test_parse_wake_phrases_normalizes_and_dedupes() -> None:
    assert parse_wake_phrases("Hey Vaani!, ok  vaani, hey vaani,") == ("hey vaani", "ok vaani")
    assert parse_wake_phrases("") == ()


@pytest.mark.parametrize(
    ("text", "rest"),
    [
        ("Hey Vaani, what time is it?", "what time is it?"),
        ("hey vani what's the weather", "what's the weather"),  # Whisper's spelling varies
        ("Hey, Vaani. Turn it up.", "Turn it up."),
        ("Okay hey Vaani set a timer", "set a timer"),  # a word or two before it is fine
        ("Hey Vaani, a quick question", "a quick question"),  # "a" is not swallowed
        ("Hey Vaani.", ""),
        ("hey vaani", ""),
    ],
)
def test_wake_phrase_is_found_and_stripped(text: str, rest: str) -> None:
    g, _ = gate()
    assert g.match(text) == rest


@pytest.mark.parametrize(
    "text",
    ["what time is it", "hey there how are you", "I was telling my friend hey vaani is cool", "have funny day"],
)
def test_speech_without_the_wake_phrase_is_not_matched(text: str) -> None:
    g, _ = gate()
    assert g.match(text) is None


def test_alternative_phrases() -> None:
    g, _ = gate("hey vaani, vaani")
    assert g.match("Vaani, play some music") == "play some music"


def test_follow_up_window() -> None:
    g, clock = gate(follow_up_s=8)
    assert not g.awake

    g.wake()
    clock.now += 7.9
    assert g.awake
    clock.now += 0.2
    assert not g.awake

    g.hold()
    clock.now += 1000
    assert g.awake
    g.sleep()
    assert not g.awake


def test_hotwords_are_the_wake_words() -> None:
    g, _ = gate("hey vaani, ok vaani")
    assert g.hotwords() == "hey vaani ok"


class FakeEngine:
    """openWakeWord stand-in: fires on loud audio."""

    def __init__(self) -> None:
        self.steps = 0
        self.resets = 0

    def predict(self, step):
        self.steps += 1
        assert len(step) == OpenWakeWordDetector.STEP_SAMPLES
        return {"hey_vaani": 0.9 if np.abs(step).mean() > 1000 else 0.01}

    def reset(self) -> None:
        self.resets += 1


SAMPLES = 320
LOUD = struct.pack("<" + "h" * SAMPLES, *([3000] * SAMPLES))
QUIET = b"\x00\x00" * SAMPLES


def test_detector_scores_80ms_steps_and_resets_after_firing() -> None:
    engine = FakeEngine()
    detector = OpenWakeWordDetector("hey_vaani", engine=engine)

    assert [detector.detect(QUIET) for _ in range(4)] == [False] * 4
    assert engine.steps == 1  # 4 x 20 ms = one 80 ms step

    fired = [detector.detect(LOUD) for _ in range(4)]
    assert fired == [False, False, False, True]
    assert engine.resets == 1


class Rec:
    streaming = True

    def __init__(self) -> None:
        self.fed = 0

    def accept_waveform(self, frame: bytes) -> None:
        self.fed += 1

    def partial_result(self):
        return "", 0.0

    def final_result(self, utterance: bytes):
        return "what time is it", 1.0

    def reset(self) -> None:
        pass


def test_asr_only_recognizes_speech_after_the_acoustic_wake_word() -> None:
    g, clock = gate("", follow_up_s=5)
    engine = FakeEngine()
    rec = Rec()
    asr = StreamingASR(
        sample_rate=16_000,
        chunk_size=SAMPLES,
        vad=VoiceActivityDetector(VADConfig(sample_rate=16_000, frame_ms=20, mode="energy")),
        model_path="",
        endpoint_silence_ms=100,
        recognizer=rec,
        wake_detector=OpenWakeWordDetector("hey_vaani", engine=engine),
        wake_gate=g,
    )

    # Asleep: quiet speech-free audio never reaches the recognizer.
    for _ in range(8):
        assert asr.process_frame(QUIET) == []
    assert rec.fed == 0

    events = [e for _ in range(4) for e in asr.process_frame(LOUD)]
    assert [e.type for e in events] == ["wake"]
    assert g.awake

    events = [e for f in [LOUD] * 5 + [QUIET] * 8 for e in asr.process_frame(f)]
    assert [e.text for e in events if e.type == "final"] == ["what time is it"]

    clock.now += 6  # follow-up window over: back to the detector only
    fed = rec.fed
    for _ in range(3):
        asr.process_frame(LOUD)
    assert rec.fed == fed


async def test_orchestrator_answers_only_when_addressed() -> None:
    llm = FakeLLM(["It's", " noon", "."])
    g = WakeWordGate(parse_wake_phrases("hey vaani"), follow_up_s=0.3)
    orch, asr, tts, _ = make(llm, wake_gate=g)

    async def body():
        asr.say("I was just talking to my friend")
        await asyncio.sleep(0.05)
        assert llm.calls == []

        asr.say("Hey Vaani, what time is it?")
        await wait_for(lambda: len(llm.calls) == 1 and not orch.is_responding)
        assert llm.calls[0][-1] == {"role": "user", "content": "what time is it?"}

        # Follow-up within the window: no wake word needed.
        asr.say("and tomorrow")
        await wait_for(lambda: len(llm.calls) == 2 and not orch.is_responding)

        await asyncio.sleep(0.4)  # window closes
        asr.say("what about friday")
        await asyncio.sleep(0.05)
        assert len(llm.calls) == 2

    await run_with(orch, body)


async def test_wake_word_alone_plays_a_chime_and_waits_for_the_request() -> None:
    llm = FakeLLM(["Sure", "."])
    g = WakeWordGate(parse_wake_phrases("hey vaani"), follow_up_s=5)
    orch, asr, tts, _ = make(llm, wake_gate=g)
    queued = []
    original_put = orch.audio_queue.put

    async def spy(chunk):
        queued.append(chunk)
        await original_put(chunk)

    orch.audio_queue.put = spy  # type: ignore[method-assign]

    async def body():
        asr.say("Hey Vaani.")
        await wait_for(lambda: g.awake and queued)
        assert llm.calls == []

        asr.say("tell me a joke")
        await wait_for(lambda: len(llm.calls) == 1 and not orch.is_responding)

    await run_with(orch, body)


async def test_acoustic_wake_event_plays_the_chime() -> None:
    g = WakeWordGate((), follow_up_s=5)
    orch, asr, _, _ = make(FakeLLM(["ok"]), wake_gate=g)
    chimes = []
    orch._enqueue_wake_tone = lambda: chimes.append(1) or asyncio.sleep(0)  # type: ignore[method-assign]

    async def body():
        asr.events.put_nowait(ASREvent("wake", "", 1.0, 0))
        await wait_for(lambda: chimes)

    await run_with(orch, body)


def test_settings_build_the_wake_word(monkeypatch) -> None:
    monkeypatch.setenv("WAKE_WORD", "Hey Vaani, ok vaani")
    monkeypatch.setenv("WAKE_WORD_FOLLOW_UP_S", "4")
    settings = Settings()

    built = settings.build_wake_gate()

    assert settings.wake_phrases == ("hey vaani", "ok vaani")
    assert built is not None and built.follow_up_s == 4
    assert settings.build_wake_detector() is None


def test_no_wake_word_by_default(monkeypatch) -> None:
    monkeypatch.delenv("WAKE_WORD", raising=False)
    monkeypatch.delenv("WAKE_WORD_MODEL", raising=False)
    assert Settings().build_wake_gate() is None


def test_wake_word_settings_are_validated(monkeypatch) -> None:
    monkeypatch.setenv("WAKE_WORD_THRESHOLD", "1.5")
    monkeypatch.setenv("WAKE_WORD", "!!!")
    problems = " ".join(Settings().check_ranges())
    assert "WAKE_WORD_THRESHOLD" in problems
    assert "WAKE_WORD has no words" in problems
