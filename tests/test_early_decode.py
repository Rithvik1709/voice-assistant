from __future__ import annotations

import struct
import threading

from voice_assistant.asr.stream import StreamingASR
from voice_assistant.asr.vad import VADConfig, VoiceActivityDetector

FRAME_MS = 20
SAMPLES = 16_000 * FRAME_MS // 1000
SPEECH = struct.pack("<" + "h" * SAMPLES, *([3000] * SAMPLES))
SILENCE = b"\x00\x00" * SAMPLES


class FakeDecoder:
    """A Whisper-like recognizer: no streaming partials, decodes whole utterances."""

    streaming = False

    def __init__(self, text: str = "what is the weather", gate: threading.Event | None = None) -> None:
        self.text = text
        self.gate = gate
        self.decodes: list[int] = []
        self.finals: list[int] = []
        self.last_language: str | None = None

    def accept_waveform(self, frame: bytes) -> None:
        pass

    def partial_result(self) -> tuple[str, float]:
        return "", 0.0

    def decode(self, utterance: bytes) -> tuple[str, float, str | None]:
        if self.gate is not None:
            self.gate.wait(5)
        self.decodes.append(len(utterance))
        return self.text, 0.9, "en"

    def final_result(self, utterance: bytes) -> tuple[str, float]:
        self.finals.append(len(utterance))
        return self.text, 0.9

    def reset(self) -> None:
        pass


def make_asr(rec: FakeDecoder, early_ms: int = 60, partial_ms: int = 0, endpoint_ms: int = 200) -> StreamingASR:
    vad = VoiceActivityDetector(VADConfig(sample_rate=16_000, frame_ms=FRAME_MS, mode="energy"))
    return StreamingASR(
        sample_rate=16_000,
        chunk_size=SAMPLES,
        vad=vad,
        model_path="",
        endpoint_silence_ms=endpoint_ms,
        hold_silence_ms=1000,
        recognizer=rec,
        early_decode_ms=early_ms,
        partial_interval_ms=partial_ms,
    )


def run(asr: StreamingASR, frames: list[bytes]):
    events = []
    for frame in frames:
        events.extend(asr.process_frame(frame))
    return events


def test_endpoint_reuses_the_decode_started_during_the_pause() -> None:
    rec = FakeDecoder()
    asr = make_asr(rec)

    events = run(asr, [SPEECH] * 10 + [SILENCE] * 15)

    finals = [e for e in events if e.type == "final"]
    assert [e.text for e in finals] == ["what is the weather"]
    assert finals[0].language == "en"
    # One background decode and no decode at the endpoint.
    assert len(rec.decodes) == 1
    assert rec.finals == []


def test_speech_after_the_early_decode_forces_a_fresh_decode() -> None:
    rec = FakeDecoder()
    asr = make_asr(rec)

    # Pause long enough for an early decode, then keep talking.
    events = run(asr, [SPEECH] * 5 + [SILENCE] * 5 + [SPEECH] * 5)
    asr._snapshot.future.result(5)  # let the stale decode finish
    events += run(asr, [SILENCE] * 15)

    assert [e.text for e in events if e.type == "final"] == ["what is the weather"]
    # The second pause decoded again, covering the longer utterance, and that was reused.
    assert len(rec.decodes) == 2
    assert rec.decodes[1] > rec.decodes[0]
    assert rec.finals == []


def test_endpoint_waits_for_an_unfinished_early_decode() -> None:
    gate = threading.Event()
    rec = FakeDecoder(gate=gate)
    asr = make_asr(rec, endpoint_ms=100)

    run(asr, [SPEECH] * 5 + [SILENCE] * 4)
    assert asr._snapshot is not None and not asr._snapshot.future.done()

    timer = threading.Timer(0.05, gate.set)
    timer.start()
    events = run(asr, [SILENCE] * 10)
    timer.join()

    assert [e.text for e in events if e.type == "final"] == ["what is the weather"]
    assert len(rec.decodes) == 1
    assert rec.finals == []


def test_early_decode_text_triggers_the_hold_for_unfinished_sentences() -> None:
    rec = FakeDecoder(text="turn on the")
    asr = make_asr(rec, endpoint_ms=200)

    events = run(asr, [SPEECH] * 5 + [SILENCE] * 4)
    asr._snapshot.future.result(5)
    # 200 ms of silence would normally end the turn; "the" holds it open.
    events += run(asr, [SILENCE] * 15)

    assert "final" not in [e.type for e in events]
    assert [e.text for e in events if e.type == "partial"] == ["turn on the"]
    assert asr.in_utterance

    events = run(asr, [SILENCE] * 40)
    assert [e.text for e in events if e.type == "final"] == ["turn on the"]


def test_live_partials_while_speaking() -> None:
    rec = FakeDecoder()
    asr = make_asr(rec, early_ms=0, partial_ms=100)

    run(asr, [SPEECH] * 6)
    asr._snapshot.future.result(5)
    events = run(asr, [SPEECH])

    assert [e.text for e in events if e.type == "partial"] == ["what is the weather"]


def test_disabled_early_decode_decodes_at_the_endpoint() -> None:
    rec = FakeDecoder()
    asr = make_asr(rec, early_ms=0, partial_ms=0)

    events = run(asr, [SPEECH] * 10 + [SILENCE] * 15)

    assert [e.text for e in events if e.type == "final"] == ["what is the weather"]
    assert rec.decodes == []
    assert len(rec.finals) == 1


def test_streaming_recognizers_never_decode_early() -> None:
    rec = FakeDecoder()
    rec.streaming = True
    asr = make_asr(rec)

    assert asr.early_decode_s is None
    assert asr.partial_interval_s is None


def test_close_stops_the_decoder_thread() -> None:
    rec = FakeDecoder()
    asr = make_asr(rec)
    run(asr, [SPEECH] * 5 + [SILENCE] * 5)

    asr.close()

    assert asr._executor is None
