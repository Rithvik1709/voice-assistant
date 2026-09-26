from __future__ import annotations

import json
import struct

from voice_assistant.asr.stream import StreamingASR, _VoskRecognizer
from voice_assistant.asr.vad import VADConfig, VoiceActivityDetector

FRAME_MS = 20
SAMPLES = 16_000 * FRAME_MS // 1000
SPEECH = struct.pack("<" + "h" * SAMPLES, *([3000] * SAMPLES))
SILENCE = b"\x00\x00" * SAMPLES


class FakeRecognizer:
    streaming = True

    def __init__(self) -> None:
        self.fed: list[bytes] = []
        self.resets = 0

    def accept_waveform(self, frame: bytes) -> None:
        self.fed.append(frame)

    def partial_result(self) -> tuple[str, float]:
        return "hel", 0.5

    def final_result(self, utterance: bytes) -> tuple[str, float]:
        return ("hello" if utterance else ""), 0.9

    def reset(self) -> None:
        self.resets += 1


def make_asr(endpoint_ms: int = 100, start_frames: int = 3) -> tuple[StreamingASR, FakeRecognizer]:
    rec = FakeRecognizer()
    vad = VoiceActivityDetector(VADConfig(sample_rate=16_000, frame_ms=FRAME_MS, mode="energy"))
    asr = StreamingASR(
        sample_rate=16_000,
        chunk_size=SAMPLES,
        vad=vad,
        model_path="",
        endpoint_silence_ms=endpoint_ms,
        speech_start_frames=start_frames,
        recognizer=rec,
    )
    return asr, rec


def feed(asr: StreamingASR, frames: list[bytes]) -> list[str]:
    types: list[str] = []
    for frame in frames:
        types.extend(e.type for e in asr.process_frame(frame))
    return types


def test_speech_start_fires_after_consecutive_frames() -> None:
    asr, _ = make_asr(start_frames=3)

    assert "speech_start" not in feed(asr, [SPEECH, SPEECH])
    assert feed(asr, [SPEECH]).count("speech_start") == 1
    assert "speech_start" not in feed(asr, [SPEECH] * 5)


def test_short_pause_does_not_end_utterance() -> None:
    asr, _ = make_asr(endpoint_ms=100)

    events = feed(asr, [SPEECH] * 4 + [SILENCE] * 3 + [SPEECH] * 4)

    assert "final" not in events
    assert asr.in_utterance


def test_endpoint_silence_emits_single_final() -> None:
    asr, rec = make_asr(endpoint_ms=100)

    frames = [SILENCE] * 5 + [SPEECH] * 4 + [SILENCE] * 10
    events = []
    for frame in frames:
        events.extend(asr.process_frame(frame))

    finals = [e for e in events if e.type == "final"]
    assert len(finals) == 1
    assert finals[0].text == "hello"
    assert finals[0].speech_end_ts is not None
    # Pre-roll frames from before the VAD fired are fed to the recognizer,
    # and pauses inside the utterance are fed too.
    assert rec.fed[:3] == [SILENCE] * 3
    assert not asr.in_utterance


def test_partials_are_deduplicated() -> None:
    asr, _ = make_asr()
    events = []
    for frame in [SPEECH] * 6:
        events.extend(asr.process_frame(frame))

    assert [e.text for e in events if e.type == "partial"] == ["hel"]


def test_muted_asr_discards_audio_and_resets() -> None:
    asr, _ = make_asr()
    feed(asr, [SPEECH] * 4)
    asr.muted = True

    assert feed(asr, [SPEECH] * 10 + [SILENCE] * 10) == []
    assert not asr.in_utterance


def test_finalize_flushes_utterance_in_progress() -> None:
    asr, _ = make_asr(endpoint_ms=1000)
    feed(asr, [SPEECH] * 4)

    events = asr.finalize()

    assert [e.text for e in events] == ["hello"]
    assert asr.finalize() == []


def test_wrong_frame_size_is_ignored() -> None:
    asr, rec = make_asr()
    assert asr.process_frame(SPEECH[:-2]) == []
    assert rec.fed == []


class FakeKaldi:
    """Mimics Vosk: AcceptWaveform returns True when Kaldi finds an endpoint."""

    def __init__(self, script: list[tuple[bool, str]]) -> None:
        self.script = script
        self.final = ""

    def AcceptWaveform(self, _audio: bytes) -> bool:
        done, self._text = self.script.pop(0)
        return done

    def Result(self) -> str:
        return json.dumps({"text": self._text})

    def PartialResult(self) -> str:
        return json.dumps({"partial": "in progress"})

    def FinalResult(self) -> str:
        return json.dumps({"text": self.final})

    def Reset(self) -> None:
        pass


def test_vosk_segments_from_internal_endpoints_are_not_lost() -> None:
    rec = _VoskRecognizer.__new__(_VoskRecognizer)
    rec.recognizer = FakeKaldi([(False, ""), (True, "turn on"), (False, "")])
    rec._segments = []
    rec.recognizer.final = "the lights"

    for _ in range(3):
        rec.accept_waveform(b"\x00\x00")

    assert rec.partial_result()[0] == "turn on in progress"
    assert rec.final_result(b"")[0] == "turn on the lights"
    assert rec._segments == []
