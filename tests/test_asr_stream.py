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


class ScriptedPartials(FakeRecognizer):
    def __init__(self, partial: str) -> None:
        super().__init__()
        self.partial = partial

    def partial_result(self) -> tuple[str, float]:
        return self.partial, 0.5


def _frames_until_final(partial: str, endpoint_ms: int = 100, hold_ms: int = 400) -> int:
    rec = ScriptedPartials(partial)
    vad = VoiceActivityDetector(VADConfig(sample_rate=16_000, frame_ms=FRAME_MS, mode="energy"))
    asr = StreamingASR(
        sample_rate=16_000, chunk_size=SAMPLES, vad=vad, model_path="",
        endpoint_silence_ms=endpoint_ms, hold_silence_ms=hold_ms, recognizer=rec,
    )
    feed(asr, [SPEECH] * 5)
    for n in range(1, 100):
        if any(e.type == "final" for e in asr.process_frame(SILENCE)):
            return n
    raise AssertionError("no final")


def test_complete_phrase_uses_normal_endpoint() -> None:
    assert _frames_until_final("what time is it") * FRAME_MS == 120


def test_dangling_word_holds_the_turn_longer() -> None:
    assert _frames_until_final("turn on the") * FRAME_MS == 420
    assert _frames_until_final("mujhe batao ki") * FRAME_MS == 420


def test_looks_unfinished() -> None:
    from voice_assistant.asr.stream import looks_unfinished

    assert looks_unfinished("I want to go to")
    assert looks_unfinished("um")
    assert not looks_unfinished("what is the weather in paris")
    assert not looks_unfinished("")


class FakeSegment:
    def __init__(self, text: str, no_speech_prob: float = 0.01, avg_logprob: float = -0.1) -> None:
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob


class FakeWhisperModel:
    def __init__(self, segments: list[FakeSegment]) -> None:
        self.segments = segments
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"samples": len(audio), **kwargs})
        return iter(self.segments), None


def test_whisper_recognizer_drops_hallucinated_silence() -> None:
    from voice_assistant.asr.stream import load_recognizer

    model = FakeWhisperModel([FakeSegment(" Turn on the lights."), FakeSegment(" Thank you.", no_speech_prob=0.9)])
    rec = load_recognizer("whisper", "base.en", 16_000, language="en", shared=model)

    text, confidence = rec.final_result(SPEECH * 10)

    assert text == "Turn on the lights."
    assert 0.8 < confidence <= 1.0
    call = model.calls[0]
    assert call["samples"] == SAMPLES * 10
    assert call["language"] == "en"
    assert call["vad_filter"] is False
    assert rec.final_result(b"") == ("", 0.0)
    assert rec.streaming is False


def test_unknown_asr_backend_is_rejected() -> None:
    import pytest

    from voice_assistant.asr.stream import load_recognizer

    with pytest.raises(ValueError, match="Unsupported ASR backend"):
        load_recognizer("nope", "", 16_000)
