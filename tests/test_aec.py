from __future__ import annotations

import struct

import numpy as np
import pytest

from voice_assistant.asr.stream import StreamingASR
from voice_assistant.asr.vad import VADConfig, VoiceActivityDetector
from voice_assistant.audio.aec import EchoCanceller, LinearResampler, create_echo_canceller, echo_cancellation_available

SR = 16_000
FRAME_20MS = SR // 50


def test_linear_resampler_is_continuous_across_blocks() -> None:
    t = np.arange(22_050) / 22_050
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    resampler = LinearResampler(22_050, SR)
    blocks = [resampler.process(tone[i : i + 128]) for i in range(0, len(tone), 128)]
    out = np.concatenate(blocks)

    assert abs(len(out) - SR) <= 1
    expected = np.sin(2 * np.pi * 440 * (np.arange(len(out)) / SR - 1 / 22_050))
    # Linear interpolation of a 440 Hz tone is accurate to well under 1%
    # (the first sample interpolates from the silence before the stream).
    assert np.max(np.abs(out[1:] - expected[1:])) < 0.01


class FakeAPM:
    """Records what AEC3 would be given; 'cancels' by subtracting the reference."""

    def __init__(self) -> None:
        self.reference: list[bytes] = []
        self.capture: list[bytes] = []

    def process_reverse_stream(self, frame) -> None:
        self.reference.append(bytes(frame.data))

    def process_stream(self, frame) -> None:
        self.capture.append(bytes(frame.data))
        ref = np.frombuffer(self.reference[-1], dtype=np.int16)
        cap = np.frombuffer(bytes(frame.data), dtype=np.int16)
        frame.data[:] = (cap - ref).astype(np.int16)


needs_livekit = pytest.mark.skipif(not echo_cancellation_available(), reason="livekit not installed")


@needs_livekit
def test_reference_before_the_microphone_starts_is_dropped() -> None:
    apm = FakeAPM()
    aec = EchoCanceller(SR, apm=apm)

    aec.push_playback(np.full(SR, 0.5, dtype=np.float32), SR)
    aec.process(b"\x00\x00" * FRAME_20MS)

    # No reference yet: AEC3 is fed silence, never audio older than the microphone.
    assert apm.reference == [b"\x00\x00" * (SR // 100)] * 2


@needs_livekit
def test_reference_and_microphone_are_consumed_in_lockstep() -> None:
    apm = FakeAPM()
    aec = EchoCanceller(SR, apm=apm)
    aec.process(b"\x00\x00" * FRAME_20MS)  # microphone started

    played = np.linspace(-0.5, 0.5, FRAME_20MS * 3).astype(np.float32)
    aec.push_playback(played, SR)
    echo = (played * 32767).astype(np.int16).tobytes()
    out = aec.process(echo[: FRAME_20MS * 2]) + aec.process(echo[FRAME_20MS * 2 : FRAME_20MS * 4])

    assert len(out) == FRAME_20MS * 4
    assert np.abs(np.frombuffer(out, dtype=np.int16)).max() <= 1
    assert len(apm.reference) == 6


@needs_livekit
def test_playback_at_another_rate_is_resampled_for_the_reference() -> None:
    apm = FakeAPM()
    aec = EchoCanceller(SR, apm=apm)
    aec.process(b"\x00\x00" * FRAME_20MS)

    aec.push_playback(np.zeros(22_050, dtype=np.float32), 22_050)

    assert abs(len(aec._reference) // 2 - SR) <= 1


@needs_livekit
def test_reference_buffer_is_bounded() -> None:
    aec = EchoCanceller(SR, max_reference_s=0.5, apm=FakeAPM())
    aec.process(b"\x00\x00" * FRAME_20MS)

    for _ in range(10):
        aec.push_playback(np.zeros(SR, dtype=np.float32), SR)

    assert len(aec._reference) == SR  # 0.5 s of 16-bit audio


@needs_livekit
def test_webrtc_aec_removes_echo_but_keeps_the_user() -> None:
    aec = EchoCanceller(SR)
    rng = np.random.default_rng(0)
    noise = np.convolve(rng.standard_normal(SR * 5), np.ones(8) / 8, "same")
    gate = (np.sin(np.arange(len(noise)) / SR * 2 * np.pi * 1.5) > 0).astype(float)
    played = (noise * gate * 0.1).astype(np.float32)
    delay = 120 * SR // 1000

    rms_in, rms_out = [], []
    for i in range(0, len(played) - FRAME_20MS + 1, FRAME_20MS):
        aec.push_playback(played[i : i + FRAME_20MS], SR)
        echo_start = i - delay
        mic = np.zeros(FRAME_20MS)
        if echo_start >= 0:
            mic = played[echo_start : echo_start + FRAME_20MS] * 0.6 * 32767
        out = np.frombuffer(aec.process(mic.astype(np.int16).tobytes()), dtype=np.int16)
        if i >= 3 * SR:
            rms_in.append(np.sqrt(np.mean(mic**2)))
            rms_out.append(np.sqrt(np.mean(out.astype(float) ** 2)))

    # At least 30 dB of echo suppression once converged.
    assert np.mean(rms_out) < np.mean(rms_in) / 30


def test_asr_passes_microphone_frames_through_the_echo_canceller() -> None:
    class Canceller:
        def __init__(self) -> None:
            self.frames = 0

        def process(self, frame: bytes) -> bytes:
            self.frames += 1
            return b"\x00" * len(frame)  # cancels everything

    class Rec:
        streaming = True

        def accept_waveform(self, frame: bytes) -> None:
            pass

        def partial_result(self):
            return "", 0.0

        def final_result(self, utterance: bytes):
            return "heard", 1.0

        def reset(self) -> None:
            pass

    canceller = Canceller()
    asr = StreamingASR(
        sample_rate=SR,
        chunk_size=FRAME_20MS,
        vad=VoiceActivityDetector(VADConfig(sample_rate=SR, frame_ms=20, mode="energy")),
        model_path="",
        recognizer=Rec(),
        echo_canceller=canceller,
    )
    loud = struct.pack("<" + "h" * FRAME_20MS, *([3000] * FRAME_20MS))

    events = [e for _ in range(20) for e in asr.process_frame(loud)]
    asr.muted = True
    asr.process_frame(loud)

    # The loud "echo" was removed, so no speech was detected; muted frames still feed it.
    assert events == []
    assert canceller.frames == 21


def test_echo_cancellation_off_returns_none() -> None:
    assert create_echo_canceller("off", SR) is None


def test_echo_cancellation_on_without_livekit_fails_clearly(monkeypatch) -> None:
    import voice_assistant.audio.aec as aec

    monkeypatch.setattr(aec, "echo_cancellation_available", lambda: False)

    assert aec.create_echo_canceller("auto", SR) is None
    with pytest.raises(RuntimeError, match="voice-assistant\\[aec\\]"):
        aec.create_echo_canceller("on", SR)
