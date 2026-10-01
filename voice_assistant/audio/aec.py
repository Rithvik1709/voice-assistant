"""Acoustic echo cancellation, so Vaani does not hear itself on speakers.

The microphone picks up Vaani's own voice from the speakers. Echo
cancellation subtracts it: everything the player sends to the speakers is
recorded as the reference, and WebRTC's AEC3 (through the LiveKit SDK)
removes its echo from each microphone frame before voice detection and
speech recognition see it. Barge-in then works without headphones.

Alignment: reference and microphone audio are consumed in lockstep, sample
for sample. Reference recorded before the microphone starts is dropped, so
the reference never lags the echo it should cancel; AEC3 estimates the
remaining delay (the speaker and microphone latency) itself.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# AEC3 processes 10 ms frames.
_FRAME_MS = 10


def echo_cancellation_available() -> bool:
    try:
        from livekit import rtc  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


class LinearResampler:
    """Streaming linear resampler that stays continuous across blocks."""

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self.step = src_rate / dst_rate
        # Position of the next output sample, relative to the last input sample of the previous block.
        self._pos = 0.0
        self._prev = 0.0

    def process(self, block: np.ndarray) -> np.ndarray:
        n = len(block)
        if n == 0:
            return block.astype(np.float32)
        # Coordinate 0 is the previous block's last sample, k is block[k - 1].
        signal = np.concatenate(([self._prev], block))
        positions = np.arange(self._pos, n, self.step)
        out = np.interp(positions, np.arange(n + 1), signal).astype(np.float32)
        self._pos = (positions[-1] + self.step - n) if len(positions) else self._pos - n
        self._prev = float(block[-1])
        return out


class EchoCanceller:
    """Removes the assistant's own voice from microphone audio.

    `push_playback` is called from the audio output callback with every block
    sent to the speakers; `process` is called with each microphone frame and
    returns it with the echo removed.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        noise_suppression: bool = True,
        max_reference_s: float = 2.0,
        apm: Any | None = None,
    ) -> None:
        from livekit import rtc  # type: ignore

        self._rtc = rtc
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * _FRAME_MS // 1000
        self._frame_bytes = self.frame_samples * 2
        self._apm = apm or rtc.AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=noise_suppression,
            high_pass_filter=True,
        )
        self._reference = bytearray()
        self._max_reference = int(max_reference_s * sample_rate) * 2
        self._lock = threading.Lock()
        self._resamplers: dict[int, LinearResampler] = {}
        self._capturing = False
        self._pending = bytearray()

    def push_playback(self, audio: np.ndarray, sample_rate: int) -> None:
        """Record float32 audio sent to the speakers (any sample rate)."""
        if not self._capturing:
            return
        if sample_rate != self.sample_rate:
            resampler = self._resamplers.get(sample_rate)
            if resampler is None:
                resampler = self._resamplers[sample_rate] = LinearResampler(sample_rate, self.sample_rate)
            audio = resampler.process(audio)
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        with self._lock:
            self._reference.extend(pcm16)
            overflow = len(self._reference) - self._max_reference
            if overflow > 0:
                # The microphone side stalled: drop the oldest reference.
                del self._reference[:overflow]

    def process(self, frame: bytes) -> bytes:
        """Return a microphone frame (any multiple of 10 ms) with the echo removed."""
        self._capturing = True
        self._pending.extend(frame)
        out = bytearray()
        while len(self._pending) >= self._frame_bytes:
            capture = bytes(self._pending[: self._frame_bytes])
            del self._pending[: self._frame_bytes]
            out.extend(self._process_10ms(capture))
        return bytes(out)

    def _process_10ms(self, capture: bytes) -> bytes:
        with self._lock:
            reference = bytes(self._reference[: self._frame_bytes])
            del self._reference[: self._frame_bytes]
        if len(reference) < self._frame_bytes:
            # Nothing is playing yet (or playback fell behind): silence.
            reference = reference.ljust(self._frame_bytes, b"\x00")
        rtc = self._rtc
        n = self.frame_samples
        self._apm.process_reverse_stream(rtc.AudioFrame(reference, self.sample_rate, 1, n))
        frame = rtc.AudioFrame(bytearray(capture), self.sample_rate, 1, n)
        self._apm.process_stream(frame)
        return bytes(frame.data)


def create_echo_canceller(mode: str, sample_rate: int, noise_suppression: bool = True) -> EchoCanceller | None:
    """The echo canceller for ECHO_CANCELLATION=`mode` ("auto", "on" or "off"), or None."""
    if mode == "off":
        return None
    if not echo_cancellation_available():
        if mode == "on":
            raise RuntimeError(
                "Echo cancellation needs the LiveKit SDK. Install it with `pip install 'voice-assistant[aec]'`."
            )
        logger.info("Echo cancellation unavailable (pip install 'voice-assistant[aec]'); use headphones for barge-in")
        return None
    logger.info("Echo cancellation on (WebRTC AEC3)")
    return EchoCanceller(sample_rate=sample_rate, noise_suppression=noise_suppression)
