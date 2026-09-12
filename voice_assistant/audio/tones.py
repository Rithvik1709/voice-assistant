from __future__ import annotations

import numpy as np


def make_ack_tone(sample_rate: int, duration_ms: int, frequency_hz: float = 880.0) -> bytes:
    if duration_ms <= 0:
        return b""

    duration = max(0.02, duration_ms / 1000.0)
    samples = int(sample_rate * duration)

    if samples <= 0:
        return b""

    t = np.arange(samples, dtype=np.float32) / float(sample_rate)
    tone = 0.08 * np.sin(2.0 * np.pi * frequency_hz * t)

    return (
        np.clip(tone * 32767.0, -32768.0, 32767.0)
        .astype(np.int16)
        .tobytes()
    )
