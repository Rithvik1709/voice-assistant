from __future__ import annotations

import os
import struct
import wave
import pytest

from voice_assistant.asr.vad import VADConfig, VoiceActivityDetector


def _pcm_frame(value: int, samples: int) -> bytes:
    return struct.pack("<" + "h" * samples, *([value] * samples))


def test_vad_energy_mode_detects_speech_after_consecutive_frames() -> None:
    cfg = VADConfig(sample_rate=16_000, frame_ms=30, speech_frames_trigger=3, mode="energy")
    vad = VoiceActivityDetector(cfg)
    samples = int(cfg.sample_rate * cfg.frame_ms / 1000)

    silence = _pcm_frame(0, samples)
    speech = _pcm_frame(2000, samples)

    assert vad.is_speech(silence) is False
    assert vad.detect_barge_in([speech, speech]) is False
    assert vad.detect_barge_in([speech, speech, speech]) is True


def test_vad_detects_speech_in_noisy_audio() -> None:
    # 1. Setup VAD with 32ms frames (Required by Silero: 512 samples @ 16kHz)
    cfg = VADConfig(sample_rate=16_000, frame_ms=32, mode="silero", threshold=0.15)
    vad = VoiceActivityDetector(cfg)
    
    # Force a fresh state clean before feeding audio
    vad.reset()

    # 2. Get the file path
    noisy_file_path = os.path.join(os.path.dirname(__file__), "noisy_clip.wav")

    # 3. Read the entire audio file
    with wave.open(noisy_file_path, "rb") as wf:
        audio_data = wf.readframes(wf.getnframes())

    # Pad with trailing silence if the file is shorter than one full frame
    frame_length = vad.frame_bytes
    if len(audio_data) < frame_length:
        padding_needed = frame_length - len(audio_data)
        audio_data += b"\x00" * padding_needed

    # 4. Chop the audio into exact 32ms frames
    frames = [
        audio_data[i : i + frame_length]
        for i in range(0, len(audio_data), frame_length)
        if len(audio_data[i : i + frame_length]) == frame_length
    ]