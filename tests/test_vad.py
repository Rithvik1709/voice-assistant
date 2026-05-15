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
    # 1. We will set up the VAD with our new 0.3 threshold
    cfg = VADConfig(sample_rate=16_000, frame_ms=30, mode="silero", threshold=0.3)
    vad = VoiceActivityDetector(cfg)
    
    # 2. We will look for Rithvik's noisy audio file (we will change this name later)
    noisy_file_path = os.path.join(os.path.dirname(__file__), "noisy_clip.wav")
    
    # If he hasn't given us the file yet, we just skip the test for now so it doesn't crash!
    if not os.path.exists(noisy_file_path):
        pytest.skip("Noisy audio clip not found yet. Waiting for Rithvik to upload.")

    # 3. Read the audio file
    with wave.open(noisy_file_path, "rb") as wf:
        audio_data = wf.readframes(wf.getnframes())
    
    # 4. Prove that the VAD correctly detects speech in the noise!
    assert vad.is_speech(audio_data) is True