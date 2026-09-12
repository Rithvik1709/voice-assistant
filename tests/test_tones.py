from voice_assistant.audio import make_ack_tone


def test_make_ack_tone_returns_pcm16() -> None:
    pcm = make_ack_tone(sample_rate=16000, duration_ms=50)

    assert len(pcm) == 1600
    assert pcm != b"\x00" * len(pcm)
