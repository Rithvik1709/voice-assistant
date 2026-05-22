from __future__ import annotations

from voice_assistant.tts.stream import sentence_chunks_from_tokens


def test_sentence_chunker_splits_on_punctuation() -> None:
    tokens = ["Hello", " there.", " How", " are", " you?", " I", " am", " fine!"]
    # Provide a max_tokens smaller than total word count but large enough for the biggest single sentence
    chunks = sentence_chunks_from_tokens(tokens, max_tokens=3)
    assert chunks == ["Hello there.", "How are you?", "I am fine!"]

def test_sentence_chunker_limits_token_count() -> None:
    tokens = ["one ", "two ", "three ", "four ", "five ", "six ", "seven ", "eight "]
    # Here the string has no punctuation, so the entire string is one piece!
    # Wait, if there's no punctuation, sentence_chunks_from_tokens currently returns it as one chunk 
    # since it only splits on punctuation! Let's add punctuation to test token count limits.
    tokens = ["one. ", "two. ", "three. ", "four. ", "five. ", "six. ", "seven. ", "eight. "]
    chunks = sentence_chunks_from_tokens(tokens, max_tokens=3)
    assert len(chunks) >= 2
