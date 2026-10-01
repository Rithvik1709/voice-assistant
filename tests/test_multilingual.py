from __future__ import annotations

import asyncio
import io
import json
import struct
from pathlib import Path

import pytest

from voice_assistant.actions import BasicIntentActions
from voice_assistant.asr.stream import StreamingASR, default_whisper_model, load_recognizer
from voice_assistant.asr.vad import VADConfig, VoiceActivityDetector
from voice_assistant.chat import build_chat
from voice_assistant.config import Settings
from voice_assistant.lang import (
    base_language,
    count_words,
    ends_sentence,
    guess_language,
    parse_language_list,
    with_reply_language,
)
from voice_assistant.mocks import MockLLMClient
from voice_assistant.model_setup import (
    download_language_voices,
    env_template,
    format_model_plan,
    pick_piper_voices,
    recommended_models,
)
from voice_assistant.tts.normalize import normalize_for_speech
from voice_assistant.tts.queue import AudioChunkQueue
from voice_assistant.tts.stream import PiperConfig, PiperStreamingTTS, _wav_to_pcm, sentence_chunks_from_tokens
from voice_assistant.tts.voices import VoiceRouter, discover_voices, voice_language

FRAME_MS = 20
SAMPLES = 16_000 * FRAME_MS // 1000
SPEECH = struct.pack("<" + "h" * SAMPLES, *([3000] * SAMPLES))
SILENCE = b"\x00\x00" * SAMPLES


# ----- language helpers ---------------------------------------------------


def test_base_language_and_lists() -> None:
    assert base_language("en_US") == base_language("EN-us") == "en"
    assert base_language("") is None
    assert parse_language_list(" hi, ta,EN,hi ,") == ("hi", "ta", "en")


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("வணக்கம், எப்படி இருக்கிறீர்கள்?", "ta"),
        ("नमस्ते, आप कैसे हैं?", "hi"),
        ("こんにちは、元気ですか", "ja"),
        ("今天天气很好", "zh"),
        ("안녕하세요", "ko"),
        ("สวัสดีครับ", "th"),
        ("Καλημέρα", "el"),
        ("hello there", None),  # Latin script: many languages
        ("привет", None),  # Cyrillic: many languages
        ("مرحبا", None),  # Arabic script: Arabic, Persian, Urdu...
        ("1234 ?!", None),
    ],
)
def test_guess_language_from_script(text: str, language: str | None) -> None:
    assert guess_language(text) == language


def test_words_are_counted_by_characters_without_spaces() -> None:
    assert count_words("the cat sat") == 3
    assert count_words("今天天气很好") == 3
    assert ends_sentence("ठीक है।") and ends_sentence("好的。") and ends_sentence("ok.")
    assert not ends_sentence("ok,")


def test_reply_language_is_requested_without_touching_history() -> None:
    history = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "வணக்கம்"},
    ]
    messages = with_reply_language(history, "ta")

    assert messages[-1]["content"] == "வணக்கம்\n\n(Reply in Tamil.)"
    assert messages[0] == history[0]
    assert history[-1]["content"] == "வணக்கம்"
    assert with_reply_language(history, None) is history


# ----- sentence chunking and speech cleanup ------------------------------


@pytest.mark.parametrize(
    ("text", "chunks"),
    [
        ("नमस्ते। आप कैसे हैं?", ["नमस्ते।", "आप कैसे हैं?"]),
        ("你好。今天天气很好！", ["你好。", "今天天气很好！"]),
        ("مرحبا؟ كيف حالك", ["مرحبا؟", "كيف حالك"]),
        ("It is 3.5 km. Go now", ["It is 3.5 km.", "Go now"]),
    ],
)
def test_sentences_split_on_native_punctuation(text: str, chunks: list[str]) -> None:
    assert sentence_chunks_from_tokens([text]) == chunks


def test_long_unspaced_sentence_is_capped_by_characters() -> None:
    text = "一二三四五六七八九十一二三四五六七八九十"  # 20 characters, no punctuation

    chunks = sentence_chunks_from_tokens([text], max_tokens=4)

    assert chunks == ["一二三四五六七八", "九十一二三四五六", "七八九十"]
    assert "".join(chunks) == text


def test_long_spaced_sentence_is_still_capped_by_words() -> None:
    chunks = sentence_chunks_from_tokens(["one  two three four five"], max_tokens=2)
    assert chunks == ["one two", "three four", "five"]


def test_other_languages_keep_symbols_for_their_own_voice() -> None:
    raw = "**आज** 21°C और 40% नमी है 😀 https://x.io"
    assert normalize_for_speech(raw, "hi") == "आज 21°C और 40% नमी है"
    # English is unchanged.
    assert normalize_for_speech("It is 21°C", "en") == "It is 21 degrees Celsius"
    assert normalize_for_speech("It is 21°C") == "It is 21 degrees Celsius"


def test_line_break_after_native_full_stop_adds_no_extra_pause() -> None:
    assert normalize_for_speech("你好。\n再见。", "zh") == "你好。 再见。"


# ----- speech recognition ------------------------------------------------


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text
        self.no_speech_prob = 0.01
        self.avg_logprob = -0.1


class FakeInfo:
    def __init__(self, language: str, probs: list[tuple[str, float]]) -> None:
        self.language = language
        self.language_probability = probs[0][1] if probs else 0.0
        self.all_language_probs = probs


class DetectingWhisper:
    """Detects `detected` when no language is given; otherwise decodes as asked."""

    def __init__(self, detected: str, probs: list[tuple[str, float]] | None = None) -> None:
        self.detected = detected
        self.probs = probs or [(detected, 0.9)]
        self.languages: list[str | None] = []

    def transcribe(self, audio, language=None, **kwargs):
        self.languages.append(language)
        spoken = language or self.detected
        return iter([FakeSegment(f"text in {spoken}")]), FakeInfo(spoken, self.probs)


def test_whisper_auto_detects_and_reports_the_language() -> None:
    model = DetectingWhisper("ta")
    rec = load_recognizer("whisper", "large-v3-turbo", 16_000, language="auto", shared=model)

    assert rec.final_result(SPEECH) == ("text in ta", pytest.approx(0.905, abs=0.01))
    assert rec.last_language == "ta"
    assert model.languages == [None]


def test_whisper_detection_outside_the_allowlist_is_decoded_again() -> None:
    model = DetectingWhisper("ml", probs=[("ml", 0.5), ("ta", 0.3), ("hi", 0.1)])
    rec = load_recognizer("whisper", "small", 16_000, language="auto", shared=model, allowed_languages=("en", "ta", "hi"))

    text, _ = rec.final_result(SPEECH)

    assert text == "text in ta"
    assert rec.last_language == "ta"
    assert model.languages == [None, "ta"]


def test_whisper_with_one_allowed_language_skips_detection() -> None:
    model = DetectingWhisper("ml")
    rec = load_recognizer("whisper", "small", 16_000, language="auto", shared=model, allowed_languages=("hi",))

    rec.final_result(SPEECH)

    assert model.languages == ["hi"]


def test_default_whisper_model_is_multilingual_unless_english() -> None:
    assert default_whisper_model("en") == "base.en"
    assert default_whisper_model("auto") == "large-v3-turbo"
    assert default_whisper_model("hi") == "large-v3-turbo"


def _asr(recognizer, language: str) -> StreamingASR:
    vad = VoiceActivityDetector(VADConfig(sample_rate=16_000, frame_ms=FRAME_MS, mode="energy"))
    return StreamingASR(
        sample_rate=16_000, chunk_size=SAMPLES, vad=vad, model_path="",
        endpoint_silence_ms=60, recognizer=recognizer, language=language,
    )


def _final_event(asr: StreamingASR):
    events = []
    for frame in [SPEECH] * 5 + [SILENCE] * 10:
        events.extend(asr.process_frame(frame))
    return next(e for e in events if e.type == "final")


def test_final_transcript_carries_the_detected_language() -> None:
    model = DetectingWhisper("hi")
    rec = load_recognizer("whisper", "small", 16_000, language="auto", shared=model)

    assert _final_event(_asr(rec, "auto")).language == "hi"


def test_final_transcript_carries_a_fixed_language() -> None:
    class Plain:
        streaming = False

        def accept_waveform(self, frame): pass
        def partial_result(self): return "", 0.0
        def final_result(self, utterance): return "bonjour", 0.9
        def reset(self): pass

    assert _final_event(_asr(Plain(), "fr")).language == "fr"


# ----- voices ------------------------------------------------------------


def _voice(folder: Path, name: str, rate: int = 22_050, family: str | None = None) -> Path:
    onnx = folder / f"{name}.onnx"
    onnx.write_bytes(b"onnx")
    config: dict = {"audio": {"sample_rate": rate}}
    if family:
        config["language"] = {"family": family, "code": f"{family}_XX"}
    Path(f"{onnx}.json").write_text(json.dumps(config), encoding="utf-8")
    return onnx


def test_voices_are_found_by_language(tmp_path: Path) -> None:
    en = _voice(tmp_path, "en_US-lessac-medium")
    hi = _voice(tmp_path, "hi_IN-pratham-medium", rate=16_000)
    odd = _voice(tmp_path, "custom-voice", family="ta")
    (tmp_path / "fr_FR-no-config.onnx").write_bytes(b"onnx")  # no .json: ignored

    assert voice_language(odd) == "ta"
    assert discover_voices(tmp_path) == {"en": en.resolve(), "hi": hi.resolve(), "ta": odd.resolve()}
    assert discover_voices(tmp_path / "missing") == {}


def test_router_prefers_the_default_voice_and_reports_gaps(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    voices.mkdir()
    folder_en = _voice(voices, "en_GB-alan-medium")
    hi = _voice(voices, "hi_IN-pratham-medium")
    default = _voice(tmp_path, "en_US-lessac-medium")

    router = VoiceRouter(default, voices)

    assert router.voice_for(None) == default
    assert router.voice_for("en") == default != folder_en
    assert router.voice_for("hi_IN") == hi.resolve()
    assert router.voice_for("ta") is None
    assert router.languages == ["en", "hi"]

    # Without PIPER_VOICE, the folder's English voice is the default.
    assert VoiceRouter(None, voices).voice_for(None) == folder_en.resolve()


def test_router_with_an_unidentified_voice_speaks_everything(tmp_path: Path) -> None:
    voice = tmp_path / "myvoice.onnx"
    router = VoiceRouter(voice)
    assert router.voice_for("ta") == voice


class RecordingSynth:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[str] = []

    def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        return b"\x01\x00" * 100

    def close(self) -> None:
        pass


async def _speak(tts: PiperStreamingTTS, sentences: list[tuple[str, str | None]]) -> None:
    await tts.start()
    for text, language in sentences:
        assert await tts.synthesize_sentence(text, language=language)
    await asyncio.wait_for(tts.flush(), 2)
    await tts.stop()


def _drain(q: AudioChunkQueue) -> list:
    chunks = []
    while not q.empty():
        chunks.append(q._q.get_nowait())
    return chunks


async def test_each_language_is_spoken_by_its_own_voice(tmp_path: Path, monkeypatch) -> None:
    default = _voice(tmp_path, "en_US-lessac-medium")
    voices = tmp_path / "voices"
    voices.mkdir()
    _voice(voices, "hi_IN-pratham-medium", rate=16_000)
    synths: dict[str, RecordingSynth] = {}

    def create(config: PiperConfig) -> RecordingSynth:
        return synths.setdefault(Path(config.voice_path).name, RecordingSynth(Path(config.voice_path).name))

    monkeypatch.setattr("voice_assistant.tts.stream.create_synthesizer", create)
    q = AudioChunkQueue(maxsize=64)
    tts = PiperStreamingTTS(PiperConfig(default, voices_dir=voices, fallback="none"), q)

    await _speak(tts, [("Hello.", "en"), ("नमस्ते।", "hi"), ("வணக்கம்.", "ta"), ("Bye.", None)])

    assert synths["en_US-lessac-medium.onnx"].calls == ["Hello.", "Bye."]
    assert synths["hi_IN-pratham-medium.onnx"].calls == ["नमस्ते।"]
    rates = {chunk.debug_text: chunk.sample_rate for chunk in _drain(q)}
    assert rates == {"Hello.": 22_050, "नमस्ते।": 16_000, "Bye.": 22_050}  # Tamil: no voice, silent


async def test_language_without_voice_can_use_the_default_voice(tmp_path: Path, monkeypatch) -> None:
    default = _voice(tmp_path, "en_US-lessac-medium")
    synth = RecordingSynth("default")
    monkeypatch.setattr("voice_assistant.tts.stream.create_synthesizer", lambda config: synth)
    tts = PiperStreamingTTS(PiperConfig(default, fallback="default"), AudioChunkQueue(maxsize=64))

    await _speak(tts, [("Hola.", "es")])

    assert synth.calls == ["Hola."]


async def test_language_without_voice_uses_espeak(tmp_path: Path, monkeypatch) -> None:
    default = _voice(tmp_path, "en_US-lessac-medium")
    monkeypatch.setattr("voice_assistant.tts.stream.create_synthesizer", lambda config: RecordingSynth("default"))
    commands: list[list[str]] = []

    class Done:
        returncode = 0
        stderr = b""
        stdout = _wav(b"\x02\x00" * 50, rate=16_000, size=0xFFFFFFFF)

    def run(cmd, **kwargs):
        commands.append(cmd)
        return Done()

    monkeypatch.setattr("voice_assistant.tts.stream.subprocess.run", run)
    q = AudioChunkQueue(maxsize=64)
    tts = PiperStreamingTTS(PiperConfig(default, fallback="espeak"), q)

    await _speak(tts, [("你好。", "zh")])

    assert commands == [["espeak-ng", "-v", "cmn", "-b", "1", "--stdout", "--", "你好。"]]
    [chunk] = _drain(q)
    assert chunk.sample_rate == 16_000
    assert chunk.pcm16 == b"\x02\x00" * 50


def _wav(pcm: bytes, rate: int, size: int | None = None) -> bytes:
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    data_size = len(pcm) if size is None else size
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", data_size) + pcm
    )


def test_wav_parsing() -> None:
    assert _wav_to_pcm(_wav(b"\x01\x00\x02\x00", 22_050)) == (b"\x01\x00\x02\x00", 22_050)
    with pytest.raises(RuntimeError):
        _wav_to_pcm(b"not a wav file")


# ----- pipeline ----------------------------------------------------------


class RecordingLLM(MockLLMClient):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def stream_tokens(self, messages, out_queue, bench=None) -> str:
        self.prompts.append(messages[-1]["content"])
        return await super().stream_tokens(messages, out_queue, bench)


class RecordingTTS:
    sample_rate = 22_050

    def __init__(self) -> None:
        self.playback_queue = AudioChunkQueue(maxsize=8)
        self.spoken: list[tuple[str, str | None]] = []

    async def start(self) -> None: pass
    async def stop(self) -> None: pass
    def cancel_pending(self) -> None: pass
    async def flush(self) -> None: pass

    async def synthesize_sentence(self, sentence: str, language: str | None = None) -> bool:
        self.spoken.append((sentence, language))
        return True


async def _chat(lines: list[str], monkeypatch, language: str) -> tuple[RecordingLLM, RecordingTTS]:
    for name in ("CONVERSATION_MEMORY_PATH", "ENABLE_WEATHER", "ASR_LANGUAGES", "ASR_MODEL_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ASR_LANGUAGE", language)
    it = iter(lines)

    def read_line(_prompt: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    llm, tts = RecordingLLM(), RecordingTTS()
    orchestrator, _ = build_chat(Settings(), llm, read_line=read_line, out=io.StringIO(), tts=tts)
    await asyncio.wait_for(orchestrator.run(), 5)
    return llm, tts


async def test_reply_follows_the_language_the_user_wrote_in(monkeypatch) -> None:
    llm, tts = await _chat(["வணக்கம், எப்படி இருக்கிறீர்கள்?"], monkeypatch, language="auto")

    assert llm.prompts == ["வணக்கம், எப்படி இருக்கிறீர்கள்?\n\n(Reply in Tamil.)"]
    assert tts.spoken and {language for _, language in tts.spoken} == {"ta"}


async def test_english_setup_sends_prompts_unchanged(monkeypatch) -> None:
    llm, tts = await _chat(["tell me something"], monkeypatch, language="en")

    assert llm.prompts == ["tell me something"]
    assert {language for _, language in tts.spoken} == {"en"}


def test_actions_leave_other_languages_to_the_llm() -> None:
    actions = BasicIntentActions()
    time_intent = {"intent": "time", "confidence": 0.9}

    assert actions.handle("what time is it", time_intent).handled
    assert actions.handle("what time is it", {**time_intent, "spoken_language": "en"}).handled
    assert actions.handle("time batao", {**time_intent, "spoken_language": "hi"}).handled
    assert not actions.handle("quelle heure", {**time_intent, "spoken_language": "fr"}).handled


# ----- configuration and setup -------------------------------------------


@pytest.mark.parametrize(
    ("env", "problem"),
    [
        ({"ASR_LANGUAGE": "klingon"}, "ASR_LANGUAGE must be"),
        ({"ASR_LANGUAGE": "auto", "ASR_LANGUAGES": "hi,xx"}, "unknown language codes: xx"),
        ({"ASR_LANGUAGE": "auto", "ASR_BACKEND": "vosk"}, "needs ASR_BACKEND=whisper"),
        ({"ASR_LANGUAGE": "auto", "ASR_MODEL_PATH": "models/whisper-base.en"}, "English-only Whisper model"),
        ({"TTS_FALLBACK": "loud"}, "TTS_FALLBACK must be"),
    ],
)
def test_language_settings_are_checked(monkeypatch, env: dict[str, str], problem: str) -> None:
    for name in ("ASR_LANGUAGE", "ASR_LANGUAGES", "ASR_BACKEND", "ASR_MODEL_PATH", "TTS_FALLBACK"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert any(problem in p for p in Settings().check_ranges())


def test_multilingual_settings(monkeypatch) -> None:
    monkeypatch.setenv("ASR_LANGUAGE", "auto")
    monkeypatch.setenv("ASR_LANGUAGES", "en, hi,ta")
    monkeypatch.delenv("ASR_MODEL_PATH", raising=False)
    monkeypatch.delenv("ASR_BACKEND", raising=False)
    settings = Settings()

    assert settings.check_ranges() == []
    assert settings.multilingual
    assert settings.asr_languages == ("en", "hi", "ta")
    assert settings.whisper_model() == "large-v3-turbo"


VOICES_INDEX = {
    "hi_IN-pratham-medium": {
        "language": {"family": "hi", "code": "hi_IN"}, "quality": "medium", "num_speakers": 1,
        "files": {
            "hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx": {},
            "hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx.json": {},
            "hi/hi_IN/pratham/medium/MODEL_CARD": {},
        },
    },
    "hi_IN-crowd-high": {
        "language": {"family": "hi"}, "quality": "high", "num_speakers": 1,
        "files": {"hi/a/high.onnx": {}, "hi/a/high.onnx.json": {}},
    },
    "pt_PT-tugão-medium": {
        "language": {"family": "pt"}, "quality": "medium", "num_speakers": 3,
        "files": {"pt/pt_PT/tugão/medium/pt_PT-tugão-medium.onnx": {}, "pt/pt_PT/tugão/medium/pt_PT-tugão-medium.onnx.json": {}},
    },
}


def test_piper_voices_are_picked_per_language() -> None:
    chosen, missing = pick_piper_voices(VOICES_INDEX, ["hi", "pt", "ta"])

    assert chosen["hi"] == [
        "hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx",
        "hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx.json",
    ]
    assert chosen["pt"][0].endswith("pt_PT-tugão-medium.onnx")
    assert missing == ["ta"]


def test_language_voices_download_without_network(tmp_path: Path) -> None:
    fetched: list[tuple[str, Path]] = []

    def download(url: str, target: Path) -> None:
        fetched.append((url, target))
        target.write_bytes(b"voice")

    messages = download_language_voices(tmp_path, ["hi", "pt", "ta"], fetch_index=lambda url: VOICES_INDEX, download=download)

    names = sorted(target.name for _, target in fetched)
    assert names == sorted([
        "hi_IN-pratham-medium.onnx", "hi_IN-pratham-medium.onnx.json",
        "pt_PT-tugão-medium.onnx", "pt_PT-tugão-medium.onnx.json",
    ])
    assert all(target.parent == tmp_path / "voices" for _, target in fetched)
    assert any("tug%C3%A3o" in url for url, _ in fetched)
    assert any("no Piper voice for Tamil" in m for m in messages)

    # Already downloaded voices are skipped.
    fetched.clear()
    download_language_voices(tmp_path, ["hi"], fetch_index=lambda url: VOICES_INDEX, download=download)
    assert fetched == []


def test_multilingual_model_plan_and_env() -> None:
    assert [a.target for a in recommended_models()] == [a.target for a in recommended_models(["en"])]
    assert "whisper-large-v3-turbo" in [a.target for a in recommended_models(["hi"])]

    text = env_template(Path("models"), ["hi", "ta"])
    assert 'ASR_MODEL_PATH="models/whisper-large-v3-turbo"' in text
    assert 'ASR_LANGUAGE="auto"' in text
    assert 'ASR_LANGUAGES="en,hi,ta"' in text
    assert 'PIPER_VOICES_DIR="models/voices"' in text
    assert "PIPER_VOICES_DIR" not in env_template(Path("models"))

    plan = format_model_plan(Path("models"), ["hi", "ta"])
    assert "Hindi, Tamil" in plan


def test_voice_language_from_espeak_config(tmp_path: Path) -> None:
    voice = tmp_path / "old.onnx"
    Path(f"{voice}.json").write_text(json.dumps({"espeak": {"voice": "cmn"}}), encoding="utf-8")
    assert voice_language(voice) == "zh"
