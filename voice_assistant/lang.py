"""Language helpers shared by the ASR, LLM and TTS stages.

Language codes are Whisper's (ISO 639-1 where one exists), which is also what
Piper voice files use as their prefix ("hi_IN-...", "ta_IN-...").
"""
from __future__ import annotations

import re

# Every language Whisper can transcribe, with the English name used in prompts.
LANGUAGE_NAMES: dict[str, str] = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese", "az": "Azerbaijani",
    "ba": "Bashkir", "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali", "bo": "Tibetan",
    "br": "Breton", "bs": "Bosnian", "ca": "Catalan", "cs": "Czech", "cy": "Welsh",
    "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish", "fo": "Faroese",
    "fr": "French", "gl": "Galician", "gu": "Gujarati", "ha": "Hausa", "haw": "Hawaiian",
    "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian",
    "hy": "Armenian", "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese",
    "jw": "Javanese", "ka": "Georgian", "kk": "Kazakh", "km": "Khmer", "kn": "Kannada",
    "ko": "Korean", "la": "Latin", "lb": "Luxembourgish", "ln": "Lingala", "lo": "Lao",
    "lt": "Lithuanian", "lv": "Latvian", "mg": "Malagasy", "mi": "Maori", "mk": "Macedonian",
    "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi", "ms": "Malay", "mt": "Maltese",
    "my": "Burmese", "ne": "Nepali", "nl": "Dutch", "nn": "Norwegian Nynorsk", "no": "Norwegian",
    "oc": "Occitan", "pa": "Punjabi", "pl": "Polish", "ps": "Pashto", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sa": "Sanskrit", "sd": "Sindhi", "si": "Sinhala",
    "sk": "Slovak", "sl": "Slovenian", "sn": "Shona", "so": "Somali", "sq": "Albanian",
    "sr": "Serbian", "su": "Sundanese", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil",
    "te": "Telugu", "tg": "Tajik", "th": "Thai", "tk": "Turkmen", "tl": "Tagalog",
    "tr": "Turkish", "tt": "Tatar", "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek",
    "vi": "Vietnamese", "yi": "Yiddish", "yo": "Yoruba", "yue": "Cantonese", "zh": "Chinese",
}

AUTO = "auto"

# Sentence-final punctuation. ASCII marks only end a sentence when followed by
# whitespace ("3.5" and "e.g." stay whole); the others end it outright.
SENTENCE_END_CHARS = ".!?。！？｡।॥؟۔։።။។"
# Clause punctuation where an early TTS chunk may be cut.
CLAUSE_CHARS = ",;:，、；：،"

# Chinese and Japanese are written without spaces between words, so word counts
# fall back to characters: two characters count as one word.
_NO_SPACE = "　-〿぀-ヿ㐀-䶿一-鿿豈-﫿＀-￯"
_WORD_UNIT = re.compile(rf"[{_NO_SPACE}]{{1,2}}|[^\s{_NO_SPACE}]+")

# Scripts that identify one language well enough to pick a voice from text
# alone (typed chat). Latin, Cyrillic and Arabic are shared by many languages.
_SCRIPTS: list[tuple[str, str]] = [
    ("぀-ヿ", "ja"),  # kana before Han: Japanese mixes both
    ("가-힯ᄀ-ᇿ㄰-㆏", "ko"),
    ("一-鿿㐀-䶿", "zh"),
    ("ऀ-ॿ", "hi"),
    ("ঀ-৿", "bn"),
    ("਀-੿", "pa"),
    ("઀-૿", "gu"),
    ("஀-௿", "ta"),
    ("ఀ-౿", "te"),
    ("ಀ-೿", "kn"),
    ("ഀ-ൿ", "ml"),
    ("඀-෿", "si"),
    ("฀-๿", "th"),
    ("຀-໿", "lo"),
    ("ༀ-࿿", "bo"),
    ("က-႟", "my"),
    ("Ⴀ-ჿ", "ka"),
    ("ሀ-፿", "am"),
    ("ក-៿", "km"),
    ("Ͱ-Ͽ", "el"),
    ("֐-׿", "he"),
    ("԰-֏", "hy"),
]
_SCRIPT_PATTERNS = [(re.compile(f"[{chars}]"), code) for chars, code in _SCRIPTS]
_LETTER = re.compile(r"[^\W\d_]")


def base_language(code: str | None) -> str | None:
    """'en_US', 'en-us' and 'EN' all become 'en'."""
    if not code:
        return None
    return re.split(r"[_-]", code.strip().lower(), maxsplit=1)[0] or None


def is_english(code: str | None) -> bool:
    """True for English, and for an unknown language (the historical default)."""
    return base_language(code) in {None, "en"}


def language_name(code: str | None) -> str | None:
    base = base_language(code)
    return LANGUAGE_NAMES.get(base) if base else None


def parse_language_list(raw: str) -> tuple[str, ...]:
    """'hi, ta,EN' -> ('hi', 'ta', 'en'), without duplicates."""
    seen: dict[str, None] = {}
    for part in raw.split(","):
        base = base_language(part)
        if base:
            seen.setdefault(base, None)
    return tuple(seen)


def guess_language(text: str) -> str | None:
    """Language implied by the writing system, or None when it is ambiguous."""
    letters = len(_LETTER.findall(text))
    if not letters:
        return None
    best, best_count = None, 0
    for pattern, code in _SCRIPT_PATTERNS:
        count = len(pattern.findall(text))
        if count > best_count:
            best, best_count = code, count
    # Japanese text is mostly Han characters with some kana.
    if best == "zh" and _SCRIPT_PATTERNS[0][0].search(text):
        best = "ja"
    return best if best_count * 2 >= letters else None


def word_spans(text: str) -> list[tuple[int, int]]:
    """Spans of the words in `text`, counting Chinese/Japanese by characters."""
    return [m.span() for m in _WORD_UNIT.finditer(text)]


def count_words(text: str) -> int:
    return len(_WORD_UNIT.findall(text))


def ends_sentence(text: str) -> bool:
    return text.rstrip().endswith(tuple(SENTENCE_END_CHARS))


def with_reply_language(messages: list[dict[str, str]], language: str | None) -> list[dict[str, str]]:
    """Copy of `messages` asking for the reply in `language`.

    The request is added to the last user message rather than the system
    prompt, so the cached system-prompt prefix is reused and chat templates
    that reject a second system message still work.
    """
    name = language_name(language)
    if not name:
        return messages
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i] = {**out[i], "content": f"{out[i]['content']}\n\n(Reply in {name}.)"}
            break
    return out
