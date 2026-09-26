"""Turn LLM output into text that sounds right when spoken.

Chat models format replies for screens: markdown emphasis, bullet lists, code,
links and emoji. Piper reads those characters literally ("asterisk asterisk"),
so they are removed or replaced with words just before synthesis.
"""
from __future__ import annotations

import re

_CODE_BLOCK = re.compile(r"```.*?(?:```|$)", re.S)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]*)\)")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.I)
_LINE_MARKER = re.compile(r"^\s*(?:#{1,6}\s+|>\s*|[-*+•]\s+|\d+[.)]\s+)", re.M)
_EMPHASIS = re.compile(r"(\*{1,3}|_{2,3}|~~)(?=\S)(.+?)(?<=\S)\1")
_STRAY_MARKUP = re.compile(r"[*#`~|<>\[\]{}]|_{2,}")
_TEMPERATURE = re.compile(r"(-?\d+(?:\.\d+)?)\s*°\s*([CF])\b")
_DEGREES = re.compile(r"(\d)\s*°")
_PERCENT = re.compile(r"(\d)\s*%")
_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\s*\n\s*")

# Emoji and pictographs: dropped, they have no sensible spoken form.
_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"  # pictographs, emoticons, transport, symbols
    "☀-➿"  # misc symbols and dingbats
    "⬀-⯿"  # arrows and stars
    "️‍"  # variation selector, zero-width joiner
    "]+"
)

_SYMBOL_WORDS = {
    "&": " and ",
    "=": " equals ",
    "→": " to ",
    "≈": " about ",
    "×": " times ",
}

_CURRENCIES = {"$": "dollars", "€": "euros", "£": "pounds", "₹": "rupees"}
_MONEY = re.compile(r"([$€£₹])\s?(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)")

_UNITS = {"C": "Celsius", "F": "Fahrenheit"}


def normalize_for_speech(text: str) -> str:
    """Return `text` with visual formatting removed, ready for TTS."""
    if not text:
        return ""

    text = _CODE_BLOCK.sub(" ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("a link", text)
    text = _LINE_MARKER.sub("", text)
    text = _EMPHASIS.sub(r"\2", text)
    text = _EMOJI.sub(" ", text)

    text = _TEMPERATURE.sub(lambda m: f"{m.group(1)} degrees {_UNITS[m.group(2)]}", text)
    text = _DEGREES.sub(r"\1 degrees", text)
    text = _PERCENT.sub(r"\1 percent", text)
    for symbol, words in _SYMBOL_WORDS.items():
        text = text.replace(symbol, words)
    text = _MONEY.sub(lambda m: f"{m.group(2)} {_CURRENCIES[m.group(1)]}", text)

    text = _STRAY_MARKUP.sub(" ", text)
    # Line breaks separate list items and paragraphs: speak them as pauses.
    text = _BLANK_LINES.sub(". ", text.strip())
    text = re.sub(r"([.!?,;:])\s*\.\s", r"\1 ", text)
    text = _WHITESPACE.sub(" ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text.strip().lstrip(".,;: ")
