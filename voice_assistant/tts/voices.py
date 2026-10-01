"""Find Piper voices on disk and pick one for the language of a reply."""
from __future__ import annotations

import json
import re
from pathlib import Path

from voice_assistant.lang import base_language

_FILE_LANGUAGE = re.compile(r"^([a-z]{2,3})(?:[_-]|$)", re.I)
# espeak-ng voice names that differ from the language code.
_ESPEAK_LANGUAGES = {"cmn": "zh", "nb": "no", "jv": "jw"}


def voice_language(voice_path: Path | str) -> str | None:
    """Language of a Piper voice, from its config or else its file name.

    Newer voice configs have a "language" block; older ones only name their
    espeak-ng phonemizer voice ("en-us", "cmn").

    Piper voice files are named "<lang>_<REGION>-<name>-<quality>.onnx", for
    example "hi_IN-pratham-medium.onnx".
    """
    path = Path(voice_path)
    try:
        data = json.loads(Path(f"{path}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    language = data.get("language") if isinstance(data, dict) else None
    if isinstance(language, dict):
        code = base_language(str(language.get("family") or language.get("code") or ""))
        if code:
            return code
    espeak = data.get("espeak") if isinstance(data, dict) else None
    if isinstance(espeak, dict):
        code = base_language(str(espeak.get("voice") or ""))
        if code:
            return _ESPEAK_LANGUAGES.get(code, code)
    match = _FILE_LANGUAGE.match(path.name)
    return match.group(1).lower() if match else None


def discover_voices(voices_dir: Path | None) -> dict[str, Path]:
    """Language -> voice for every Piper voice (with its .json config) in `voices_dir`.

    When a folder has several voices for one language, the first by name wins.
    """
    if voices_dir is None or not voices_dir.is_dir():
        return {}
    voices: dict[str, Path] = {}
    for onnx in sorted(voices_dir.glob("*.onnx")):
        if not Path(f"{onnx}.json").exists():
            continue
        language = voice_language(onnx)
        if language:
            voices.setdefault(language, onnx.resolve())
    return voices


class VoiceRouter:
    """Maps a reply's language to a Piper voice.

    `default_voice` (PIPER_VOICE) speaks when the language is unknown and
    takes precedence for its own language. Returns None when no voice covers a
    language, so the caller can fall back.
    """

    def __init__(self, default_voice: Path | None, voices_dir: Path | None = None) -> None:
        self.voices = discover_voices(voices_dir)
        self.default_language: str | None = None
        if default_voice is not None:
            self.default_voice: Path | None = Path(default_voice)
            self.default_language = voice_language(default_voice)
            if self.default_language:
                self.voices[self.default_language] = self.default_voice
        else:
            # No PIPER_VOICE: English if the folder has it, else the first voice.
            self.default_language = "en" if "en" in self.voices else next(iter(sorted(self.voices)), None)
            self.default_voice = self.voices.get(self.default_language) if self.default_language else None

    @property
    def languages(self) -> list[str]:
        return sorted(self.voices)

    def voice_for(self, language: str | None) -> Path | None:
        base = base_language(language)
        if base is None:
            return self.default_voice
        voice = self.voices.get(base)
        if voice is not None:
            return voice
        # A lone voice of unknown language (an oddly named PIPER_VOICE) keeps
        # speaking everything, as before languages were tracked.
        if self.default_language is None and len(self.voices) == 0:
            return self.default_voice
        return None
