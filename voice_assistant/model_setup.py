from __future__ import annotations

import json
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from voice_assistant.lang import base_language, language_name


@dataclass(frozen=True, slots=True)
class ModelAsset:
    name: str
    url: str
    target: str
    license: str
    source: str
    extract_to: str | None = None


RECOMMENDED_MODELS = [
    ModelAsset(
        name="LLM: Qwen2.5 0.5B Instruct GGUF Q4_K_M",
        url="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        target="Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        license="Apache-2.0",
        source="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    ),
    ModelAsset(
        name="TTS: Piper en_US lessac medium",
        url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        target="en_US-lessac-medium.onnx",
        license="MIT",
        source="rhasspy/piper-voices",
    ),
    ModelAsset(
        name="TTS config: Piper en_US lessac medium",
        url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        target="en_US-lessac-medium.onnx.json",
        license="MIT",
        source="rhasspy/piper-voices",
    ),
    ModelAsset(
        name="ASR: Whisper base.en (faster-whisper)",
        url="",  # fetched with faster_whisper.download_model
        target="whisper-base.en",
        license="MIT",
        source="Systran/faster-whisper-base.en",
    ),
]


# Replaces Whisper base.en when other languages are wanted.
MULTILINGUAL_WHISPER = ModelAsset(
    name="ASR: Whisper large-v3-turbo (faster-whisper, 99 languages)",
    url="",  # fetched with faster_whisper.download_model
    target="whisper-large-v3-turbo",
    license="MIT",
    source="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
)

PIPER_VOICES_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
VOICES_SUBDIR = "voices"
# Medium voices balance quality and speed; single-speaker voices need no speaker id.
_QUALITY_RANK = {"medium": 0, "high": 1, "low": 2, "x_low": 3}


def is_multilingual(languages: Sequence[str]) -> bool:
    return any(code != "en" for code in languages)


def recommended_models(languages: Sequence[str] = ()) -> list[ModelAsset]:
    if not is_multilingual(languages):
        return list(RECOMMENDED_MODELS)
    return [MULTILINGUAL_WHISPER if a.target == "whisper-base.en" else a for a in RECOMMENDED_MODELS]


def env_template(models_dir: Path, languages: Sequence[str] = ()) -> str:
    models = models_dir.as_posix()
    multilingual = is_multilingual(languages)
    asr_model = MULTILINGUAL_WHISPER.target if multilingual else "whisper-base.en"
    lines = [
        f'MODEL_PATH="{models}/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"',
        'DRAFT_MODEL_PATH=""',
        f'PIPER_VOICE="{models}/en_US-lessac-medium.onnx"',
        f'ASR_MODEL_PATH="{models}/{asr_model}"',
        'ASR_BACKEND="whisper"',
        "VAD_AGGRESSIVENESS=2",
        "CHUNK_MS=20",
        "ASR_ENDPOINT_SILENCE_MS=300",
        "ENABLE_BARGE_IN=1",
        "BARGE_IN_MS=240",
        "ACK_TONE_MS=55",
        "ENABLE_ACK_TONE=1",
        'VAANI_PROFILE="low_latency"',
        'ASSISTANT_SYSTEM_PROMPT="You are Vaani, a concise voice assistant. Your replies are spoken aloud, so answer in one or two short plain sentences unless the user asks for detail, and never use markdown, lists, code blocks or emoji."',
        "TTS_SENTENCE_MAX_TOKENS=8",
        "TTS_EAGER_MIN_WORDS=3",
        "PLAYER_BLOCKSIZE=128",
        "GRPC_PORT=50051",
        'CONVERSATION_MEMORY_PATH="data/session.jsonl"',
    ]
    if multilingual:
        lines += [
            f'PIPER_VOICES_DIR="{models}/{VOICES_SUBDIR}"',
            'TTS_FALLBACK="auto"',
        ]
        # English stays: the default voice speaks it.
        spoken = dict.fromkeys(["en", *languages])
        lines += ['ASR_LANGUAGE="auto"', f'ASR_LANGUAGES="{",".join(spoken)}"']
    return "\n".join(lines) + "\n"


def write_env_file(path: Path, models_dir: Path, overwrite: bool = False, languages: Sequence[str] = ()) -> bool:
    if path.exists() and not overwrite:
        return False

    path.write_text(env_template(models_dir, languages), encoding="utf-8")
    return True


def format_model_plan(models_dir: Path, languages: Sequence[str] = ()) -> str:
    lines = ["Recommended open model set", f"Target directory: {models_dir}"]
    for asset in recommended_models(languages):
        action = "download and extract" if asset.extract_to else "download"
        lines.append(f"- {asset.name}")
        lines.append(f"  {action}: {asset.target}")
        lines.append(f"  license: {asset.license}")
        lines.append(f"  source: {asset.source}")
    if is_multilingual(languages):
        names = ", ".join(language_name(code) or code for code in languages)
        lines.append(f"- TTS: one Piper voice per language ({names})")
        lines.append(f"  download: {VOICES_SUBDIR}/ (chosen from rhasspy/piper-voices voices.json)")
        lines.append("  license: varies per voice, see each voice's MODEL_CARD")
        lines.append("  source: rhasspy/piper-voices")
        lines.append(
            "Note: the 0.5B LLM answers poorly outside English. For other languages point "
            "MODEL_PATH at a bigger multilingual GGUF (Qwen2.5 7B, Gemma 3) or use LLM_BACKEND=openai."
        )
    return "\n".join(lines)


def pick_piper_voices(index: dict, languages: Sequence[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Choose one voice per language from Piper's voices.json.

    Returns language -> the voice's .onnx and .onnx.json paths in the
    rhasspy/piper-voices repo, and the languages Piper has no voice for.
    """
    candidates: dict[str, list[tuple[tuple, list[str]]]] = {}
    for key, voice in index.items():
        if not isinstance(voice, dict):
            continue
        lang = voice.get("language") or {}
        code = base_language(str(lang.get("family") or lang.get("code") or ""))
        if code not in languages:
            continue
        files = sorted(f for f in voice.get("files", {}) if f.endswith((".onnx", ".onnx.json")))
        if len(files) != 2:
            continue
        rank = (_QUALITY_RANK.get(str(voice.get("quality")), 9), int(voice.get("num_speakers", 1)) != 1, key)
        candidates.setdefault(code, []).append((rank, files))
    chosen = {code: min(options)[1] for code, options in candidates.items()}
    return chosen, [code for code in languages if code not in chosen]


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_language_voices(
    models_dir: Path,
    languages: Sequence[str],
    fetch_index: Callable[[str], dict] = _fetch_json,
    download: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Download a Piper voice for each language into models_dir/voices."""
    download = download or _download
    voices_dir = models_dir / VOICES_SUBDIR
    voices_dir.mkdir(parents=True, exist_ok=True)
    chosen, missing = pick_piper_voices(fetch_index(PIPER_VOICES_URL + "voices.json"), languages)
    messages: list[str] = []
    for code, files in sorted(chosen.items()):
        for repo_path in files:
            target = voices_dir / Path(repo_path).name
            if target.exists():
                messages.append(f"skip {code} voice: {target} already exists")
                continue
            print(f"downloading {code} voice {target.name} ...", file=sys.stderr)
            download(PIPER_VOICES_URL + urllib.parse.quote(repo_path), target)
            messages.append(f"downloaded {code} voice: {target}")
    for code in missing:
        messages.append(
            f"no Piper voice for {language_name(code) or code}: install espeak-ng to speak it (TTS_FALLBACK=auto)"
        )
    return messages


def _download(url: str, target: Path, show_progress: bool = True) -> None:
    """Download to a temporary `.part` file so an interrupted download is never
    mistaken for a finished one on the next run."""
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as fh:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            fh.write(block)
            done += len(block)
            if show_progress and total and sys.stderr.isatty():
                print(f"\r  {target.name}: {done * 100 // total}% of {total >> 20} MB", end="", file=sys.stderr)
    if show_progress and total and sys.stderr.isatty():
        print(file=sys.stderr)
    partial.replace(target)


def _download_whisper(asset: ModelAsset, target: Path) -> str:
    try:
        from faster_whisper import download_model  # type: ignore
    except ImportError:
        return f"skip {asset.name}: faster-whisper is not installed (pip install -e '.[local]')"
    size = asset.target.removeprefix("whisper-")
    partial = target.with_name(target.name + ".part")
    shutil.rmtree(partial, ignore_errors=True)
    download_model(size, output_dir=str(partial))
    partial.replace(target)
    return f"downloaded {asset.name}: {target}"


def download_recommended_models(models_dir: Path, languages: Sequence[str] = ()) -> list[str]:
    models_dir.mkdir(parents=True, exist_ok=True)
    messages: list[str] = []

    for asset in recommended_models(languages):
        target = models_dir / asset.target
        extract_to = models_dir / asset.extract_to if asset.extract_to else None

        if extract_to and extract_to.exists():
            messages.append(f"skip {asset.name}: {extract_to} already exists")
            continue
        if target.exists() and not extract_to:
            messages.append(f"skip {asset.name}: {target} already exists")
            continue

        print(f"downloading {asset.name} ...", file=sys.stderr)
        if not asset.url:
            messages.append(_download_whisper(asset, target))
            continue
        _download(asset.url, target)
        messages.append(f"downloaded {asset.name}: {target}")

        if extract_to:
            staging = models_dir / f".{asset.extract_to}.extracting"
            shutil.rmtree(staging, ignore_errors=True)
            with zipfile.ZipFile(target) as archive:
                archive.extractall(staging)
            extracted = staging / asset.extract_to
            (extracted if extracted.exists() else staging).replace(extract_to)
            shutil.rmtree(staging, ignore_errors=True)
            target.unlink()
            messages.append(f"extracted {asset.name}: {extract_to}")

    if is_multilingual(languages):
        messages.extend(download_language_voices(models_dir, [c for c in languages if c != "en"]))
    return messages
