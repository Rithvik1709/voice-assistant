"""
WER Benchmark Script for Voice Assistant STT Module

Evaluates Word Error Rate of the STT pipeline on a test dataset.
Supports English, Hindi, and Telugu.

Usage:
    python benchmarks/run_wer.py --lang en --data data/sample_tests/en_sample.csv
    python benchmarks/run_wer.py --lang hi --data data/sample_tests/hi_sample.csv
    python benchmarks/run_wer.py --lang en --data data/sample_tests/en_sample.csv --output results.csv
    python benchmarks/run_wer.py --lang en --data data/sample_tests/en_sample.csv --backend whispercpp --model models/ggml-base.bin

References:
    - jiwer: https://github.com/jitsi/jiwer
    - Issue #17: https://github.com/Rithvik1709/voice-assistant/issues/17
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import wave
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── dependency check ────────────────────────────────────────────────────────
try:
    from jiwer import wer as compute_wer
except ImportError:
    logger.error("jiwer not installed. Run: pip install jiwer")
    sys.exit(1)

# ── constants ────────────────────────────────────────────────────────────────
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
}

SAMPLE_RATE = 16_000


# ── audio helpers ────────────────────────────────────────────────────────────

def read_wav_pcm(audio_path: str) -> bytes:
    """Read a WAV file and return raw PCM bytes (mono, 16-bit, 16kHz)."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != SAMPLE_RATE:
            raise RuntimeError(
                f"Invalid WAV format in {audio_path}. "
                f"Expected mono, 16-bit, {SAMPLE_RATE}Hz."
            )
        n_frames = wf.getnframes()
        pcm = wf.readframes(n_frames)
    return pcm


# ── STT backends ─────────────────────────────────────────────────────────────




def load_vosk_model(model_path: str):
    """Load Vosk model once and return it."""
    try:
        from vosk import Model
    except ImportError:
        raise RuntimeError("vosk not installed. Run: pip install vosk")
    return Model(model_path)


def load_whispercpp_model(model_path: str):
    """Load WhisperCpp model once and return it."""
    try:
        from whisper_cpp_python import Whisper
    except ImportError:
        raise RuntimeError(
            "whisper_cpp_python not installed. "
            "Run: pip install whisper-cpp-python"
        )
    return Whisper(model_path)


def transcribe_with_vosk(audio_path: str, model) -> str:
    """Transcribe audio using a pre-loaded Vosk model."""
    from vosk import KaldiRecognizer

    pcm = read_wav_pcm(audio_path)
    rec = KaldiRecognizer(model, SAMPLE_RATE)

    chunk_size = 4000
    for i in range(0, len(pcm), chunk_size):
        rec.AcceptWaveform(pcm[i: i + chunk_size])

    result = json.loads(rec.FinalResult())
    return result.get("text", "").strip().lower()


def transcribe_with_whispercpp(audio_path: str, model) -> str:
    """Transcribe audio using a pre-loaded WhisperCpp model."""
    import numpy as np

    pcm = read_wav_pcm(audio_path)
    audio = np.frombuffer(pcm, dtype=np.int16).astype("float32") / 32768.0
    segments = model.transcribe(audio, beam_size=1)
    text = " ".join(seg.text for seg in segments).strip().lower()
    return text


def transcribe_stub(audio_path: str, lang: str) -> str:
    """
    Stub transcriber used when no model path is provided.
    Returns a placeholder so the benchmark pipeline can be tested end-to-end
    without downloading a model.

    TODO: Replace with actual STT call once model files are available.
    """
    logger.warning(
        "No --model provided. Using stub transcriber for '%s'. "
        "WER results will NOT reflect real STT accuracy.",
        audio_path,
    )
    stub_outputs = {
        "en": "this is a test audio clip with some extra words",
        "hi": "यह एक परीक्षण क्लिप है",
        "te": "ఇది ఒక పరీక్ష క్లిప్",
    }
    return stub_outputs.get(lang, "stub output")


# ── data loading ─────────────────────────────────────────────────────────────

def load_test_data(data_path: str) -> list[dict[str, str]]:
    """Load test samples from a CSV file with columns: audio_file, transcript."""
    path = Path(data_path)
    if not path.exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    samples: list[dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"audio_file", "transcript"}.issubset(reader.fieldnames):
            logger.error("CSV missing required columns: audio_file, transcript")
            sys.exit(1)
        for row in reader:
            samples.append({
                "audio_file": row["audio_file"].strip(),
                "transcript": row["transcript"].strip().lower(),
            })

    if not samples:
        logger.error("No samples found in: %s", data_path)
        sys.exit(1)

    return samples


# ── report helpers ────────────────────────────────────────────────────────────

def save_csv_report(results: list[dict], output_path: str) -> None:
    """Save per-sample WER results to a CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["audio_file", "reference", "hypothesis", "sample_wer_pct"],
        )
        writer.writeheader()
        writer.writerows(results)
    logger.info("Detailed report saved → %s", output_path)


def print_summary(lang: str, overall_wer: float, n_samples: int) -> None:
    """Print the final WER summary line matching the issue's expected output."""
    lang_name = LANGUAGE_NAMES.get(lang, lang)
    print(
        f"\nLanguage: {lang_name} | "
        f"WER: {overall_wer:.1f}% | "
        f"Samples: {n_samples}"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WER Benchmark for Voice Assistant STT Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--lang",
        required=True,
        choices=list(LANGUAGE_NAMES.keys()),
        help="Language code: en | hi | te",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to CSV test file (columns: audio_file, transcript)",
    )
    parser.add_argument(
        "--backend",
        choices=["vosk", "whispercpp"],
        default=None,
        help="STT backend to use (omit to run in stub/demo mode)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to STT model file (required when --backend is set)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save per-sample CSV report",
    )

    args = parser.parse_args()

    # Validate model path when backend is specified
    if args.backend and not args.model:
        parser.error("--model is required when --backend is specified")

    lang = args.lang
    lang_name = LANGUAGE_NAMES[lang]

    print(f"\n{'=' * 52}")
    print(f"  WER Benchmark — {lang_name}")
    print(f"  Data    : {args.data}")
    print(f"  Backend : {args.backend or 'stub (demo mode)'}")
    print(f"  Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 52}\n")

    samples = load_test_data(args.data)
    logger.info("Loaded %d sample(s) from %s", len(samples), args.data)

    references: list[str] = []
    hypotheses: list[str] = []
    detailed_results: list[dict] = []

    # Load model ONCE before the loop
    stt_model = None
    if args.backend == "vosk":
        logger.info("Loading Vosk model from %s ...", args.model)
        stt_model = load_vosk_model(args.model)
    elif args.backend == "whispercpp":
        logger.info("Loading WhisperCpp model from %s ...", args.model)
        stt_model = load_whispercpp_model(args.model)

    for i, sample in enumerate(samples, start=1):
        audio_file = sample["audio_file"]
        reference = sample["transcript"]

        # Transcribe
        try:
            if args.backend == "vosk":
                hypothesis = transcribe_with_vosk(audio_file, stt_model)
            elif args.backend == "whispercpp":
                hypothesis = transcribe_with_whispercpp(audio_file, stt_model)
            else:
                hypothesis = transcribe_stub(audio_file, lang)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.error("Skipping %s — %s", audio_file, exc)
            continue

        sample_wer = compute_wer(reference, hypothesis) * 100

        references.append(reference)
        hypotheses.append(hypothesis)
        detailed_results.append({
            "audio_file": audio_file,
            "reference": reference,
            "hypothesis": hypothesis,
            "sample_wer_pct": f"{sample_wer:.1f}",
        })

        print(f"  [{i}/{len(samples)}] {audio_file}")
        print(f"    REF : {reference}")
        print(f"    HYP : {hypothesis}")
        print(f"    WER : {sample_wer:.1f}%\n")

    if not references:
        logger.error("No samples were successfully transcribed. Exiting.")
        sys.exit(1)

    overall_wer = compute_wer(references, hypotheses) * 100

    print(f"{'=' * 52}")
    print_summary(lang, overall_wer, len(references))
    print(f"{'=' * 52}\n")

    if args.output:
        save_csv_report(detailed_results, args.output)


if __name__ == "__main__":
    main()