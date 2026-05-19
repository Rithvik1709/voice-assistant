# voice-assistant

Low-latency real-time streaming voice assistant pipeline:

Mic → StreamingASR → PartialText → LLM (token streaming) → StreamingTTS → Speaker

## Latency Breakdown (target)

```text
[Mic frames 20ms] -> [VAD boundary + partial ASR]
                    ~20-80ms
              -> [LLM TTFT]
                    ~60-180ms
              -> [TTS eager chunk]
                    ~40-120ms
              -> [Speaker out]
                    ~6-20ms

Perceived first-response latency target: < 500ms local
Default low-latency mode target: sub-100ms perceived updates (ack tone + eager chunking)
```

## Features

- Chunked ASR with VAD gating and partial/final events
- Streaming `llama-cpp-python` token callbacks with TTFT instrumentation
- Speculative decoding helper (draft + target verify)
- KV-cache reuse between turns via llama state save/load
- Sentence-chunked Piper TTS synthesis streamed to non-blocking player
- Low-latency mode defaults: shorter endpointing, eager TTS flush, 128-sample playback blocks
- Optional immediate acknowledgement tone before LLM completion for sub-100ms perceived feedback
- Async orchestrator with queue backpressure
- Barge-in interrupt handling (speech during playback cancels output)
- gRPC bidirectional streaming mode for remote inference
- Benchmark metrics: ASR latency, TTFT, TTS first chunk, end-to-end, RTF

## Quickstart (local)

1. Install Python 3.11+.
2. Install dependencies:
   - CPU: `pip install -e .`
   - CUDA: `pip install -e .[cuda]`
   - Apple Metal: `pip install -e .[metal]`
3. Copy env:
   - `cp .env.example .env`
4. Fill model paths in `.env`.
5. Run:
   - `python -m voice_assistant.main --mode local`

## 📂 Model Download & Setup Guide

To run this voice assistant in local mode, you need to download the Large Language Model (LLM) and Text-to-Speech (TTS) files manually and configure their paths.

### 1. Large Language Model (LLM)
This pipeline uses `llama-cpp-python` for local inference, which requires models in the **GGUF** format.
* **Recommended Model:** Llama-3-8B-Instruct or Mistral-7B-Instruct.
* **Where to Download:** Search on Hugging Face (popular repositories include `QuantFactory` or `Bartowski`).
* **Recommended Quantization:** Download the `Q4_K_M` or `Q5_K_M` version (e.g., `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`). This provides the best speed-to-performance ratio for low latency.

### 2. Text-to-Speech (TTS) Model
This project utilizes **Piper TTS** for sentence-chunked, non-blocking audio generation.
* **Where to Download:** Visit the official Piper voice repository on Hugging Face or GitHub.
* **Files Needed:** You need both the model file (`.onnx`) and its configuration file (`.json`).
* **Recommended Voice:** `en_US-lessac-medium.onnx` and `en_US-lessac-medium.onnx.json`.

### 3. Updating your `.env` File
1. Create a new folder named `models` in the root directory of this project.
2. Place your downloaded `.gguf`, `.onnx`, and `.json` files inside that folder.
3. Open your `.env` file and update the paths to point to your files:

```env
MODEL_PATH="models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
PIPER_VOICE="models/en_US-lessac-medium.onnx"
ASR_MODEL_PATH="models/vosk-model-small-en-us-0.15"
ASR_BACKEND="vosk"
VAD_AGGRESSIVENESS=2
CHUNK_MS=20
ASR_ENDPOINT_SILENCE_MS=60
TTS_SENTENCE_MAX_TOKENS=8
TTS_EAGER_MIN_WORDS=3
PLAYER_BLOCKSIZE=128
GRPC_PORT=50051
```

### 4. Installing Piper TTS

Piper is used as the TTS engine and must be available on your `PATH`:

**Option A — pip (Linux/macOS):**

```bash
pip install piper-tts
```

**Option B — Pre-built binary:**

Download the latest release from [piper releases](https://github.com/rhasspy/piper/releases), extract the binary, and add it to your `PATH`.

Verify the installation:

```bash
piper --help
```

## gRPC mode

Generate protobuf stubs once:

- `python -m grpc_tools.protoc -I voice_assistant/transport --python_out=voice_assistant/transport --grpc_python_out=voice_assistant/transport voice_assistant/transport/voice_assistant.proto`

Server:

- `python -m voice_assistant.main --mode server --host 0.0.0.0 --port 50051`

Client:

- `python -m voice_assistant.main --mode client --target localhost:50051`

## Tests

- `pytest -q`

Covers:

- VAD frame/speech boundary behavior
- Sentence chunking for streaming TTS
- Speculative decode acceptance rate and fallback logic
