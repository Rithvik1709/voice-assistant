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
- Optional offline wake-word detection via openWakeWord (`--wakeword` flag)

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
```

## gRPC mode

Generate protobuf stubs once:

- `python -m grpc_tools.protoc -I voice_assistant/transport --python_out=voice_assistant/transport --grpc_python_out=voice_assistant/transport voice_assistant/transport/voice_assistant.proto`

Server:

- `python -m voice_assistant.main --mode server --host 0.0.0.0 --port 50051`

Client:

- `python -m voice_assistant.main --mode client --target localhost:50051`

## Wake Word Support

The assistant supports fully **offline** wake word detection powered by [openWakeWord](https://github.com/dscripka/openWakeWord). When enabled, the pipeline waits silently until the chosen wake word is spoken before starting the speech-to-text workflow.

### Install dependencies

```bash
pip install -e .
```

`openwakeword`, `sounddevice`, and `numpy` are included in the standard dependency list and will be installed automatically.

### Run with wake word detection

```bash
python -m voice_assistant --wakeword alexa
```

Replace `alexa` with any model name supported by openWakeWord (e.g. `hey_mycroft`, `hey_jarvis`). The model is downloaded automatically by openWakeWord on first use.

### Run without wake word detection

Simply omit the `--wakeword` flag — the assistant behaves exactly as before:

```bash
python -m voice_assistant.main --mode local
```

### How it works

| Invocation | Behaviour |
|---|---|
| `--wakeword alexa` | Listens for "Alexa" before starting the pipeline |
| `--wakeword <model>` | Listens for the specified openWakeWord model |
| *(flag omitted)* | No wake word gate; pipeline starts immediately |

## Tests

- `pytest -q`

Covers:

- VAD frame/speech boundary behavior
- Sentence chunking for streaming TTS
- Speculative decode acceptance rate and fallback logic
