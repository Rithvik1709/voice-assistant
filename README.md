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
3. Run the automated model downloader:
   - `python setup_models.py`
4. Run the voice assistant:
   - `python -m voice_assistant.main --mode local`

## 📂 Model Download & Setup Guide

For the easiest setup, simply run the automated CLI tool:
```bash
python setup_models.py
```
This will prompt you to choose an LLM (either a fast 398MB Qwen model for testing, or a high-quality Llama-3-8B model), automatically download the Piper TTS voices and the Vosk ASR model, and generate your `.env` file for you.

### Manual Configuration (Optional)

If you prefer to configure your models manually:

1. Create a folder named `models` in the root directory.
2. Place your downloaded `.gguf`, `.onnx` (with `.onnx.json`), and extracted Vosk ASR folder inside `models/`.
3. Create/update your `.env` file:

```env
MODEL_PATH="models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
PIPER_VOICE="models/en_US-lessac-medium.onnx"
ASR_MODEL_PATH="models/vosk-model-small-en-us-0.15"
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
