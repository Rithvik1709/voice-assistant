# Vaani

Vaani is a low-latency, open-source real-time voice AI pipeline for local and remote voice interaction.

It combines streaming speech recognition, LLM response generation, and chunked text-to-speech into a single end-to-end system optimized for conversational responsiveness.

## Release Status

Vaani v1.0.0 is the first stable open-source release baseline. This version includes a working local pipeline, gRPC streaming mode, mock-model validation for CI, packaging metadata for distribution, and verified build/test checks.

## Architecture

```text
Mic / Audio Input
    ↓
Streaming ASR + VAD
    ↓
Partial / Final transcript
    ↓
LLM token streaming
    ↓
Sentence chunked TTS
    ↓
Audio playback
```

## Highlights

- Real-time ASR with VAD gating and partial/final speech detection
- Low-latency LLM token streaming with TTFT instrumentation
- Speculative decoding support and KV-cache compatibility hooks
- Sentence-chunked Piper TTS synthesis with non-blocking playback
- Barge-in interruption handling for live assistant interaction
- gRPC server/client mode for remote deployment
- Benchmark metrics for ASR latency, TTFT, TTS first-chunk latency, end-to-end latency, and RTF
- Production-oriented configuration validation for model paths and runtime dependencies

## Performance Targets

```text
[Mic frames 20ms] -> [VAD boundary + partial ASR]
                    ~20-80ms
              -> [LLM TTFT]
                    ~60-180ms
              -> [TTS eager chunk]
                    ~40-120ms
              -> [Speaker out]
                    ~6-20ms

Target perceived first-response latency: < 500ms local
Target low-latency mode: sub-100ms perceived feedback for enabled updates
```

## Current Project Status

This release is intended as a stable baseline for open-source experimentation and extension. It is ready for local deployment, benchmarking, and further community-driven development.

### Verified in v1.0.0

- 17 automated tests passing
- source distribution build successful
- wheel build successful
- package metadata validation successful via Twine

## Quickstart

### Requirements

- Python 3.11+
- pip
- local or remote model assets for ASR, LLM, and TTS

### Install

```bash
pip install -e .
```

For CUDA or Metal builds:

```bash
pip install -e .[cuda]
pip install -e .[metal]
```

### Run locally

```bash
python -m voice_assistant.main --mode local
```

## Model Setup

To run Vaani locally, place the required model assets inside a `models/` directory and configure them in your environment.

### 1. LLM model

Vaani uses `llama-cpp-python` and expects a GGUF model.

Recommended choices:
- Llama-3-8B-Instruct
- Mistral-7B-Instruct

Preferred quantization:
- Q4_K_M
- Q5_K_M

### 2. TTS model

Vaani uses Piper TTS for streaming speech synthesis.

Required files:
- `en_US-lessac-medium.onnx`
- `en_US-lessac-medium.onnx.json`

Install Piper:

```bash
pip install piper-tts
```

Or use the official prebuilt binary and ensure it is available in your `PATH`.

```bash
piper --help
```

### 3. ASR model

Vaani supports Vosk-based local ASR.

Recommended model:
- `vosk-model-small-en-us-0.15`

Download from the official Vosk model repository and extract it into `models/`.

## Environment Configuration

Create a `.env` file with paths similar to the following:

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

## gRPC Mode

### Generate protobuf stubs

```bash
python -m grpc_tools.protoc \
  -I. \
  --python_out=. \
  --grpc_python_out=. \
  voice_assistant/transport/voice_assistant.proto
```

### Start server

```bash
python -m voice_assistant.main --mode server --host 0.0.0.0 --port 50051
```

### Connect client

```bash
python -m voice_assistant.main --mode client --target localhost:50051
```

## Benchmarking

A local benchmark is included under the project’s benchmark tooling for latency and throughput measurements.

Example:

```bash
PYTHONPATH=. python3 scripts/bench/load_asr.py \
  --host 127.0.0.1 \
  --port 50051 \
  --concurrency 10 \
  --frames-per-client 30 \
  --frame-ms 30 \
  --sample-rate 16000
```

## Testing

Install development dependencies and run the test suite:

```bash
pip install -e .[dev]
pytest -q
```

## Project Roadmap

This release provides the stable core foundation. Planned areas of development include:

- broader model compatibility
- improved latency tuning
- better multi-turn memory handling
- stronger deployment tooling
- wider platform integrations

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome. Please open an issue or pull request for features, fixes, or improvements.

## Acknowledgements

Vaani builds on the open-source ecosystem around `llama-cpp-python`, Piper TTS, Vosk, and the broader voice AI community.
