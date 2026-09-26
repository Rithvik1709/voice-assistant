from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from voice_assistant.config import ConfigError, Settings
from voice_assistant.llm import create_llm
from voice_assistant.llm.openai_compat import OpenAICompatConfig, OpenAICompatLLMClient


class FakeOpenAI(BaseHTTPRequestHandler):
    tokens = ["Paris", " is", " the", " capital", "."]
    delay = 0.0
    requests: list[dict] = []
    fail_status: int | None = None

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenAI.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        if FakeOpenAI.fail_status:
            payload = json.dumps({"error": {"message": "invalid api key"}}).encode()
            self.send_response(FakeOpenAI.fail_status)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if not body.get("stream"):
            payload = json.dumps({"choices": [{"message": {"content": "Hi"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            self.wfile.write(b": keep-alive\n\n")
            for tok in FakeOpenAI.tokens:
                chunk = {"choices": [{"delta": {"content": tok}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                time.sleep(FakeOpenAI.delay)
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.fixture
def server():
    FakeOpenAI.requests = []
    FakeOpenAI.delay = 0.0
    FakeOpenAI.fail_status = None
    FakeOpenAI.tokens = ["Paris", " is", " the", " capital", "."]
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def client(base_url: str, **kw) -> OpenAICompatLLMClient:
    return OpenAICompatLLMClient(OpenAICompatConfig(base_url=base_url, model="qwen2.5:7b", **kw))


async def test_streams_tokens_and_sends_auth(server) -> None:
    out: asyncio.Queue[str] = asyncio.Queue()

    reply = await client(server, api_key="sk-test", max_tokens=64).stream_tokens(
        [{"role": "user", "content": "capital of france?"}], out
    )

    assert reply == "Paris is the capital."
    assert out.qsize() == 5
    req = FakeOpenAI.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["auth"] == "Bearer sk-test"
    assert req["body"]["model"] == "qwen2.5:7b"
    assert req["body"]["stream"] is True
    assert req["body"]["max_tokens"] == 64


async def test_http_error_message_is_surfaced(server) -> None:
    FakeOpenAI.fail_status = 401

    with pytest.raises(RuntimeError, match="HTTP 401: invalid api key"):
        await client(server).stream_tokens([], asyncio.Queue())


async def test_cancellation_returns_promptly(server) -> None:
    FakeOpenAI.tokens = [f" w{i}" for i in range(200)]
    FakeOpenAI.delay = 0.02
    out: asyncio.Queue[str] = asyncio.Queue()

    task = asyncio.create_task(client(server).stream_tokens([], out))
    while out.qsize() < 2:
        await asyncio.sleep(0.005)
    start = time.perf_counter()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert time.perf_counter() - start < 0.2


def test_warmup_sends_one_token_request(server) -> None:
    assert client(server).warmup("Be brief.") >= 0
    body = FakeOpenAI.requests[0]["body"]
    assert body["max_tokens"] == 1
    assert body["stream"] is False
    assert body["messages"][0] == {"role": "system", "content": "Be brief."}


def test_create_llm_selects_backend_and_validates(monkeypatch) -> None:
    monkeypatch.delenv("MOCK_MODELS", raising=False)
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "llama3.1")
    monkeypatch.setenv("MODEL_PATH", "")

    settings = Settings()
    assert isinstance(create_llm(settings), OpenAICompatLLMClient)
    settings.validate(need_asr=False, need_tts=False)  # no MODEL_PATH needed

    monkeypatch.setenv("LLM_MODEL", "")
    with pytest.raises(ConfigError, match="LLM_MODEL"):
        Settings().validate(need_asr=False, need_tts=False)
