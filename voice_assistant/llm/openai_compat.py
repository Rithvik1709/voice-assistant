"""Streaming client for any OpenAI-compatible chat completions API.

Lets Vaani use a larger or hosted model instead of a local GGUF file:
Ollama (http://localhost:11434/v1), LM Studio (http://localhost:1234/v1),
vLLM, llama.cpp's server, or a hosted provider. Uses only the standard
library, and streams tokens with the same interface as StreamingLLMClient.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from opentelemetry import trace

from voice_assistant.benchmark import BenchmarkTracker

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

_DONE = object()


@dataclass(slots=True)
class OpenAICompatConfig:
    base_url: str
    model: str
    api_key: str = ""
    max_tokens: int = 256
    temperature: float = 0.7
    timeout_s: float = 30.0


class OpenAICompatLLMClient:
    def __init__(self, config: OpenAICompatConfig, bench: BenchmarkTracker | None = None) -> None:
        if not config.base_url:
            raise ValueError("LLM_BASE_URL is required for the openai LLM backend")
        if not config.model:
            raise ValueError("LLM_MODEL is required for the openai LLM backend")
        self.config = config
        self.bench = bench
        self._url = config.base_url.rstrip("/") + "/chat/completions"

    def _request(self, messages: list[dict[str, str]], max_tokens: int, stream: bool) -> urllib.request.Request:
        body = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.config.temperature,
            "stream": stream,
        }
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return urllib.request.Request(self._url, data=json.dumps(body).encode("utf-8"), headers=headers)

    @staticmethod
    def _describe_http_error(exc: urllib.error.HTTPError) -> str:
        try:
            detail = json.loads(exc.read().decode("utf-8", "replace"))
            message = detail.get("error", {}).get("message") if isinstance(detail.get("error"), dict) else detail.get("error")
        except Exception:
            message = None
        return f"HTTP {exc.code}" + (f": {message}" if message else "")

    async def stream_tokens(
        self,
        messages: list[dict[str, str]],
        out_queue: asyncio.Queue[str],
        bench: BenchmarkTracker | None = None,
    ) -> str:
        """Stream reply tokens into `out_queue` and return the full reply.

        Cancelling the awaiting task closes the HTTP stream at the next chunk.
        """
        active_bench = bench or self.bench
        loop = asyncio.get_running_loop()
        tokens: asyncio.Queue[object] = asyncio.Queue()
        stop = threading.Event()

        def produce() -> None:
            try:
                request = self._request(messages, self.config.max_tokens, stream=True)
                with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                    for raw in response:
                        if stop.is_set():
                            break
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        choices = json.loads(data).get("choices") or [{}]
                        token = (choices[0].get("delta") or {}).get("content") or ""
                        if token:
                            loop.call_soon_threadsafe(tokens.put_nowait, token)
            except urllib.error.HTTPError as exc:
                loop.call_soon_threadsafe(tokens.put_nowait, RuntimeError(self._describe_http_error(exc)))
            except BaseException as exc:
                loop.call_soon_threadsafe(tokens.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(tokens.put_nowait, _DONE)

        with tracer.start_as_current_span("llm.stream_tokens") as span:
            span.set_attribute("llm.backend", "openai")
            assembled: list[str] = []
            start = time.perf_counter()
            worker = loop.run_in_executor(None, produce)
            try:
                while True:
                    item = await tokens.get()
                    if item is _DONE:
                        break
                    if isinstance(item, BaseException):
                        raise RuntimeError(f"LLM request to {self._url} failed: {item}") from item
                    token = str(item)
                    if not assembled:
                        if active_bench:
                            active_bench.mark("first_token_ts")
                        ttft = (time.perf_counter() - start) * 1000.0
                        logger.info("TTFT_ms=%.2f", ttft)
                        span.set_attribute("llm.ttft_ms", ttft)
                    try:
                        await asyncio.wait_for(out_queue.put(token), timeout=10.0)
                    except TimeoutError as exc:
                        raise RuntimeError("LLM output queue full; generation aborted") from exc
                    assembled.append(token)
            finally:
                # The worker closes the HTTP stream when it sees the flag; it is
                # not awaited so a slow server cannot delay an interruption.
                stop.set()
                del worker
            span.set_attribute("llm.completion_tokens", len(assembled))
            return "".join(assembled)

    def warmup(self, system_prompt: str = "") -> float:
        """Send a one-token request so the server loads the model and caches the prompt."""
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": "Hi"})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(self._request(messages, 1, stream=False), timeout=120):
                pass
        except urllib.error.HTTPError as exc:
            raise RuntimeError(self._describe_http_error(exc)) from exc
        return time.perf_counter() - start
