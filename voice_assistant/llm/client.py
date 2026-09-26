from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace

from voice_assistant.benchmark import BenchmarkTracker

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

try:
    from llama_cpp import Llama  # type: ignore
    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False
    class Llama:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

_DONE = object()


@dataclass(slots=True)
class LLMConfig:
    model_path: str
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    max_tokens: int = 256
    temperature: float = 0.7


class StreamingLLMClient:
    def __init__(self, config: LLMConfig, bench: BenchmarkTracker | None = None) -> None:
        if not _LLAMA_AVAILABLE:
            raise RuntimeError(
                "llama-cpp-python is required for the LLM. Install it with "
                "`pip install 'voice-assistant[local]'`."
            )

        self._llama = Llama(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            logits_all=False,
            embedding=False,
            verbose=False,
        )
        self.config = config
        self.bench = bench
        # llama.cpp contexts are not thread-safe; gRPC streams share one model.
        self._lock = threading.Lock()

    async def stream_tokens(
        self,
        messages: list[dict[str, str]],
        out_queue: asyncio.Queue[str],
        bench: BenchmarkTracker | None = None,
    ) -> str:
        """Stream reply tokens into `out_queue` and return the full reply.

        Generation runs in a worker thread so audio capture, TTS and playback
        keep running while tokens are produced. Cancelling the awaiting task
        stops generation at the next token.
        """
        active_bench = bench or self.bench
        loop = asyncio.get_running_loop()
        tokens: asyncio.Queue[object] = asyncio.Queue()
        stop = threading.Event()

        def produce() -> None:
            try:
                with self._lock:
                    stream = self._llama.create_chat_completion(
                        messages=messages,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        stream=True,
                    )
                    for packet in stream:
                        if stop.is_set():
                            break
                        token = packet["choices"][0].get("delta", {}).get("content", "")
                        if token:
                            loop.call_soon_threadsafe(tokens.put_nowait, token)
            except BaseException as exc:  # forwarded to the awaiting coroutine
                loop.call_soon_threadsafe(tokens.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(tokens.put_nowait, _DONE)

        with tracer.start_as_current_span("llm.stream_tokens") as span:
            assembled: list[str] = []
            start = time.perf_counter()
            worker = loop.run_in_executor(None, produce)

            try:
                while True:
                    item = await tokens.get()
                    if item is _DONE:
                        break
                    if isinstance(item, BaseException):
                        raise RuntimeError(f"LLM generation failed: {item}") from item

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
                        logger.error("LLM output queue backpressure; aborting generation task")
                        raise RuntimeError("LLM output queue full; generation aborted") from exc
                    assembled.append(token)
            finally:
                stop.set()
                # Let the worker observe the stop flag so the model lock is
                # released before the next request starts.
                await asyncio.shield(worker)

            span.set_attribute("llm.completion_tokens", len(assembled))
            return "".join(assembled)

    def warmup(self, system_prompt: str = "") -> float:
        """Process the system prompt once so the first real turn is faster.

        llama.cpp keeps the evaluated tokens and reuses the matching prefix of
        the next prompt, so the system prompt is not processed again when the
        user asks their first question. Returns the time taken in seconds.
        """
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": "Hi"})
        start = time.perf_counter()
        with self._lock:
            self._llama.create_chat_completion(messages=messages, max_tokens=1, temperature=0.0)
        return time.perf_counter() - start

    def tokenize(self, text: str) -> list[int]:
        return self._llama.tokenize(text.encode("utf-8"), add_bos=False)

    def detokenize(self, tokens: list[int]) -> str:
        return self._llama.detokenize(tokens).decode("utf-8", errors="ignore")

    def token_logprobs(self, prompt: str, candidates: list[str]) -> dict[str, float]:
        """Log-probabilities of each candidate being the next token after `prompt`."""
        with self._lock:
            completion = self._llama.create_completion(
                prompt=prompt,
                max_tokens=1,
                temperature=0.0,
                logprobs=max(8, len(candidates)),
                echo=False,
            )
        logprobs = completion["choices"][0].get("logprobs") or {}
        top = (logprobs.get("top_logprobs") or [{}])[0] or {}
        return {cand: float(top.get(cand, -100.0)) for cand in candidates}


async def warm_up_llm(llm, system_prompt: str) -> None:
    warmup = getattr(llm, "warmup", None)
    if warmup is None:
        return
    try:
        seconds = await asyncio.to_thread(warmup, system_prompt.strip())
        logger.info("LLM warmed up in %.2fs", seconds)
    except Exception as exc:  # a failed warm-up only costs latency
        logger.warning("LLM warm-up failed: %s", exc)
