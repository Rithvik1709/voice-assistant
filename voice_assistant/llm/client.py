from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from voice_assistant.benchmark import BenchmarkTracker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMConfig:
    model_path: str
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    max_tokens: int = 256
    temperature: float = 0.7


class StreamingLLMClient:
    def __init__(self, config: LLMConfig, bench: BenchmarkTracker | None = None) -> None:
        try:
            from llama_cpp import Llama
        except Exception as exc:  # pragma: no cover - runtime dep
            raise RuntimeError("llama-cpp-python is required") from exc

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

    async def stream_tokens(self, prompt: str, out_queue: asyncio.Queue[str]) -> str:
        if self.bench:
            self.bench.mark("prompt_sent_ts")

        first_token_seen = False
        assembled: list[str] = []

        stream = self._llama.create_completion(
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stream=True,
        )

        start = time.perf_counter()
        for packet in stream:
            token = packet["choices"][0]["text"]
            if not token:
                continue
            if not first_token_seen:
                first_token_seen = True
                if self.bench:
                    self.bench.mark("first_token_ts")
                logger.info("TTFT_ms=%.2f", (time.perf_counter() - start) * 1000.0)
            assembled.append(token)
            await out_queue.put(token)

        return "".join(assembled)

    def tokenize(self, text: str) -> list[int]:
        return self._llama.tokenize(text.encode("utf-8"), add_bos=False)

    def detokenize(self, tokens: list[int]) -> str:
        return self._llama.detokenize(tokens).decode("utf-8", errors="ignore")

    def token_logprobs(self, prompt: str, candidates: list[str]) -> dict[str, float]:
        result: dict[str, float] = {}
        for cand in candidates:
            completion = self._llama.create_completion(
                prompt=prompt,
                max_tokens=1,
                temperature=0.0,
                logprobs=8,
                echo=False,
            )
            top = completion["choices"][0].get("logprobs", {}).get("top_logprobs", [{}])[0]
            result[cand] = float(top.get(cand, -100.0))
        return result
