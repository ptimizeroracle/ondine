#!/usr/bin/env python
"""Measure a real run, and separate the part ondine owns from the part it does not.

`benchmark_throughput.py` answers "how fast is ondine with the provider removed".
That number is necessary — it is the ceiling — but it cannot tell you whether
ondine *stays* efficient once calls actually take time. Queueing, concurrency
limits, retries and rate limiting only exist under real latency, and a run that
is perfectly efficient at 0ms can waste half its wall clock at 400ms.

So this measures a real provider, then splits the result:

    floor      = sum(call latencies) / concurrency
                 the wall clock a perfect scheduler would need — every worker
                 busy from start to finish, zero orchestration cost
    actual     = measured wall clock
    efficiency = floor / actual

Efficiency is the number worth reporting, because it survives changing the
provider, the model, the tier and the day. A raw rows/second figure from a real
run says more about the provider's fleet than about this library.

Costs real money. Defaults are deliberately small.

    uv run python scripts/benchmark_real_provider.py --rows 500
    uv run python scripts/benchmark_real_provider.py --rows 2000 --concurrency 16
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ondine import PipelineBuilder  # noqa: E402
from ondine.adapters.llm_client import LLMClient  # noqa: E402
from ondine.adapters.unified_litellm_client import UnifiedLiteLLMClient  # noqa: E402
from ondine.core.models import LLMResponse  # noqa: E402, TC001
from ondine.core.specifications import LLMProvider, LLMSpec  # noqa: E402
from scripts.benchmark_throughput import synthetic_frame  # noqa: E402


class TimingClient(LLMClient):
    """The real client, with a stopwatch around every call.

    A decorator rather than a subclass: the point is to time exactly what the
    pipeline would have run anyway, without changing how it runs. Latency is
    measured here rather than read from `LLMResponse.latency_ms` so that
    whatever the provider adapter does — retries inside LiteLLM included —
    lands inside the measurement.
    """

    def __init__(self, spec: LLMSpec) -> None:
        super().__init__(spec)
        self._inner = UnifiedLiteLLMClient(spec)
        self._lock = threading.Lock()
        self.latencies_s: list[float] = []
        self.failures = 0

    def _record(self, started: float) -> None:
        with self._lock:
            self.latencies_s.append(time.perf_counter() - started)

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        started = time.perf_counter()
        try:
            return self._inner.invoke(prompt, **kwargs)
        finally:
            self._record(started)

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        started = time.perf_counter()
        try:
            return await self._inner.ainvoke(prompt, **kwargs)
        finally:
            self._record(started)

    def structured_invoke(self, prompt: str, output_cls: Any, **kwargs: Any):
        started = time.perf_counter()
        try:
            return self._inner.structured_invoke(prompt, output_cls, **kwargs)
        finally:
            self._record(started)

    async def structured_invoke_async(
        self, prompt: str, output_cls: Any, **kwargs: Any
    ):
        started = time.perf_counter()
        try:
            return await self._inner.structured_invoke_async(
                prompt, output_cls, **kwargs
            )
        finally:
            self._record(started)

    def estimate_tokens(self, text: str) -> int:
        return self._inner.estimate_tokens(text)

    async def start(self) -> None:
        await self._inner.start()

    async def stop(self) -> None:
        await self._inner.stop()


# Cheap, fast models. The claim being measured is about orchestration, so the
# right model is whichever answers a one-word classification quickly.
PRESETS = {
    # $0.08/M input, $0.16/M output at the time of writing — the cheapest of
    # these by an order of magnitude, and the reason to disable thinking:
    # reasoning tokens are billed as output.
    "deepseek": (
        "OPENROUTER_API_KEY",
        LLMProvider.OPENAI,
        "openrouter/deepseek/deepseek-v4-flash-latest",
    ),
    "qwen": ("OPENROUTER_API_KEY", LLMProvider.OPENAI, "openrouter/qwen/qwen3.6-27b"),
    "groq": ("GROQ_API_KEY", LLMProvider.GROQ, "groq/openai/gpt-oss-20b"),
    "cerebras": ("CEREBRAS_API_KEY", LLMProvider.OPENAI, "cerebras/llama3.1-8b"),
    "together": (
        "TOGETHER_API_KEY",
        LLMProvider.OPENAI,
        "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--provider", choices=tuple(PRESETS), default="deepseek")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help=(
            "Output cap per call. 0 derives it from batch size. Reasoning "
            "models spend this budget thinking before they answer, so a cap "
            "tuned for a direct model silently truncates every response."
        ),
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help=(
            "Let a reasoning model think before answering. Off by default: "
            "for bulk classification the reasoning tokens are pure latency and "
            "cost, and a cap that truncates them mid-thought returns an empty "
            "body — which the pipeline cannot tell from a provider outage."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env_var, provider, model = PRESETS[args.provider]
    if not os.getenv(env_var):
        raise SystemExit(f"{env_var} is not set — this benchmark calls a real API")

    spec = LLMSpec(
        provider=provider,
        model=model,
        temperature=0.0,
        # One word per row plus JSON scaffolding. A flat cap here truncates
        # every batched response, and a truncated batch is indistinguishable
        # from a provider outage — which is exactly how #... was found.
        max_tokens=args.max_tokens or max(64, args.batch_size * 24),
        input_cost_per_1k_tokens=Decimal("0.0001"),
        output_cost_per_1k_tokens=Decimal("0.0005"),
    )
    # OpenRouter takes reasoning control in the request body; LiteLLM passes
    # extra_body straight through. Harmless on models that do not reason.
    if not args.thinking:
        spec.extra_params = {"extra_body": {"reasoning": {"enabled": False}}}

    client = TimingClient(spec)

    frame = synthetic_frame(args.rows, seed=args.seed)
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            frame, input_columns=["description"], output_columns=["category"]
        )
        .with_prompt(
            "Reply with exactly one word naming this product's category: {description}"
        )
        .with_custom_llm_client(client)
        .with_batch_size(args.batch_size)
        .with_concurrency(args.concurrency)
        .build()
    )

    print(
        f"provider={args.provider} model={model} rows={args.rows:,} "
        f"batch_size={args.batch_size} concurrency={args.concurrency}"
    )

    started = time.perf_counter()
    result = pipeline.execute()
    wall_s = time.perf_counter() - started

    # Ask the pipeline how many rows it lost, rather than guessing from the
    # cells. Sniffing for empty strings and the `[SKIPPED]` marker missed the
    # literal string "null" that a failed batch leaves behind — this script
    # reported "200/200 answered" on a run that lost 180 rows.
    lost = result.metrics.skipped_rows + result.metrics.failed_rows
    answered = args.rows - lost
    latencies = sorted(client.latencies_s)
    if not latencies:
        raise SystemExit("no calls were recorded — nothing to measure")

    # What a perfect scheduler would have taken: every worker busy end to end.
    floor_s = sum(latencies) / args.concurrency
    efficiency = floor_s / wall_s

    def percentile(fraction: float) -> float:
        return latencies[min(len(latencies) - 1, int(len(latencies) * fraction))]

    print(f"\n  wall clock        {wall_s:8.2f}s")
    print(f"  provider floor    {floor_s:8.2f}s   (sum of call time / concurrency)")
    print(f"  ondine overhead   {wall_s - floor_s:8.2f}s")
    print(f"  efficiency        {efficiency:8.1%}")
    print(
        f"\n  calls {len(latencies):,}  "
        f"p50 {statistics.median(latencies) * 1000:,.0f}ms  "
        f"p95 {percentile(0.95) * 1000:,.0f}ms  "
        f"max {latencies[-1] * 1000:,.0f}ms"
    )
    print(
        f"  rows {answered:,}/{args.rows:,} answered   "
        f"{args.rows / wall_s:,.1f} rows/s   "
        f"cost ${result.costs.total_cost:.4f}   "
        f"skipped {result.metrics.skipped_rows}"
    )

    per_million_s = 1_000_000 / (args.rows / wall_s)
    print(
        f"\n  at this provider, model, batch size and concurrency: "
        f"{per_million_s / 60:,.0f} min per 1M rows"
    )


if __name__ == "__main__":
    main()
