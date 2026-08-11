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
#: (env var, provider, model, base URL, body that turns reasoning off).
#:
#: Every provider spells "do not think" differently, and getting it wrong is
#: not a no-op: reasoning tokens bill as output, and they push a batched reply
#: past any token cap tuned for the answer alone — which arrives as a truncated
#: body the pipeline cannot tell from an outage. DeepSeek v4 flash spends 20
#: output tokens thinking about "what colour is the sky" and 1 answering it.
PRESETS = {
    # DeepSeek's own endpoint rather than a reseller: first-party pricing, and
    # one less hop to blame when a call is slow. $0.14/M in, $0.28/M out.
    "deepseek": (
        "DEEPSEEK_API_KEY",
        LLMProvider.OPENAI,
        "deepseek/deepseek-v4-flash",
        "https://api.deepseek.com/v1",
        {"thinking": {"type": "disabled"}},
    ),
    "qwen": (
        "OPENROUTER_API_KEY",
        LLMProvider.OPENAI,
        "openrouter/qwen/qwen3.6-27b",
        None,
        {"reasoning": {"enabled": False}},
    ),
    "groq": (
        "GROQ_API_KEY",
        LLMProvider.GROQ,
        "groq/openai/gpt-oss-20b",
        None,
        {},
    ),
    # Model ids come from each account's own /models endpoint. Guessing them
    # from a provider's marketing page produced NotFoundError for every call,
    # which the batch parser then reported as unreadable responses.
    "cerebras": (
        "CEREBRAS_API_KEY",
        LLMProvider.OPENAI,
        "cerebras/gemma-4-31b",
        None,
        {},
    ),
    "together": (
        "TOGETHER_API_KEY",
        LLMProvider.OPENAI,
        "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
        None,
        {},
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

    env_var, provider, model, base_url, no_thinking = PRESETS[args.provider]
    if not os.getenv(env_var):
        raise SystemExit(f"{env_var} is not set — this benchmark calls a real API")

    spec = LLMSpec(
        provider=provider,
        model=model,
        temperature=0.0,
        # One word per row plus JSON scaffolding. A flat cap here truncates
        # every batched response, and a truncated batch is indistinguishable
        # from a provider outage — which is exactly how #... was found.
        max_tokens=args.max_tokens or max(256, args.batch_size * 48),
        base_url=base_url,
        input_cost_per_1k_tokens=Decimal("0.0001"),
        output_cost_per_1k_tokens=Decimal("0.0005"),
    )
    # LiteLLM passes extra_body straight through to the provider.
    if not args.thinking and no_thinking:
        spec.extra_params = {"extra_body": no_thinking}

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

    # A rate is only a rate if rows were answered. One run had every call
    # rejected with HTTP 400 and this script still reported "543 rows/s" and
    # "31 min per 1M rows" — wall clock divided by rows, whether or not any
    # work happened. That is the same silent-success shape the library was
    # just fixed for, and a benchmark that reports throughput for zero output
    # is worse than one that reports nothing.
    if lost:
        print(
            f"\n  NO THROUGHPUT NUMBER: {lost:,} of {args.rows:,} rows produced "
            f"no answer. Fix the run before reading any rate from it."
        )
        raise SystemExit(1)

    per_million_s = 1_000_000 / (args.rows / wall_s)
    print(
        f"\n  at this provider, model, batch size and concurrency: "
        f"{per_million_s / 60:,.0f} min per 1M rows"
    )


if __name__ == "__main__":
    main()
