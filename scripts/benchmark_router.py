#!/usr/bin/env python
"""Measure whether routing across providers actually adds their throughputs.

One provider is a ceiling you cannot buy past quickly: DeepSeek tops out near
660 rows/s on this account and Groq at 250,000 tokens/minute, and no amount of
concurrency moves either. Routing is the only lever that raises a ceiling
rather than approaching it — so the question is whether the sum is real.

It mostly is not, when the deployments differ in speed. Measured over 10,000
rows against DeepSeek (660 rows/s alone) plus Together (200 rows/s alone):

    latency-based-routing   437 rows/s     +5% over DeepSeek alone
    simple-shuffle          377 rows/s     -9%
    least-busy              364 rows/s    -13%

A fixed share of requests goes to the slow deployment and holds a worker for
longer, so the mix finishes behind the fast provider on its own. Routing sums
throughput only across deployments of *similar* speed — two accounts on one
provider, say. For a mixed pool its value is failover and cost control.

Two rules this script exists to enforce, both learned by getting them wrong:

1. Hold row count constant. A 2,000-row run is ~80 calls; connection-pool
   warm-up dominates it, and comparing that against a 10,000-row run made
   routing look 3.5x slower than the direct client when the two are identical
   (416.0 vs 416.5 rows/s at equal size).
2. Never report a rate over lost rows. A run where every call failed still
   divides rows by seconds and prints a confident number.

    uv run python scripts/benchmark_router.py --rows 10000
    uv run python scripts/benchmark_router.py --rows 10000 --strategy latency-based-routing
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ondine import PipelineBuilder  # noqa: E402
from ondine.core.specifications import LLMProvider, LLMSpec  # noqa: E402
from scripts.benchmark_throughput import synthetic_frame  # noqa: E402

#: Each entry is one deployment in the router's pool, plus the standalone
#: throughput measured for it earlier — that is what the sum is checked
#: against. `rpm` lets the router weight its dispatch instead of splitting work
#: evenly between a fast and a slow provider.
DEPLOYMENTS = {
    "deepseek": {
        "env": "DEEPSEEK_API_KEY",
        "measured_rows_per_s": 660.0,
        "litellm_params": {
            "model": "deepseek/deepseek-v4-flash",
            "api_base": "https://api.deepseek.com/v1",
            "rpm": 10_000,
            "extra_body": {"thinking": {"type": "disabled"}},
        },
    },
    "together": {
        "env": "TOGETHER_API_KEY",
        "measured_rows_per_s": 200.0,
        "litellm_params": {
            "model": "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
            "rpm": 3_000,
        },
    },
    "groq": {
        "env": "GROQ_API_KEY",
        "measured_rows_per_s": 41.0,
        "litellm_params": {
            "model": "groq/openai/gpt-oss-20b",
            # Groq's ceiling is 250,000 tokens/minute, which at ~100 tokens a
            # row is ~2,500 rows/minute. Told to the router as requests per
            # minute so it stops dispatching before Groq starts rejecting.
            "rpm": 100,
        },
    },
}

PROMPT = "Reply with exactly one word naming this product's category: {description}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=64)
    # Best of the three measured on a mixed pool; see the module docstring.
    parser.add_argument("--strategy", default="latency-based-routing")
    parser.add_argument(
        "--lean",
        action="store_true",
        help=(
            "Turn off the router's retry/cooldown bookkeeping, to separate "
            "the cost of routing from the cost of resilience."
        ),
    )
    parser.add_argument(
        "--providers",
        default="deepseek,together",
        help="Comma-separated pool. Every one needs its key in the environment.",
    )
    args = parser.parse_args()

    chosen = [name.strip() for name in args.providers.split(",") if name.strip()]
    missing = [n for n in chosen if not os.getenv(DEPLOYMENTS[n]["env"])]
    if missing:
        raise SystemExit(f"missing keys for: {', '.join(missing)}")

    model_list = [
        {
            "model_name": "classifier",
            "litellm_params": {
                **DEPLOYMENTS[name]["litellm_params"],
                "api_key": os.environ[DEPLOYMENTS[name]["env"]],
            },
        }
        for name in chosen
    ]
    expected = sum(DEPLOYMENTS[n]["measured_rows_per_s"] for n in chosen)

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            synthetic_frame(args.rows),
            input_columns=["description"],
            output_columns=["category"],
        )
        .with_prompt(PROMPT)
        .with_llm_spec(
            LLMSpec(
                provider=LLMProvider.OPENAI,
                model="classifier",
                temperature=0.0,
                max_tokens=max(256, args.batch_size * 48),
                # Reasoning off at the spec level, not only per deployment.
                # Deployment-level extra_body did not reach the call, so the
                # model was thinking on every request — which is why routing
                # looked 3.5x slower than the direct client rather than faster.
                extra_params={"extra_body": {"thinking": {"type": "disabled"}}},
                input_cost_per_1k_tokens=Decimal("0.0001"),
                output_cost_per_1k_tokens=Decimal("0.0005"),
            )
        )
        .with_router(
            model_list=model_list,
            routing_strategy=args.strategy,
            **(
                {"num_retries": 0, "allowed_fails": 0, "cooldown_time": 0}
                if args.lean
                else {}
            ),
        )
        .with_batch_size(args.batch_size)
        .with_concurrency(args.concurrency)
        .build()
    )

    print(
        f"pool={'+'.join(chosen)} strategy={args.strategy} rows={args.rows:,} "
        f"batch={args.batch_size} concurrency={args.concurrency}"
    )

    started = time.perf_counter()
    result = pipeline.execute()
    wall_s = time.perf_counter() - started

    lost = result.metrics.skipped_rows + result.metrics.failed_rows
    rate = args.rows / wall_s

    print(f"\n  wall clock      {wall_s:8.2f}s")
    print(f"  rows lost       {lost:8,}")
    print(f"  cost            ${result.costs.total_cost:.4f}")

    if lost:
        print(
            f"\n  NO THROUGHPUT NUMBER: {lost:,} of {args.rows:,} rows produced no "
            f"answer. A rate measured over lost rows is not a rate."
        )
        raise SystemExit(1)

    print(f"  throughput      {rate:8,.0f} rows/s")
    print(f"  sum of parts    {expected:8,.0f} rows/s  ({' + '.join(chosen)})")
    print(f"  routing yield   {rate / expected:8.0%}")
    print(f"\n  1M rows at this rate: {1_000_000 / rate / 60:,.0f} min")


if __name__ == "__main__":
    main()
