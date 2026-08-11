#!/usr/bin/env python
"""Measure how fast ondine itself can move rows.

The question this answers is "how many rows per second does ondine cost you",
not "how fast is the provider". Those are different numbers owned by different
people: provider latency belongs to OpenAI and changes with tier, model and
time of day, while the orchestration around it — templating, batching, async
scheduling, parsing, cost accounting, checkpointing, DataFrame assembly — is
ondine's, and it is the only part this repository can improve or regress.

So the provider here answers instantly from the prompt's own content. What is
left on the clock is ondine. That is the ceiling: no real run goes faster.

A run also reports peak RSS, because throughput that only holds while the
whole dataset fits in memory is not throughput for the 5M-row case.

Usage:

    uv run python scripts/benchmark_throughput.py --rows 100000
    uv run python scripts/benchmark_throughput.py --rows 5000000 --batch-size 50
    uv run python scripts/benchmark_throughput.py --rows 100000 --latency-ms 200
"""

from __future__ import annotations

import argparse
import random
import resource
import statistics
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ondine import PipelineBuilder  # noqa: E402
from ondine.adapters.llm_client import LLMClient  # noqa: E402
from ondine.core.models import LLMResponse  # noqa: E402
from ondine.core.specifications import LLMProvider, LLMSpec  # noqa: E402

# Row text long enough to be worth tokenising. Product descriptions and support
# tickets — the workloads this library is used for — sit around here; a 20-char
# row would measure an empty template rather than a realistic one.
WORD_POOL = [
    "wireless",
    "bluetooth",
    "headphones",
    "noise",
    "cancelling",
    "battery",
    "charge",
    "stainless",
    "steel",
    "kitchen",
    "knife",
    "ergonomic",
    "office",
    "chair",
    "lumbar",
    "support",
    "cotton",
    "shirt",
    "regular",
    "fit",
    "ceramic",
    "mug",
    "dishwasher",
    "safe",
    "leather",
    "wallet",
    "card",
    "slots",
    "usb",
    "cable",
    "braided",
    "nylon",
    "fast",
    "charging",
    "water",
    "bottle",
    "insulated",
    "vacuum",
]


def synthetic_frame(rows: int, seed: int = 0) -> pd.DataFrame:
    """`rows` rows of realistic-length text, deterministic for a given seed."""
    rng = random.Random(seed)  # noqa: S311  # nosec B311 - benchmark rows, not secrets
    return pd.DataFrame(
        {
            "description": [
                " ".join(rng.choices(WORD_POOL, k=rng.randint(20, 60)))
                for _ in range(rows)
            ]
        }
    )


#: Rows generated per slice when writing the input file. Keeps the *writing*
#: of a 5M-row dataset off the memory budget the benchmark is measuring.
GENERATION_SLICE = 250_000


def _prepared_input(rows: int, seed: int, fmt: str) -> Path:
    """A dataset of `rows` synthetic rows, generated once and cached on disk.

    Parquet is written with an explicit ParquetWriter rather than repeated
    `to_parquet` calls: each call would produce a *separate* file, so appending
    means appending row groups to one open writer. Same reason the CSV path
    appends — neither format may require the whole dataset in memory at once.
    """
    path = Path(__file__).parent / ".benchmark-data" / f"rows-{rows}-seed-{seed}.{fmt}"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        writer = None
        try:
            written = 0
            while written < rows:
                take = min(GENERATION_SLICE, rows - written)
                table = pa.Table.from_pandas(
                    synthetic_frame(take, seed=seed + written), preserve_index=False
                )
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="snappy")
                writer.write_table(table)
                written += take
        finally:
            if writer is not None:
                writer.close()
        return path

    written = 0
    while written < rows:
        take = min(GENERATION_SLICE, rows - written)
        synthetic_frame(take, seed=seed + written).to_csv(
            path, mode="a", header=written == 0, index=False
        )
        written += take
    return path


class InstantClient(LLMClient):
    """A provider with no latency and no memory of what it was asked.

    Deliberately not the conformance harness's LedgerClient: recording every
    call is what makes that class useful in tests and useless here, since a
    ledger of 5,000,000 entries would be measuring its own bookkeeping.

    `latency_ms` simulates a real provider when the question is about
    concurrency rather than overhead.
    """

    def __init__(self, spec: LLMSpec, latency_ms: float = 0.0) -> None:
        super().__init__(spec)
        self._latency_s = latency_ms / 1000.0
        self.calls = 0

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.calls += 1
        if self._latency_s:
            time.sleep(self._latency_s)
        return LLMResponse(
            text=self._answer(prompt),
            tokens_in=64,
            tokens_out=8,
            model=self.model,
            cost=Decimal("0.00001"),
            latency_ms=self._latency_s * 1000,
        )

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.calls += 1
        if self._latency_s:
            import asyncio

            await asyncio.sleep(self._latency_s)
        return LLMResponse(
            text=self._answer(prompt),
            tokens_in=64,
            tokens_out=8,
            model=self.model,
            cost=Decimal("0.00001"),
            latency_ms=self._latency_s * 1000,
        )

    def structured_invoke(self, prompt: str, output_cls: Any, **kwargs: Any):
        return self.invoke(prompt)

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def _answer(prompt: str) -> str:
        """A batched prompt needs a batched answer, or the parser rejects it."""
        import json
        import re

        match = re.search(r"^\s*(\[.*?\])\s*$", prompt, re.MULTILINE | re.DOTALL)
        if match:
            try:
                items = json.loads(match.group(1))
            except json.JSONDecodeError:
                return "electronics"
            return json.dumps(
                [{"id": item["id"], "result": "electronics"} for item in items]
            )
        return "electronics"


def peak_rss_mb() -> float:
    """Peak resident set size. ru_maxrss is bytes on macOS, KiB on Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def run(
    rows: int,
    batch_size: int,
    concurrency: int,
    latency_ms: float,
    seed: int,
    chunk_size: int = 0,
    fmt: str = "parquet",
) -> dict[str, float]:
    spec = LLMSpec(
        provider=LLMProvider.OPENAI,
        model="benchmark-1",
        temperature=0.0,
        input_cost_per_1k_tokens=Decimal("0.0001"),
        output_cost_per_1k_tokens=Decimal("0.0001"),
    )
    client = InstantClient(spec, latency_ms=latency_ms)

    build_start = time.perf_counter()
    # Reading from disk, not from a DataFrame handed in whole.
    #
    # Building the input frame in this process would put every row in memory
    # before the pipeline starts, and that cost would land in peak RSS as if
    # ondine had spent it. At 1M rows the input frame alone is most of a
    # gigabyte — enough to hide whether streaming is bounding memory at all,
    # which is the only question worth asking at 5M.
    source = _prepared_input(rows, seed, fmt)
    reader = PipelineBuilder.create()
    builder = (
        (
            reader.from_parquet(
                str(source),
                input_columns=["description"],
                output_columns=["category"],
            )
            if fmt == "parquet"
            else reader.from_csv(
                str(source),
                input_columns=["description"],
                output_columns=["category"],
            )
        )
        .with_prompt("Categorise this product: {description}")
        .with_custom_llm_client(client)
        .with_batch_size(batch_size)
        .with_concurrency(concurrency)
    )
    if chunk_size:
        builder = builder.with_streaming(chunk_size=chunk_size)
    pipeline = builder.build()
    setup_s = time.perf_counter() - build_start

    run_start = time.perf_counter()
    if chunk_size:
        # Streaming yields a result per chunk. Only the answers are kept: the
        # point of streaming is that the whole output never exists at once, and
        # concatenating it here would rebuild exactly what it avoids.
        produced = 0
        for chunk in pipeline.execute_stream():
            produced += int((chunk.to_pandas()["category"] == "electronics").sum())
        elapsed_s = time.perf_counter() - run_start
    else:
        result = pipeline.execute()
        elapsed_s = time.perf_counter() - run_start
        produced = int((result.to_pandas()["category"] == "electronics").sum())
    # A throughput number for a run that dropped rows is worthless — check
    # before reporting, not after publishing.
    if produced != rows:
        raise SystemExit(
            f"benchmark invalid: {produced}/{rows} rows produced a real answer"
        )

    return {
        "rows": rows,
        "setup_s": setup_s,
        "elapsed_s": elapsed_s,
        "rows_per_s": rows / elapsed_s,
        "api_calls": client.calls,
        "peak_rss_mb": peak_rss_mb(),
        "input_mb": source.stat().st_size / (1024 * 1024),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument(
        "--latency-ms",
        type=float,
        default=0.0,
        help="Simulated provider latency per call. 0 measures ondine alone.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Stream in chunks of this many rows. 0 loads everything at once.",
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Input file format. Parquet is what a 5M-row dataset should be in.",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(
        f"rows={args.rows:,} batch_size={args.batch_size} "
        f"concurrency={args.concurrency} latency_ms={args.latency_ms} "
        f"chunk_size={args.chunk_size or 'off'} format={args.format}"
    )

    runs = [
        run(
            args.rows,
            args.batch_size,
            args.concurrency,
            args.latency_ms,
            args.seed + i,
            chunk_size=args.chunk_size,
            fmt=args.format,
        )
        for i in range(args.repeat)
    ]

    for index, measured in enumerate(runs, start=1):
        print(
            f"  run {index}: {measured['elapsed_s']:8.2f}s  "
            f"{measured['rows_per_s']:10,.0f} rows/s  "
            f"{measured['api_calls']:>9,} calls  "
            f"peak RSS {measured['peak_rss_mb']:>5,.0f} MB  "
            f"input {measured['input_mb']:,.0f} MB"
        )

    rates = [measured["rows_per_s"] for measured in runs]
    best = max(rates)
    print(f"\nmedian {statistics.median(rates):,.0f} rows/s   best {best:,.0f} rows/s")
    print(
        f"5,000,000 rows at the median: {5_000_000 / statistics.median(rates) / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
