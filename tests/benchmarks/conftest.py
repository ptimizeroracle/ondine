"""Shared fixtures for benchmark tests.

Provides realistic data generators matching Ondine's internal types.
All fixtures produce data shapes identical to what pipeline stages
receive during real execution.

Requires pytest-benchmark (install via: uv sync --group perf).
Entire directory skipped when pytest-benchmark is not available.
"""

import asyncio
from decimal import Decimal
from typing import Any

import pytest

# Skip all benchmark tests if pytest-benchmark is not installed
pytest.importorskip("pytest_benchmark")

import pytest

from ondine.adapters.llm_client import LLMClient
from ondine.core.models import LLMResponse, PromptBatch, ResponseBatch, RowMetadata
from ondine.core.specifications import LLMProvider, LLMSpec
from ondine.strategies.json_batch_strategy import JsonBatchStrategy
from ondine.strategies.models import BatchMetadata
from ondine.utils.rate_limiter import RateLimiter

# ---------------------------------------------------------------------------
# Data generators (functions, not fixtures — so benchmarks can call with args)
# ---------------------------------------------------------------------------


def make_prompt_batches(
    n_rows: int,
    batch_size: int = 1,
    prompt_template: str = "Classify this product review: {text}",
) -> list[PromptBatch]:
    """Generate realistic PromptBatch list as PromptFormatterStage would produce.

    Args:
        n_rows: Total number of rows (prompts).
        batch_size: Rows per PromptBatch (1 = unbatched, >1 = pre-chunked).
        prompt_template: Template used to generate prompts.

    Returns:
        List of PromptBatch objects ready for BatchAggregatorStage.
    """
    prompts = [
        prompt_template.format(text=f"Product {i} is great quality and fast shipping")
        for i in range(n_rows)
    ]
    metadata = [
        RowMetadata(
            row_index=i,
            row_id=i,
            custom={"system_message": "You are a product classifier."},
        )
        for i in range(n_rows)
    ]

    # Group into batches of batch_size
    batches = []
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        batches.append(
            PromptBatch(
                prompts=prompts[start:end],
                metadata=metadata[start:end],
                batch_id=start // batch_size,
            )
        )
    return batches


def make_response_batches(
    n_rows: int,
    response_text: str = '{"result": "positive", "confidence": 0.95}',
) -> list[ResponseBatch]:
    """Generate realistic ResponseBatch list as LLMInvocationStage would produce.

    Args:
        n_rows: Total number of responses.
        response_text: Text for each LLMResponse.

    Returns:
        List of ResponseBatch objects ready for BatchDisaggregatorStage.
    """
    responses = [
        LLMResponse(
            text=response_text,
            tokens_in=50,
            tokens_out=20,
            model="mock-model",
            cost=Decimal("0.001"),
            latency_ms=150.0,
        )
        for _ in range(n_rows)
    ]
    metadata = [RowMetadata(row_index=i, row_id=i, custom={}) for i in range(n_rows)]
    return [
        ResponseBatch(
            responses=responses,
            metadata=metadata,
            tokens_used=sum(r.tokens_in + r.tokens_out for r in responses),
            cost=sum(r.cost for r in responses),
            batch_id=0,
            latencies_ms=[r.latency_ms for r in responses],
        )
    ]


def make_aggregated_response_batches(
    n_batches: int,
    rows_per_batch: int = 10,
) -> list[ResponseBatch]:
    """Generate ResponseBatch list that came from BatchAggregator (mega-prompts).

    Each batch has 1 response representing rows_per_batch original rows.
    """
    import json

    batches = []
    for batch_idx in range(n_batches):
        row_start = batch_idx * rows_per_batch
        row_ids = list(range(row_start, row_start + rows_per_batch))

        # Simulate JSON array response from LLM
        items = [
            {"id": i + 1, "result": f"positive_{row_start + i}"}
            for i in range(rows_per_batch)
        ]
        response_json = json.dumps(items)

        batch_metadata = BatchMetadata(
            original_count=rows_per_batch,
            row_ids=row_ids,
            prompt_template=None,
        )

        meta = RowMetadata(
            row_index=row_start,
            row_id=row_start,
            custom={
                "is_batch": True,
                "batch_size": rows_per_batch,
                "batch_metadata": batch_metadata.model_dump(),
                "system_message": "You are a classifier.",
            },
        )

        response = LLMResponse(
            text=response_json,
            tokens_in=200,
            tokens_out=100,
            model="mock-model",
            cost=Decimal("0.005"),
            latency_ms=500.0,
        )

        batches.append(
            ResponseBatch(
                responses=[response],
                metadata=[meta],
                tokens_used=300,
                cost=Decimal("0.005"),
                batch_id=batch_idx,
                latencies_ms=[500.0],
            )
        )
    return batches


# ---------------------------------------------------------------------------
# Mock LLM client with configurable async delay
# ---------------------------------------------------------------------------


class BenchmarkMockLLMClient(LLMClient):
    """Mock LLM client with configurable latency for benchmarking concurrency."""

    def __init__(self, delay_ms: float = 50.0):
        spec = LLMSpec(model="bench-mock", provider=LLMProvider.OPENAI)
        super().__init__(spec)
        self.delay_ms = delay_ms
        self.call_count = 0

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            text='{"result": "positive"}',
            tokens_in=len(prompt) // 4,
            tokens_out=10,
            model="bench-mock",
            cost=Decimal("0.001"),
            latency_ms=self.delay_ms,
        )

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.call_count += 1
        await asyncio.sleep(self.delay_ms / 1000.0)
        return LLMResponse(
            text='{"result": "positive"}',
            tokens_in=len(prompt) // 4,
            tokens_out=10,
            model="bench-mock",
            cost=Decimal("0.001"),
            latency_ms=self.delay_ms,
        )

    def structured_invoke(self, prompt: str, output_cls, **kwargs: Any) -> LLMResponse:
        return self.invoke(prompt, **kwargs)

    async def structured_invoke_async(
        self, prompt: str, output_cls, **kwargs: Any
    ) -> LLMResponse:
        return await self.ainvoke(prompt, **kwargs)

    async def start(self):
        pass

    async def stop(self):
        pass

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def json_strategy():
    """JsonBatchStrategy instance for aggregation benchmarks."""
    return JsonBatchStrategy()


@pytest.fixture
def rate_limiter_60rpm():
    """RateLimiter configured at 60 RPM (1 req/sec)."""
    return RateLimiter(requests_per_minute=60)


@pytest.fixture
def rate_limiter_600rpm():
    """RateLimiter configured at 600 RPM (10 req/sec)."""
    return RateLimiter(requests_per_minute=600)


@pytest.fixture
def bench_llm_client():
    """Mock LLM client with 50ms simulated latency."""
    return BenchmarkMockLLMClient(delay_ms=50.0)
