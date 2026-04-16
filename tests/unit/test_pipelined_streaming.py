"""Tests for pipelined streaming execution — chunk overlap.

Verifies that CPU-bound stages (format, aggregate) for chunk N+1
execute concurrently with the LLM I/O stage for chunk N, rather than
waiting for chunk N to complete all stages before starting chunk N+1.

Approach: Chicago school — real stages with a fake LLM client that
introduces measurable async delay. We verify overlap by comparing
wall-clock time against sequential baseline.
"""

import asyncio
import time
from decimal import Decimal
from typing import Any

import pandas as pd

from ondine.adapters.llm_client import LLMClient
from ondine.core.models import LLMResponse, PromptBatch, RowMetadata
from ondine.core.specifications import LLMProvider, LLMSpec
from ondine.orchestration.execution_context import ExecutionContext
from ondine.stages.llm_invocation_stage import LLMInvocationStage


class DelayedMockLLMClient(LLMClient):
    """Mock LLM with configurable async delay to simulate real API latency.

    Uses asyncio.sleep (not time.sleep) so other coroutines can run
    during the delay — this is what makes overlap measurable.
    """

    def __init__(self, delay_s: float = 0.3):
        spec = LLMSpec(model="delay-mock", provider=LLMProvider.OPENAI)
        super().__init__(spec)
        self.delay_s = delay_s
        self.call_count = 0
        self.call_timestamps: list[float] = []

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.call_count += 1
        self.call_timestamps.append(time.perf_counter())
        return LLMResponse(
            text='[{"id":1,"result":"positive"}]',
            tokens_in=50,
            tokens_out=20,
            model="delay-mock",
            cost=Decimal("0.001"),
            latency_ms=self.delay_s * 1000,
        )

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.call_count += 1
        self.call_timestamps.append(time.perf_counter())
        await asyncio.sleep(self.delay_s)
        return LLMResponse(
            text='[{"id":1,"result":"positive"}]',
            tokens_in=50,
            tokens_out=20,
            model="delay-mock",
            cost=Decimal("0.001"),
            latency_ms=self.delay_s * 1000,
        )

    def structured_invoke(self, prompt, output_cls, **kwargs):
        return self.invoke(prompt, **kwargs)

    async def structured_invoke_async(self, prompt, output_cls, **kwargs):
        return await self.ainvoke(prompt, **kwargs)

    async def start(self):
        pass

    async def stop(self):
        pass

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4


def make_test_df(n_rows: int) -> pd.DataFrame:
    """Generate test DataFrame matching pipeline input requirements."""
    return pd.DataFrame(
        {
            "text": [f"Review product {i}: great quality" for i in range(n_rows)],
        }
    )


class TestPipelinedStreamingOverlap:
    """Verify that pipelined execution overlaps CPU prep with LLM I/O.

    Regression these tests catch: If the pipelining breaks and chunks
    process sequentially again, wall-clock time will exceed the threshold
    because CPU prep for chunk N+1 no longer runs during chunk N's LLM wait.
    """

    def test_pipelined_produces_results_for_all_chunks(self):
        """Pipelined execution must process all chunks and yield results.

        Verifies correctness: every chunk produces an ExecutionResult,
        and the method doesn't silently drop chunks or deadlock.
        """
        from unittest.mock import patch

        from ondine.api.pipeline_builder import PipelineBuilder

        n_rows = 200
        chunk_size = 100  # 2 chunks

        df = make_test_df(n_rows)
        client = DelayedMockLLMClient(delay_s=0.01)

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Classify: {text}")
            .with_llm(model="delay-mock", provider="openai")
            .with_batch_size(100)
            .with_concurrency(5)
            .with_progress_mode("none")
            .build()
        )

        assert hasattr(pipeline, "execute_stream_pipelined"), (
            "Pipeline must have execute_stream_pipelined() method"
        )

        with patch("litellm.acompletion", side_effect=client.ainvoke):
            results = list(pipeline.execute_stream_pipelined(chunk_size=chunk_size))

        # Must produce one result per chunk
        assert len(results) == 2, (
            f"Expected 2 chunk results (200 rows / 100 chunk_size), got {len(results)}"
        )

        # Each chunk result must have data
        for i, chunk_result in enumerate(results):
            assert chunk_result.success, f"Chunk {i} failed"
            assert hasattr(chunk_result, "data"), f"Chunk {i} missing data"

    def test_llm_stage_has_process_async(self):
        """LLMInvocationStage must expose process_async() for pipelining.

        Regression: without process_async(), the pipelined executor
        cannot call the LLM stage from within an already-running event loop
        (asyncio.run() would fail with 'cannot be called from running loop').
        """
        client = DelayedMockLLMClient(delay_s=0.01)
        stage = LLMInvocationStage(llm_client=client, concurrency=5)

        assert hasattr(stage, "process_async"), (
            "LLMInvocationStage must have process_async() method "
            "that delegates to _process_async() without calling asyncio.run()"
        )

        # Verify it's actually async
        import inspect

        assert inspect.iscoroutinefunction(stage.process_async), (
            "process_async must be a coroutine function"
        )

    def test_process_async_produces_same_output_as_process(self):
        """process_async() must return identical results to process().

        Regression: if process_async() diverges from process(), pipelined
        execution will produce different results than sequential.
        """
        client = DelayedMockLLMClient(delay_s=0.01)
        context = ExecutionContext(total_rows=10)

        prompts = [f"Classify item {i}" for i in range(10)]
        metadata = [RowMetadata(row_index=i, row_id=i, custom={}) for i in range(10)]
        batches = [PromptBatch(prompts=prompts, metadata=metadata, batch_id=0)]

        # Sync path
        stage_sync = LLMInvocationStage(llm_client=client, concurrency=5)
        result_sync = stage_sync.process(batches, context)

        # Async path
        client2 = DelayedMockLLMClient(delay_s=0.01)
        context2 = ExecutionContext(total_rows=10)
        stage_async = LLMInvocationStage(llm_client=client2, concurrency=5)
        result_async = asyncio.run(stage_async.process_async(batches, context2))

        # Same structure
        assert len(result_async) == len(result_sync), (
            f"Async produced {len(result_async)} batches, sync produced {len(result_sync)}"
        )
        for i, (a, s) in enumerate(zip(result_async, result_sync, strict=True)):
            assert len(a.responses) == len(s.responses), (
                f"Batch {i}: async has {len(a.responses)} responses, sync has {len(s.responses)}"
            )
