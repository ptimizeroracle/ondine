"""End-to-end pipeline benchmarks for Ondine.

Measures full pipeline execution with mock LLM to establish wall-clock
baselines and per-stage timing breakdown.

Run with:
    pytest tests/benchmarks/test_bench_pipeline_e2e.py --benchmark-only -s
"""

import time

import pytest

from ondine.orchestration.execution_context import ExecutionContext
from ondine.stages.batch_aggregator_stage import BatchAggregatorStage
from ondine.stages.batch_disaggregator_stage import BatchDisaggregatorStage
from ondine.stages.llm_invocation_stage import LLMInvocationStage
from ondine.strategies.json_batch_strategy import JsonBatchStrategy

from .conftest import BenchmarkMockLLMClient, make_prompt_batches


class TestBenchPipelineE2E:
    """End-to-end pipeline benchmarks with per-stage timing.

    Regression this catches: overall pipeline throughput degrading,
    or a specific stage becoming the bottleneck unexpectedly.
    """

    @pytest.mark.parametrize("n_rows", [100, 1000])
    def test_bench_pipeline_stages_sequential(self, n_rows):
        """Run core pipeline stages in sequence, timing each one.

        Stages: PromptFormat → BatchAggregate → LLMInvoke → BatchDisaggregate

        This is the exact execution model of Pipeline.execute() today.
        Per-stage timing reveals where time is actually spent.
        """
        batch_size = 10
        concurrency = 10

        # --- Stage 1: Prompt formatting (simulate) ---
        batches = make_prompt_batches(n_rows, batch_size=n_rows)

        # --- Stage 2: Batch aggregation ---
        aggregator = BatchAggregatorStage(
            batch_size=batch_size, strategy=JsonBatchStrategy()
        )
        t_agg_start = time.perf_counter()
        aggregated = aggregator.process(batches, None)
        t_agg = time.perf_counter() - t_agg_start

        expected_mega_batches = (n_rows + batch_size - 1) // batch_size
        assert len(aggregated) == expected_mega_batches

        # --- Stage 3: LLM invocation (mock, 10ms delay) ---
        client = BenchmarkMockLLMClient(delay_ms=10.0)
        llm_stage = LLMInvocationStage(llm_client=client, concurrency=concurrency)
        t_llm_start = time.perf_counter()
        context = ExecutionContext(total_rows=n_rows)
        responses = llm_stage.process(aggregated, context)
        t_llm = time.perf_counter() - t_llm_start

        assert len(responses) == expected_mega_batches

        # --- Stage 4: Batch disaggregation ---
        disaggregator = BatchDisaggregatorStage(strategy=JsonBatchStrategy())
        t_dis_start = time.perf_counter()
        disaggregator.process(responses, None)  # noqa: F841
        t_dis = time.perf_counter() - t_dis_start

        # --- Report ---
        total = t_agg + t_llm + t_dis
        print(f"\n{'=' * 60}")
        print(
            f"Pipeline E2E: {n_rows} rows, batch_size={batch_size}, concurrency={concurrency}"
        )
        print(f"  Aggregation:    {t_agg:.4f}s ({t_agg / total * 100:.1f}%)")
        print(f"  LLM Invocation: {t_llm:.4f}s ({t_llm / total * 100:.1f}%)")
        print(f"  Disaggregation: {t_dis:.4f}s ({t_dis / total * 100:.1f}%)")
        print(f"  Total:          {total:.4f}s")
        print(f"  Throughput:     {n_rows / total:.0f} rows/sec")
        print(f"{'=' * 60}")

        # Sanity: pipeline should complete in reasonable time
        # 1000 rows with 10ms mock delay, concurrency=10 → ~1s LLM + overhead
        max_seconds = max(5.0, n_rows * 0.02)
        assert total < max_seconds, (
            f"Pipeline took {total:.2f}s for {n_rows} rows (max allowed: {max_seconds}s)"
        )

    def test_bench_aggregation_dominance(self):
        """Prove that aggregation is the bottleneck at scale.

        At 10K rows with fast mock LLM, aggregation should consume
        the majority of wall-clock time. This motivates parallelization.
        """
        n_rows = 5000
        batch_size = 10

        batches = make_prompt_batches(n_rows, batch_size=n_rows)
        aggregator = BatchAggregatorStage(
            batch_size=batch_size, strategy=JsonBatchStrategy()
        )

        t_agg_start = time.perf_counter()
        aggregated = aggregator.process(batches, None)
        t_agg = time.perf_counter() - t_agg_start

        # Fast LLM (1ms delay, high concurrency)
        client = BenchmarkMockLLMClient(delay_ms=1.0)
        llm_stage = LLMInvocationStage(llm_client=client, concurrency=20)

        t_llm_start = time.perf_counter()
        context = ExecutionContext(total_rows=n_rows)
        llm_stage.process(aggregated, context)
        t_llm = time.perf_counter() - t_llm_start

        print(f"\n{'=' * 60}")
        print(f"Bottleneck analysis: {n_rows} rows")
        print(f"  Aggregation: {t_agg:.4f}s")
        print(f"  LLM (1ms mock): {t_llm:.4f}s")
        print(f"  Ratio (agg/llm): {t_agg / max(t_llm, 0.001):.1f}x")
        print(f"{'=' * 60}")

        # Aggregation should be meaningful portion of total time
        # (This documents the current state, not a pass/fail gate)
        total = t_agg + t_llm
        agg_pct = t_agg / total * 100
        print(f"  Aggregation is {agg_pct:.0f}% of pipeline time")
