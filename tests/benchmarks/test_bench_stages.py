"""Stage-level benchmarks for Ondine pipeline.

Measures isolated performance of each pipeline stage to establish baselines
and detect regressions. Uses pytest-benchmark for statistical rigor.

Run with:
    pytest tests/benchmarks/test_bench_stages.py --benchmark-only
"""

import pytest

from ondine.stages.batch_aggregator_stage import BatchAggregatorStage
from ondine.stages.batch_disaggregator_stage import BatchDisaggregatorStage
from ondine.strategies.json_batch_strategy import JsonBatchStrategy

from .conftest import (
    make_aggregated_response_batches,
    make_prompt_batches,
)

# ---------------------------------------------------------------------------
# BatchAggregatorStage benchmarks
# ---------------------------------------------------------------------------


class TestBenchBatchAggregator:
    """Benchmark BatchAggregatorStage.process() at various scales.

    Regression this catches: aggregation loop slowing down due to
    added validation, logging, or strategy changes.
    """

    @pytest.mark.parametrize("n_rows", [100, 1000, 10000])
    def test_bench_aggregator_process(self, benchmark, n_rows):
        """Measure batch aggregation throughput at different scales.

        The aggregator groups n_rows into mega-prompts of batch_size=10.
        This is the sequential loop we plan to parallelize.
        """
        batches = make_prompt_batches(n_rows, batch_size=n_rows)
        stage = BatchAggregatorStage(batch_size=10, strategy=JsonBatchStrategy())

        result = benchmark(stage.process, batches, None)

        # Behavioral invariant: output count = ceil(n_rows / batch_size)
        expected_batches = (n_rows + 9) // 10
        assert len(result) == expected_batches

    def test_bench_aggregator_format_batch_only(self, benchmark, json_strategy):
        """Isolate format_batch cost — the inner hot function.

        Regression this catches: JsonBatchStrategy.format_batch becoming
        slower due to JSON serialization changes or added validation.
        """
        prompts = [f"Classify: Product {i} is great quality" for i in range(10)]

        result = benchmark(json_strategy.format_batch, prompts)

        assert "Process these 10 items" in result
        assert '"id":1' in result


# ---------------------------------------------------------------------------
# BatchDisaggregatorStage benchmarks
# ---------------------------------------------------------------------------


class TestBenchBatchDisaggregator:
    """Benchmark BatchDisaggregatorStage.process() at various scales.

    Regression this catches: disaggregation (JSON parsing + reconstruction)
    slowing down, especially the fast path (structured_result) vs slow path.
    """

    @pytest.mark.parametrize("n_batches", [100, 500])
    def test_bench_disaggregator_json_parse_path(self, benchmark, n_batches):
        """Measure disaggregation via JSON string parsing (slow path).

        Each batch has 10 rows → n_batches * 10 total rows disaggregated.
        """
        response_batches = make_aggregated_response_batches(
            n_batches=n_batches, rows_per_batch=10
        )
        stage = BatchDisaggregatorStage(strategy=JsonBatchStrategy())

        result = benchmark(stage.process, response_batches, None)

        # Each aggregated batch should produce rows_per_batch individual responses
        total_responses = sum(len(b.responses) for b in result)
        assert total_responses == n_batches * 10


# ---------------------------------------------------------------------------
# JsonBatchStrategy.parse_batch_response benchmarks
# ---------------------------------------------------------------------------


class TestBenchJsonParsing:
    """Benchmark JSON batch response parsing in isolation.

    Regression this catches: regex extraction or JSON parsing degrading
    with response size or nested structures.
    """

    @pytest.mark.parametrize("n_items", [10, 50, 100])
    def test_bench_parse_batch_response(self, benchmark, json_strategy, n_items):
        """Measure parse_batch_response at different result counts."""
        import json

        items = [{"id": i + 1, "result": f"positive_{i}"} for i in range(n_items)]
        response_text = json.dumps(items)

        result = benchmark(
            json_strategy.parse_batch_response,
            response_text,
            n_items,
        )

        assert len(result) == n_items
