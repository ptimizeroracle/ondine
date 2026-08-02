"""
Regression tests for silent whole-run failure (issue #187).

Before this fix a run whose LLM calls all failed under the default SKIP
policy still returned a DataFrame full of ``[SKIPPED]`` markers with
``success=True`` and ``skipped_rows == 0`` — the failure was invisible.
These tests pin the three parts of the fix:

1. A skipped LLM response increments ``context.skipped_rows`` (by the batch
   size), so partial failures show up in the metrics — and therefore in the
   CLI/progress surfaces that read those metrics.
2. The ``[SKIPPED]`` sentinel is treated as invalid output by the quality
   validator, so an all-skipped frame scores zero valid outputs.
3. A run that produced no usable output raises ``PipelineExecutionError``
   instead of returning a frame of markers with a green checkmark — while a
   run with *some* valid output still succeeds.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pandas as pd
import pytest

from ondine.adapters.containers.result_container import ResultContainerImpl
from ondine.adapters.llm_client import LLMClient
from ondine.api.pipeline import Pipeline
from ondine.api.pipeline_builder import PipelineBuilder
from ondine.core.exceptions import PipelineExecutionError
from ondine.core.models import (
    SKIPPED_OUTPUT_MARKER,
    CostEstimate,
    ExecutionResult,
    LLMResponse,
    ProcessingStats,
    RowMetadata,
)
from ondine.core.specifications import ErrorPolicy, LLMSpec
from ondine.orchestration.execution_context import ExecutionContext
from ondine.stages.llm_invocation_stage import LLMInvocationStage


class _AlwaysFailClient(LLMClient):
    """LLM client whose every call raises — drives a total-failure run."""

    def invoke(self, prompt: str, **kwargs: Any) -> Any:
        raise RuntimeError("simulated provider failure")

    async def ainvoke(self, prompt: str, **kwargs: Any) -> Any:
        raise RuntimeError("simulated provider failure")

    def structured_invoke(self, prompt: str, output_cls, **kwargs: Any) -> Any:
        raise RuntimeError("simulated provider failure")

    async def structured_invoke_async(
        self, prompt: str, output_cls, **kwargs: Any
    ) -> Any:
        raise RuntimeError("simulated provider failure")

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


def _skipped_response() -> LLMResponse:
    """The exact response the LLM stage emits for a skipped row."""
    return LLMResponse(
        text=SKIPPED_OUTPUT_MARKER,
        tokens_in=0,
        tokens_out=0,
        model="test",
        cost=Decimal("0.0"),
        latency_ms=0.0,
        metadata={"error": "boom", "action": "skipped"},
    )


def _ok_response() -> LLMResponse:
    return LLMResponse(
        text="positive",
        tokens_in=10,
        tokens_out=3,
        model="test",
        cost=Decimal("0.001"),
        latency_ms=1.0,
        metadata={},
    )


def _fresh_context() -> ExecutionContext:
    ctx = ExecutionContext(total_rows=4)
    ctx.intermediate_data["token_tracking"] = {"input_tokens": 0, "output_tokens": 0}
    return ctx


def _stage() -> LLMInvocationStage:
    spec = LLMSpec(provider="openai", model="test-model")
    return LLMInvocationStage(_AlwaysFailClient(spec), error_policy=ErrorPolicy.SKIP)


class TestSkipCounting:
    """The root cause: skipped rows were never counted."""

    def test_skipped_response_increments_skipped_rows(self):
        stage = _stage()
        ctx = _fresh_context()

        stage._update_context(ctx, 0, RowMetadata(row_index=0), _skipped_response())

        assert ctx.skipped_rows == 1

    def test_successful_response_does_not_increment_skipped_rows(self):
        stage = _stage()
        ctx = _fresh_context()

        stage._update_context(ctx, 0, RowMetadata(row_index=0), _ok_response())

        assert ctx.skipped_rows == 0

    def test_skipped_batch_counts_every_row_in_it(self):
        """A skipped batch loses batch_size rows, not one."""
        stage = _stage()
        ctx = _fresh_context()
        batch_meta = RowMetadata(
            row_index=0, custom={"is_batch": True, "batch_size": 3}
        )

        stage._update_context(ctx, 0, batch_meta, _skipped_response())

        assert ctx.skipped_rows == 3


class TestSkippedMarkerIsInvalidOutput:
    def test_all_skipped_cells_count_as_zero_valid_outputs(self):
        """The [SKIPPED] sentinel is a failure marker, not valid output."""
        data = ResultContainerImpl([{"output": SKIPPED_OUTPUT_MARKER}] * 4)
        result = ExecutionResult(
            data=data,
            metrics=ProcessingStats(4, 4, 0, 4, 1.0, 10.0),
            costs=CostEstimate(Decimal("0"), 0, 0, 0, 4),
            execution_id=uuid4(),
            start_time=datetime.now(),
        )

        quality = result.validate_output_quality(["output"])

        assert quality.valid_outputs == 0


class TestWholeRunGuard:
    def _pipeline(self, df: pd.DataFrame) -> Pipeline:
        spec = LLMSpec(provider="openai", model="test-model")
        return (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["review"], output_columns=["sentiment"])
            .with_prompt("Classify: {review}")
            .with_custom_llm_client(_AlwaysFailClient(spec))
            .with_error_policy("skip")
            .build()
        )

    def test_all_rows_failing_raises_instead_of_reporting_success(self):
        """End-to-end: a run where every row fails must raise, not succeed."""
        df = pd.DataFrame({"review": ["a", "b", "c"]})

        with pytest.raises(PipelineExecutionError):
            self._pipeline(df).execute()

    def test_guard_does_not_raise_when_some_outputs_are_valid(self):
        """A partially-successful run is still a success — never over-raise."""
        pipeline = self._pipeline(pd.DataFrame({"review": ["a"]}))
        data = ResultContainerImpl(
            [{"sentiment": "positive"}, {"sentiment": SKIPPED_OUTPUT_MARKER}]
        )
        result = ExecutionResult(
            data=data,
            metrics=ProcessingStats(2, 2, 0, 1, 1.0, 10.0),
            costs=CostEstimate(Decimal("0.001"), 13, 10, 3, 2),
            execution_id=uuid4(),
            start_time=datetime.now(),
        )

        # Must not raise: one valid output means the run produced something.
        pipeline._guard_produced_output(result, ["sentiment"])

    def test_empty_dataset_is_not_a_failure(self):
        """Zero rows is a no-op, not a failed run."""
        pipeline = self._pipeline(pd.DataFrame({"review": ["a"]}))
        result = ExecutionResult(
            data=ResultContainerImpl([]),
            metrics=ProcessingStats(0, 0, 0, 0, 0.0, 0.0),
            costs=CostEstimate(Decimal("0"), 0, 0, 0, 0),
            execution_id=uuid4(),
            start_time=datetime.now(),
        )

        pipeline._guard_produced_output(result, ["sentiment"])
