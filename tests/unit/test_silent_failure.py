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

    def test_deliberately_blanked_output_is_not_a_failure(self):
        """A grounding/validation filter may legitimately blank every cell.

        No rows errored and tokens were consumed, so the pipeline worked —
        the data just did not survive the filter. Raising here would turn a
        working feature into an error.
        """
        pipeline = self._pipeline(pd.DataFrame({"review": ["a"]}))
        result = ExecutionResult(
            data=ResultContainerImpl([{"sentiment": ""}, {"sentiment": ""}]),
            metrics=ProcessingStats(2, 2, 0, 0, 1.0, 10.0),
            costs=CostEstimate(Decimal("0.002"), 120, 100, 20, 2),
            execution_id=uuid4(),
            start_time=datetime.now(),
        )

        pipeline._guard_produced_output(result, ["sentiment"])

    def test_zero_tokens_with_no_output_is_a_failure(self):
        """Nothing came back from the provider at all — that is a failure."""
        pipeline = self._pipeline(pd.DataFrame({"review": ["a"]}))
        result = ExecutionResult(
            data=ResultContainerImpl([{"sentiment": None}, {"sentiment": None}]),
            metrics=ProcessingStats(2, 2, 0, 0, 1.0, 10.0),
            costs=CostEstimate(Decimal("0"), 0, 0, 0, 2),
            execution_id=uuid4(),
            start_time=datetime.now(),
        )

        with pytest.raises(PipelineExecutionError):
            pipeline._guard_produced_output(result, ["sentiment"])

    def test_internal_retry_pass_is_exempt(self):
        """A retry that recovers nothing must not abort the retry loop."""
        pipeline = self._pipeline(pd.DataFrame({"review": ["a"]}))
        pipeline._is_internal_retry = True
        result = ExecutionResult(
            data=ResultContainerImpl([{"sentiment": SKIPPED_OUTPUT_MARKER}]),
            metrics=ProcessingStats(1, 1, 0, 1, 1.0, 10.0),
            costs=CostEstimate(Decimal("0"), 0, 0, 0, 1),
            execution_id=uuid4(),
            start_time=datetime.now(),
        )

        pipeline._guard_produced_output(result, ["sentiment"])


class TestIsCompleteTellsCoverageFromMerelyFinishing:
    """`is_complete` is the honest coverage signal `success` cannot be (#254).

    The default skip policy tolerates lost rows, so `success` stays True even
    when the frame has holes — that is the contract, not a bug. `is_complete`
    exists so a caller can tell "every row made it" apart from "the run merely
    finished". A regression that let `is_complete` return True while rows were
    skipped would put silent partial loss right back where #254 found it.
    """

    @staticmethod
    def _result(*, skipped: int, failed: int, total: int = 10) -> ExecutionResult:
        processed = total - skipped - failed
        return ExecutionResult(
            data=ResultContainerImpl([{"output": "x"}] * processed),
            metrics=ProcessingStats(total, processed, failed, skipped, 0.0, 0.0),
            costs=CostEstimate(Decimal("0"), 0, 0, 0, total),
        )

    @pytest.mark.parametrize(
        ("skipped", "failed", "expected_complete"),
        [
            (0, 0, True),  # every row produced output
            (1, 0, False),  # one skipped → a hole
            (0, 1, False),  # one failed outright → a hole
            (2, 3, False),  # both kinds of loss
        ],
    )
    def test_is_complete_is_true_only_when_no_row_was_lost(
        self, skipped, failed, expected_complete
    ):
        result = self._result(skipped=skipped, failed=failed)
        assert result.is_complete is expected_complete

    def test_success_and_is_complete_diverge_on_a_tolerated_skip(self):
        """The core #254 decision: a skipped run succeeds but is not complete.

        This is the one assertion that pins the semantic choice — flipping
        `success` to False here (the rejected option B) would break the
        default skip policy; the truth lives in `is_complete` instead.
        """
        result = ExecutionResult(
            data=ResultContainerImpl([{"output": "x"}] * 9),
            metrics=ProcessingStats(10, 9, 0, 1, 0.0, 0.0),
            costs=CostEstimate(Decimal("0"), 0, 0, 0, 10),
            success=True,
        )
        assert result.success is True
        assert result.is_complete is False
