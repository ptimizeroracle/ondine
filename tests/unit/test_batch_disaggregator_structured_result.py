"""
Unit tests for batch disaggregation with structured_result preservation.

Regression these tests catch:

1. Disaggregator fast path: when structured_result carries a valid Pydantic
   batch, the disaggregator must extract all items without falling back to
   JSON text parsing (which triggers PartialParseError and cascading retries).

2. Item count mismatch: when the LLM returns fewer items than expected, the
   disaggregator must pad missing positions with "null" and still produce the
   correct total response count, preventing metadata/response length mismatch.

3. Restore path: _restore_completed_response_batches must propagate
   _structured_result from in-memory completed_responses records so the
   disaggregator's Pydantic fast path is available after checkpoint restore.
"""

import json
from decimal import Decimal
from unittest.mock import Mock
from uuid import uuid4

from pydantic import BaseModel, Field

from ondine.core.models import LLMResponse, ResponseBatch, RowMetadata
from ondine.stages.batch_disaggregator_stage import BatchDisaggregatorStage
from ondine.strategies.json_batch_strategy import JsonBatchStrategy


class SwapResult(BaseModel):
    score: int = Field(ge=0, le=5)
    reason: str


class SwapBatchItem(BaseModel):
    id: int
    result: SwapResult


class SwapBatch(BaseModel):
    items: list[SwapBatchItem]


def _make_batch(n_items: int, expected_count: int | None = None) -> ResponseBatch:
    """Build a ResponseBatch with a structured_result Pydantic object."""
    if expected_count is None:
        expected_count = n_items

    row_ids = list(range(100, 100 + expected_count))
    items = [
        SwapBatchItem(id=i + 1, result=SwapResult(score=(i % 5) + 1, reason=f"r{i}"))
        for i in range(n_items)
    ]
    pydantic_batch = SwapBatch(items=items)

    return ResponseBatch(
        responses=[
            LLMResponse(
                text=pydantic_batch.model_dump_json(),
                tokens_in=50,
                tokens_out=80,
                model="azure/gpt-5-nano",
                cost=Decimal("0.001"),
                latency_ms=200.0,
                structured_result=pydantic_batch,
            )
        ],
        metadata=[
            RowMetadata(
                row_index=row_ids[0],
                row_id=row_ids[0],
                custom={
                    "is_batch": True,
                    "batch_metadata": {
                        "row_ids": row_ids,
                        "original_count": expected_count,
                    },
                },
            )
        ],
        tokens_used=130,
        cost=Decimal("0.001"),
        batch_id=0,
        latencies_ms=[200.0],
    )


class TestDisaggregatorStructuredResultFastPath:
    """Regression: removing the structured_result Pydantic fast path in
    BatchDisaggregatorStage.process() forces JSON text parsing, which
    raises PartialParseError on common LLM response variations."""

    def test_all_items_extracted_via_pydantic_fast_path(self):
        batch = _make_batch(n_items=5)
        stage = BatchDisaggregatorStage(strategy=JsonBatchStrategy())

        result = stage.process([batch], context=None)

        disagg = result[0]
        assert len(disagg.responses) == 5
        for resp in disagg.responses:
            assert resp != "null"
            parsed = json.loads(resp)
            assert "score" in parsed
            assert "reason" in parsed
            # The Pydantic fast path serializes only the .result fields
            # (no batch "id"). The JSON fallback includes "id".
            # This distinguishes which code path was taken.
            assert "id" not in parsed, (
                f"Found 'id' in response — disaggregator used the fragile "
                f"JSON fallback instead of the Pydantic fast path: {parsed}"
            )


class TestDisaggregatorItemCountMismatch:
    """Regression: before the fix, when structured_result had fewer items
    than original_count, the disaggregator produced fewer responses than
    row_ids, causing a metadata/response length mismatch downstream and
    silently dropping rows instead of marking them as null for retry."""

    def test_fewer_items_pads_missing_with_null(self):
        batch = _make_batch(n_items=3, expected_count=5)
        stage = BatchDisaggregatorStage(strategy=JsonBatchStrategy())

        result = stage.process([batch], context=None)

        disagg = result[0]
        assert len(disagg.responses) == 5, (
            f"Expected 5 responses (3 valid + 2 null), got {len(disagg.responses)}"
        )

        for i in range(3):
            assert disagg.responses[i] != "null", f"Response {i} should be valid"
            parsed = json.loads(disagg.responses[i])
            assert "score" in parsed

        for i in range(3, 5):
            assert disagg.responses[i] == "null", (
                f"Response {i} should be null (missing item)"
            )


class TestRestoreCompletedResponsesPreservesStructuredResult:
    """Regression: _restore_completed_response_batches used to rebuild
    LLMResponse without structured_result, which forced every disaggregation
    through the fragile JSON text path — even when the LLM stage had returned
    a fully validated Pydantic object moments earlier."""

    def test_restored_response_carries_structured_result(self):
        """Exercise the actual Pipeline._restore_completed_response_batches
        method with a completed_responses record that has _structured_result."""
        from ondine.api.pipeline import Pipeline
        from ondine.orchestration.execution_context import ExecutionContext

        pydantic_batch = SwapBatch(
            items=[
                SwapBatchItem(id=1, result=SwapResult(score=3, reason="good")),
                SwapBatchItem(id=2, result=SwapResult(score=1, reason="bad")),
            ]
        )

        # Build a minimal Pipeline with just enough spec to call the method
        spec_mock = Mock()
        spec_mock.llm.model = "azure/gpt-5-nano"
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.specifications = spec_mock

        context = ExecutionContext(
            session_id=uuid4(),
            intermediate_data={
                "completed_responses": [
                    {
                        "text": pydantic_batch.model_dump_json(),
                        "tokens_in": 40,
                        "tokens_out": 60,
                        "model": "azure/gpt-5-nano",
                        "cost": "0.001",
                        "latency_ms": 150.0,
                        "metadata": {},
                        "row_metadata": {
                            "row_index": 0,
                            "row_id": 0,
                            "custom": {
                                "is_batch": True,
                                "batch_metadata": {
                                    "row_ids": [0, 1],
                                    "original_count": 2,
                                },
                            },
                        },
                        "_structured_result": pydantic_batch,
                    }
                ]
            },
        )

        restored = pipeline._restore_completed_response_batches(context)

        assert len(restored) == 1
        llm_resp = restored[0].responses[0]
        assert llm_resp.structured_result is not None, (
            "structured_result was lost during restore — disaggregator will "
            "fall back to fragile JSON text parsing"
        )
        assert llm_resp.structured_result is pydantic_batch
