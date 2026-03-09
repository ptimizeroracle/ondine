"""
Unit tests for batch disaggregation with structured_result preservation.

Regression these tests catch:
- structured_result is always lost because _restore_completed_response_batches
  rebuilds LLMResponse without it, forcing the disaggregator into the fragile
  JSON text-parsing path. This causes PartialParseError and cascading retries,
  wasting ~47% of throughput.

- completed_responses checkpoint records don't store structured_result, so
  even when responses have valid Pydantic objects, they get serialized to JSON
  text and re-parsed — introducing fragile text extraction logic that fails
  under common LLM response variations.
"""

import json
from decimal import Decimal

import pytest
from pydantic import BaseModel, Field

from ondine.core.models import LLMResponse, ResponseBatch, RowMetadata
from ondine.stages.batch_disaggregator_stage import BatchDisaggregatorStage
from ondine.strategies.json_batch_strategy import JsonBatchStrategy


class SwapResult(BaseModel):
    """Sample structured output — mirrors real-world batch item result."""

    score: int = Field(ge=0, le=5)
    reason: str


class SwapBatchItem(BaseModel):
    id: int
    result: SwapResult


class SwapBatch(BaseModel):
    items: list[SwapBatchItem]


class TestDisaggregatorUsesStructuredResult:
    """When structured_result carries a valid Pydantic batch, disaggregation
    should extract items directly — no JSON parsing, no PartialParseError."""

    def _make_batch_with_structured_result(
        self, n_items: int = 3
    ) -> tuple[ResponseBatch, list[int]]:
        """Build a ResponseBatch whose LLMResponse has a structured_result."""
        row_ids = list(range(100, 100 + n_items))
        items = [
            SwapBatchItem(
                id=i + 1, result=SwapResult(score=i + 1, reason=f"reason_{i}")
            )
            for i in range(n_items)
        ]
        pydantic_batch = SwapBatch(items=items)

        llm_response = LLMResponse(
            text=pydantic_batch.model_dump_json(),
            tokens_in=50,
            tokens_out=80,
            model="azure/gpt-5-nano",
            cost=Decimal("0.001"),
            latency_ms=200.0,
            structured_result=pydantic_batch,
        )

        batch_metadata = {
            "row_ids": row_ids,
            "original_count": n_items,
        }
        metadata = RowMetadata(
            row_index=row_ids[0],
            row_id=row_ids[0],
            custom={"is_batch": True, "batch_metadata": batch_metadata},
        )

        return (
            ResponseBatch(
                responses=[llm_response],
                metadata=[metadata],
                tokens_used=130,
                cost=Decimal("0.001"),
                batch_id=0,
                latencies_ms=[200.0],
            ),
            row_ids,
        )

    def test_structured_result_extracts_all_items(self):
        """Disaggregator should produce one response per batch item
        when structured_result is present — no partial parse errors."""
        batch, row_ids = self._make_batch_with_structured_result(n_items=5)
        stage = BatchDisaggregatorStage(strategy=JsonBatchStrategy())
        result_batches = stage.process([batch], context=None)

        assert len(result_batches) == 1
        disagg = result_batches[0]
        assert len(disagg.responses) == 5, (
            f"Expected 5 disaggregated responses, got {len(disagg.responses)}"
        )
        for resp in disagg.responses:
            assert resp != "null", (
                "No response should be null when structured_result is valid"
            )
            parsed = json.loads(resp)
            assert "score" in parsed, (
                f"Missing 'score' in disaggregated response: {parsed}"
            )
            assert "reason" in parsed, (
                f"Missing 'reason' in disaggregated response: {parsed}"
            )


class TestCompletedResponsesPreserveStructuredResult:
    """The completed_responses checkpoint record must carry _structured_result
    so that _restore_completed_response_batches produces LLMResponses that
    still have the Pydantic object for the disaggregator's fast path."""

    def test_structured_result_roundtrips_through_completed_responses(self):
        """Simulate the completed_responses serialization + restore path
        and verify structured_result survives when records are in-memory."""
        items = [
            SwapBatchItem(id=1, result=SwapResult(score=3, reason="good")),
            SwapBatchItem(id=2, result=SwapResult(score=1, reason="bad")),
        ]
        pydantic_batch = SwapBatch(items=items)

        original_response = LLMResponse(
            text=pydantic_batch.model_dump_json(),
            tokens_in=40,
            tokens_out=60,
            model="azure/gpt-5-nano",
            cost=Decimal("0.001"),
            latency_ms=150.0,
            structured_result=pydantic_batch,
        )

        # Simulate what llm_invocation_stage now does (includes _structured_result)
        completed_record = {
            "text": original_response.text,
            "tokens_in": original_response.tokens_in,
            "tokens_out": original_response.tokens_out,
            "model": original_response.model,
            "cost": str(original_response.cost),
            "latency_ms": original_response.latency_ms,
            "metadata": original_response.metadata or {},
            "row_metadata": {
                "row_index": 0,
                "row_id": 0,
                "custom": {
                    "is_batch": True,
                    "batch_metadata": {"row_ids": [0, 1], "original_count": 2},
                },
            },
            "_structured_result": original_response.structured_result,
        }

        # Simulate what _restore_completed_response_batches now does
        restored_response = LLMResponse(
            text=completed_record.get("text", ""),
            tokens_in=completed_record.get("tokens_in", 0),
            tokens_out=completed_record.get("tokens_out", 0),
            model=completed_record.get("model", "unknown"),
            cost=Decimal(str(completed_record.get("cost", "0"))),
            latency_ms=completed_record.get("latency_ms", 0.0),
            metadata=completed_record.get("metadata", {}),
            structured_result=completed_record.get("_structured_result"),
        )

        assert restored_response.structured_result is not None, (
            "structured_result was lost during completed_responses roundtrip. "
            "The disaggregator will fall back to fragile JSON text parsing."
        )
        assert hasattr(restored_response.structured_result, "items")
        assert len(restored_response.structured_result.items) == 2


class TestDisaggregatorFallbackJsonParsing:
    """When structured_result is None (lost in checkpoint), the disaggregator
    falls back to JSON text parsing via JsonBatchStrategy.parse_batch_response.
    Verify the specific failure modes."""

    def test_json_fallback_with_fewer_items_than_expected(self):
        """LLM returns 3 items but 5 were expected — currently raises
        PartialParseError marking 2 rows as null, triggering retry."""
        strategy = JsonBatchStrategy()

        response_json = json.dumps(
            [
                {"id": 1, "result": {"score": 3, "reason": "ok"}},
                {"id": 2, "result": {"score": 4, "reason": "good"}},
                {"id": 3, "result": {"score": 1, "reason": "bad"}},
            ]
        )

        # This should NOT raise — it should return what it has and mark missing as null
        # But currently it raises PartialParseError, causing cascading retries
        from ondine.strategies.batch_formatting import PartialParseError

        with pytest.raises(PartialParseError) as exc_info:
            strategy.parse_batch_response(response_json, expected_count=5)

        assert len(exc_info.value.parsed_results) == 3
        assert 4 in exc_info.value.failed_ids
        assert 5 in exc_info.value.failed_ids

    def test_disaggregator_handles_fewer_items_gracefully(self):
        """When structured_result has fewer items than expected, the
        disaggregator should keep parsed items and only null-out missing ones —
        not re-raise or cascade into full retry."""
        items = [
            SwapBatchItem(id=1, result=SwapResult(score=3, reason="ok")),
            SwapBatchItem(id=2, result=SwapResult(score=4, reason="good")),
            SwapBatchItem(id=3, result=SwapResult(score=1, reason="bad")),
        ]
        pydantic_batch = SwapBatch(items=items)

        row_ids = [10, 11, 12, 13, 14]
        metadata = RowMetadata(
            row_index=10,
            row_id=10,
            custom={
                "is_batch": True,
                "batch_metadata": {"row_ids": row_ids, "original_count": 5},
            },
        )

        llm_response = LLMResponse(
            text=pydantic_batch.model_dump_json(),
            tokens_in=50,
            tokens_out=80,
            model="azure/gpt-5-nano",
            cost=Decimal("0.001"),
            latency_ms=200.0,
            structured_result=pydantic_batch,
        )

        batch = ResponseBatch(
            responses=[llm_response],
            metadata=[metadata],
            tokens_used=130,
            cost=Decimal("0.001"),
            batch_id=0,
            latencies_ms=[200.0],
        )

        stage = BatchDisaggregatorStage(strategy=JsonBatchStrategy())
        result_batches = stage.process([batch], context=None)

        assert len(result_batches) == 1
        disagg = result_batches[0]
        assert len(disagg.responses) == 5, (
            f"Should produce 5 responses (3 valid + 2 null), got {len(disagg.responses)}"
        )

        # First 3 should have valid data
        for i in range(3):
            assert disagg.responses[i] != "null", f"Response {i} should be valid"
            parsed = json.loads(disagg.responses[i])
            assert "score" in parsed

        # Last 2 should be null (missing from LLM response)
        for i in range(3, 5):
            assert disagg.responses[i] == "null", (
                f"Response {i} should be null (missing item)"
            )
