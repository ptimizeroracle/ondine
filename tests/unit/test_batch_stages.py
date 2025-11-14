"""Unit tests for batch processing stages."""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from ondine.core.models import PromptBatch, ResponseBatch, RowMetadata
from ondine.orchestration.execution_context import ExecutionContext
from ondine.stages.batch_aggregator_stage import BatchAggregatorStage
from ondine.stages.batch_disaggregator_stage import BatchDisaggregatorStage
from ondine.strategies.batch_formatting import PartialParseError
from ondine.strategies.json_batch_strategy import JsonBatchStrategy


class TestBatchAggregatorStage:
    """Tests for BatchAggregatorStage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = JsonBatchStrategy()
        self.stage = BatchAggregatorStage(
            batch_size=2, strategy=self.strategy, validate_context_window=False
        )

    def test_aggregates_prompts_into_batches(self):
        """Should aggregate multiple prompts into batch prompts."""
        # Create batch with 4 prompts
        prompts = [f"Classify: Product {i}" for i in range(4)]
        metadata = [RowMetadata(row_index=i, row_id=i) for i in range(4)]
        batch = PromptBatch(prompts=prompts, metadata=metadata, batch_id=0)

        context = ExecutionContext()
        result = self.stage.process([batch], context)

        # Should create 2 batch prompts (batch_size=2)
        assert len(result) == 2

        # Each batch should have 1 mega-prompt
        assert len(result[0].prompts) == 1
        assert len(result[1].prompts) == 1

        # Check metadata
        assert result[0].metadata[0].custom["is_batch"] is True
        assert result[0].metadata[0].custom["batch_size"] == 2

    def test_handles_partial_batch(self):
        """Should handle when rows don't divide evenly by batch_size."""
        # Create 5 prompts (batch_size=2 → 3 batches: 2, 2, 1)
        prompts = [f"Classify: Product {i}" for i in range(5)]
        metadata = [RowMetadata(row_index=i, row_id=i) for i in range(5)]
        batch = PromptBatch(prompts=prompts, metadata=metadata, batch_id=0)

        context = ExecutionContext()
        result = self.stage.process([batch], context)

        # Should create 3 batch prompts
        assert len(result) == 3

        # Last batch should have 1 row
        last_batch_metadata = result[2].metadata[0].custom
        assert last_batch_metadata["batch_size"] == 1

    def test_preserves_row_ids(self):
        """Should preserve original row IDs in metadata."""
        prompts = ["Test 1", "Test 2"]
        metadata = [
            RowMetadata(row_index=10, row_id=10),
            RowMetadata(row_index=20, row_id=20),
        ]
        batch = PromptBatch(prompts=prompts, metadata=metadata, batch_id=0)

        context = ExecutionContext()
        result = self.stage.process([batch], context)

        # Check row IDs preserved
        batch_metadata = result[0].metadata[0].custom["batch_metadata"]
        assert batch_metadata["row_ids"] == [10, 20]

    def test_validation_fails_for_invalid_batch_size(self):
        """Should raise ValueError for batch_size < 1."""
        stage = BatchAggregatorStage(batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            stage.validate(None)


class TestBatchDisaggregatorStage:
    """Tests for BatchDisaggregatorStage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = JsonBatchStrategy()
        self.stage = BatchDisaggregatorStage(
            strategy=self.strategy, retry_failed_individually=False
        )

    def test_disaggregates_batch_response(self):
        """Should split batch response into individual responses."""
        # Create batch response
        batch_response_text = """[
  {"id": 1, "result": "positive"},
  {"id": 2, "result": "negative"}
]"""

        metadata = [
            RowMetadata(
                row_index=10,
                row_id=10,
                custom={
                    "is_batch": True,
                    "batch_metadata": {
                        "original_count": 2,
                        "row_ids": [10, 20],
                        "prompt_template": None,
                    },
                },
            )
        ]

        batch = ResponseBatch(
            responses=[batch_response_text],
            metadata=metadata,
            tokens_used=100,
            cost=Decimal("0.001"),
            batch_id=0,
            latencies_ms=[100.0],
        )

        context = ExecutionContext()
        result = self.stage.process([batch], context)

        # Should create 1 batch with 2 individual responses
        assert len(result) == 1
        assert len(result[0].responses) == 2

        # Check results
        assert result[0].responses[0] == "positive"
        assert result[0].responses[1] == "negative"

        # Check metadata
        assert result[0].metadata[0].row_index == 10
        assert result[0].metadata[1].row_index == 20

    def test_passes_through_non_batch_responses(self):
        """Should pass through responses that aren't batched."""
        metadata = [RowMetadata(row_index=1, row_id=1, custom={"is_batch": False})]

        batch = ResponseBatch(
            responses=["positive"],
            metadata=metadata,
            tokens_used=50,
            cost=Decimal("0.0005"),
            batch_id=0,
            latencies_ms=[50.0],
        )

        context = ExecutionContext()
        result = self.stage.process([batch], context)

        # Should pass through unchanged
        assert len(result) == 1
        assert result[0].responses[0] == "positive"

    def test_handles_partial_parse_error(self):
        """Should handle partial parse errors gracefully."""
        # Mock strategy that raises PartialParseError
        mock_strategy = Mock(spec=JsonBatchStrategy)
        mock_strategy.parse_batch_response.side_effect = PartialParseError(
            message="Missing ID 2",
            parsed_results=["positive", "neutral"],
            failed_ids=[2],
            original_response="...",
        )

        stage = BatchDisaggregatorStage(
            strategy=mock_strategy, retry_failed_individually=False
        )

        metadata = [
            RowMetadata(
                row_index=0,
                row_id=0,
                custom={
                    "is_batch": True,
                    "batch_metadata": {
                        "original_count": 3,
                        "row_ids": [1, 2, 3],
                        "prompt_template": None,
                    },
                },
            )
        ]

        batch = ResponseBatch(
            responses=["[...]"],
            metadata=metadata,
            tokens_used=150,
            cost=Decimal("0.003"),
            batch_id=0,
            latencies_ms=[150.0],
        )

        context = ExecutionContext()
        result = stage.process([batch], context)

        # Should create 3 responses (2 successful + 1 error marker)
        assert len(result[0].responses) == 3

        # Check that failed row has error marker
        assert "[PARSE_ERROR" in result[0].responses[1]  # Row ID 2 (index 1)

    def test_handles_complete_parse_failure(self):
        """Should handle complete parse failures."""
        # Mock strategy that raises ValueError
        mock_strategy = Mock(spec=JsonBatchStrategy)
        mock_strategy.parse_batch_response.side_effect = ValueError("Invalid JSON")

        stage = BatchDisaggregatorStage(strategy=mock_strategy)

        metadata = [
            RowMetadata(
                row_index=0,
                row_id=0,
                custom={
                    "is_batch": True,
                    "batch_metadata": {
                        "original_count": 2,
                        "row_ids": [1, 2],
                        "prompt_template": None,
                    },
                },
            )
        ]

        batch = ResponseBatch(
            responses=["Not JSON at all"],
            metadata=metadata,
            tokens_used=100,
            cost=Decimal("0.002"),
            batch_id=0,
            latencies_ms=[100.0],
        )

        context = ExecutionContext()
        result = stage.process([batch], context)

        # Should create error responses for all rows
        assert len(result[0].responses) == 2
        assert "[BATCH_PARSE_ERROR" in result[0].responses[0]
        assert "[BATCH_PARSE_ERROR" in result[0].responses[1]
