"""Unit tests for batch formatting strategies."""

from unittest.mock import MagicMock

import pytest

from ondine.strategies.batch_formatting import (
    BatchFormattingStrategy,
    PartialParseError,
)
from ondine.strategies.json_batch_strategy import JsonBatchStrategy
from ondine.strategies.models import BatchItem, BatchResult


class TestBatchFormattingStrategy:
    """Tests for BatchFormattingStrategy interface."""

    def test_is_abstract(self):
        """BatchFormattingStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BatchFormattingStrategy()  # type: ignore


class TestJsonBatchStrategy:
    """Tests for JsonBatchStrategy implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = JsonBatchStrategy()

    def test_format_batch_creates_valid_prompt(self):
        """Should format prompts as JSON array with instructions."""
        prompts = ["Classify: Product A is great", "Classify: Product B is terrible"]

        result = self.strategy.format_batch(prompts)

        # Check structure
        assert "Process these 2 items" in result
        assert '"id": 1' in result
        assert '"id": 2' in result
        assert "Product A is great" in result
        assert "Product B is terrible" in result
        assert "OUTPUT FORMAT" in result
        assert "JSON Array:" in result

    def test_format_batch_with_metadata(self):
        """Should handle metadata in format_batch."""
        prompts = ["Test prompt"]
        metadata = {"task": "sentiment classification"}

        result = self.strategy.format_batch(prompts, metadata=metadata)

        # Should still work (metadata is optional)
        assert "Test prompt" in result

    def test_parse_batch_response_success(self):
        """Should parse valid JSON array response."""
        response = """[
  {"id": 1, "result": "positive"},
  {"id": 2, "result": "negative"}
]"""

        results = self.strategy.parse_batch_response(response, expected_count=2)

        assert results == ["positive", "negative"]

    def test_parse_batch_response_with_markdown(self):
        """Should extract JSON from markdown code blocks."""
        response = """Here are the results:

```json
[
  {"id": 1, "result": "positive"},
  {"id": 2, "result": "negative"}
]
```

All done!"""

        results = self.strategy.parse_batch_response(response, expected_count=2)

        assert results == ["positive", "negative"]

    def test_parse_batch_response_unordered(self):
        """Should handle unordered results (sorts by ID)."""
        response = """[
  {"id": 2, "result": "negative"},
  {"id": 1, "result": "positive"}
]"""

        results = self.strategy.parse_batch_response(response, expected_count=2)

        # Should be sorted by ID
        assert results == ["positive", "negative"]

    def test_parse_batch_response_partial_results(self):
        """Should raise PartialParseError when some results missing."""
        response = """[
  {"id": 1, "result": "positive"},
  {"id": 3, "result": "neutral"}
]"""

        with pytest.raises(PartialParseError) as exc_info:
            self.strategy.parse_batch_response(response, expected_count=3)

        error = exc_info.value
        assert len(error.parsed_results) == 2
        assert error.failed_ids == [2]
        assert "positive" in error.parsed_results
        assert "neutral" in error.parsed_results

    def test_parse_batch_response_no_json(self):
        """Should raise ValueError when no JSON found."""
        response = "I cannot provide results in JSON format."

        with pytest.raises(ValueError, match="No JSON array found"):
            self.strategy.parse_batch_response(response, expected_count=2)

    def test_parse_batch_response_invalid_json(self):
        """Should raise ValueError for malformed JSON."""
        response = "[{id: 1, result: positive}]"  # Missing quotes

        with pytest.raises(ValueError, match="Invalid JSON"):
            self.strategy.parse_batch_response(response, expected_count=1)

    def test_parse_batch_response_not_array(self):
        """Should handle JSON object by wrapping in array."""
        response = '{"id": 1, "result": "positive"}'  # Object, not array

        # Strategy wraps single object in array automatically
        results = self.strategy.parse_batch_response(response, expected_count=1)

        assert results == ["positive"]

    def test_parse_batch_response_missing_fields(self):
        """Should handle items with missing result field (None)."""
        response = '[{"id": 1}]'  # Missing "result" field

        # Pydantic allows None for optional fields
        results = self.strategy.parse_batch_response(response, expected_count=1)

        # Result should be empty string (None converted)
        assert results == [""]

    def test_estimate_batch_tokens(self):
        """Should estimate tokens with JSON overhead."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = [
            [1] * 10,  # First prompt: 10 tokens
            [1] * 15,  # Second prompt: 15 tokens
        ]

        prompts = ["Short prompt", "Slightly longer prompt"]
        estimated = self.strategy.estimate_batch_tokens(prompts, mock_tokenizer)

        # Base overhead (200) + per-item overhead (25*2) + prompt tokens (10+15)
        expected = 200 + (25 * 2) + 10 + 15
        assert estimated == expected


class TestBatchModels:
    """Tests for Pydantic batch models."""

    def test_batch_item_validation(self):
        """Should validate BatchItem fields."""
        # Valid item
        item = BatchItem(id=1, input="test", result="output")
        assert item.id == 1
        assert item.input == "test"
        assert item.result == "output"

        # Invalid ID (< 1) - Pydantic raises ValidationError
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            BatchItem(id=0, input="test")

    def test_batch_result_to_list(self):
        """Should convert BatchResult to sorted list."""
        result = BatchResult(
            results=[
                BatchItem(id=2, result="negative"),
                BatchItem(id=1, result="positive"),
                BatchItem(id=3, result="neutral"),
            ]
        )

        results_list = result.to_list()

        # Should be sorted by ID
        assert results_list == ["positive", "negative", "neutral"]

    def test_batch_result_get_missing_ids(self):
        """Should identify missing IDs."""
        result = BatchResult(
            results=[
                BatchItem(id=1, result="positive"),
                BatchItem(id=3, result="neutral"),
            ]
        )

        missing = result.get_missing_ids(expected_count=3)

        assert missing == [2]

    def test_batch_result_no_missing_ids(self):
        """Should return empty list when all IDs present."""
        result = BatchResult(
            results=[
                BatchItem(id=1, result="positive"),
                BatchItem(id=2, result="negative"),
            ]
        )

        missing = result.get_missing_ids(expected_count=2)

        assert missing == []
