"""
Integration test for auto-retry with multi-row batching.

This test verifies that auto_retry_failed works correctly when batch_size > 1,
ensuring failed rows are retried at the row level even when initial processing
used multi-row batching.
"""

from decimal import Decimal
from unittest.mock import Mock, patch

import pandas as pd

from ondine import PipelineBuilder
from ondine.core.models import LLMResponse


class TestAutoRetryWithBatching:
    """Test auto-retry compatibility with multi-row batching."""

    @patch("ondine.adapters.provider_registry.ProviderRegistry.get")
    def test_auto_retry_works_with_batch_size(self, mock_get):
        """
        CRITICAL: Verify auto_retry extracts failed rows from batches and retries them.

        Scenario:
        - Initial pass: 20 rows with batch_size=5 → 4 API calls
        - Rows 3, 7, 12, 18 return empty → need retry
        - Retry: 4 failed rows with batch_size=5 → 1 API call (all fit in one batch)
        - Total: 5 API calls (4 initial + 1 retry)
        """
        # Create 20 rows
        df = pd.DataFrame({"product": [f"PRODUCT_{i}" for i in range(20)]})

        # Track all LLM invocations
        call_count = [0]
        batch_calls = []  # Track which rows are in each batch call

        def mock_invoke(prompt, **kwargs):
            """Mock LLM that processes batches and fails specific rows."""
            call_count[0] += 1

            # Parse batch (JSON array format with id and text)
            rows_in_batch = []
            for i in range(20):
                if f"PRODUCT_{i}" in prompt:
                    rows_in_batch.append(i)

            batch_calls.append(rows_in_batch)

            # Check if this is a batch call (multiple rows)
            if len(rows_in_batch) > 1:
                # Multi-row batch: Return JSON array
                results = []
                for idx, row_num in enumerate(rows_in_batch):
                    # Fail rows 3, 7, 12, 18 on first attempt
                    # (first time we see them in batch_calls)
                    times_seen = sum(
                        1 for batch in batch_calls[:-1] if row_num in batch
                    )

                    if times_seen == 0 and row_num in [3, 7, 12, 18]:
                        # First time seeing this row - fail it
                        result = {"brand": None, "size": None, "type": None}
                    else:
                        # Success or retry
                        result = {
                            "brand": f"Brand_{row_num}",
                            "size": f"{row_num}oz",
                            "type": "Food",
                        }

                    results.append({"id": idx + 1, "result": result})

                import json

                text = json.dumps(results)
            else:
                # Single row (shouldn't happen with batch_size=5, but handle it)
                row_num = rows_in_batch[0]
                result = {
                    "brand": f"Brand_{row_num}",
                    "size": f"{row_num}oz",
                    "type": "Food",
                }
                import json

                text = json.dumps(result)

            return LLMResponse(
                text=text,
                tokens_in=100,
                tokens_out=50,
                model="test-model",
                cost=Decimal("0.001"),
                latency_ms=200.0,
            )

        # Setup mock
        mock_client = Mock()
        mock_client.invoke = Mock(side_effect=mock_invoke)
        mock_client.spec = Mock(
            model="test-model",
            input_cost_per_1k_tokens=Decimal("0.0001"),
            output_cost_per_1k_tokens=Decimal("0.0001"),
        )
        mock_client.estimate_tokens = Mock(return_value=100)
        mock_client.calculate_cost = Mock(return_value=Decimal("0.001"))
        mock_client_class = Mock(return_value=mock_client)
        mock_get.return_value = mock_client_class

        # Build pipeline with multi-row batching
        from ondine.stages.response_parser_stage import JSONParser

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                df, input_columns=["product"], output_columns=["brand", "size", "type"]
            )
            .with_prompt(template="Extract product info: {{ product }}")
            .with_llm(provider="openai", model="gpt-4o-mini", temperature=0.0)
            .with_parser(JSONParser(strict=False))
            .with_jinja2(True)
            .with_batch_size(5)  # Multi-row batching: 5 rows per API call
            .with_concurrency(2)
            .build()
        )

        # Enable auto-retry
        pipeline.specifications.processing.auto_retry_failed = True
        pipeline.specifications.processing.max_retry_attempts = 1

        # Execute
        result = pipeline.execute()

        # Verify batch structure
        # Initial pass: 20 rows / 5 per batch = 4 calls
        # Retry: 4 failed rows / 5 per batch = 1 call
        # Total: 5 calls
        assert call_count[0] == 5, (
            f"Expected 5 API calls (4 + 1 retry), got {call_count[0]}"
        )

        # Verify initial batches
        assert len(batch_calls[0]) == 5, "First batch should have 5 rows"
        assert len(batch_calls[1]) == 5, "Second batch should have 5 rows"
        assert len(batch_calls[2]) == 5, "Third batch should have 5 rows"
        assert len(batch_calls[3]) == 5, "Fourth batch should have 5 rows"

        # Verify retry batch contains only failed rows
        assert len(batch_calls[4]) == 4, "Retry batch should have 4 failed rows"
        assert set(batch_calls[4]) == {3, 7, 12, 18}, (
            "Retry should process exactly the failed rows"
        )

        # Verify final quality - all should be valid after retry
        assert result.data["brand"].notna().sum() == 20, (
            "All brands should be populated"
        )
        assert result.data["size"].notna().sum() == 20, "All sizes should be populated"
        assert result.data["type"].notna().sum() == 20, "All types should be populated"

        # Verify specific retried rows
        assert result.data.loc[3, "brand"] == "Brand_3", (
            "Row 3 should be retried successfully"
        )
        assert result.data.loc[7, "brand"] == "Brand_7", (
            "Row 7 should be retried successfully"
        )
        assert result.data.loc[12, "brand"] == "Brand_12", (
            "Row 12 should be retried successfully"
        )
        assert result.data.loc[18, "brand"] == "Brand_18", (
            "Row 18 should be retried successfully"
        )

    @patch("ondine.adapters.provider_registry.ProviderRegistry.get")
    def test_auto_retry_with_single_failed_row_in_large_batch(self, mock_get):
        """
        Edge case: Only 1 row fails in a batch of 10.

        Verify:
        - Only the 1 failed row is retried (not the entire batch)
        - Successful rows in the same batch are NOT retried
        """
        df = pd.DataFrame({"text": [f"Item {i}" for i in range(10)]})

        invocations = []

        def mock_invoke(prompt, **kwargs):
            """Track each invocation and fail row 5 only."""
            rows_in_call = []
            for i in range(10):
                if f"Item {i}" in prompt:
                    rows_in_call.append(i)

            invocations.append(rows_in_call)

            # Fail row 5 on first attempt (first time we see it)
            if len(rows_in_call) > 1:
                # Batch call
                results = []
                for idx, row_num in enumerate(rows_in_call):
                    times_seen = sum(1 for inv in invocations[:-1] if row_num in inv)

                    if times_seen == 0 and row_num == 5:
                        result = None  # Fail row 5
                    else:
                        result = f"Processed_{row_num}"

                    results.append({"id": idx + 1, "result": result})

                import json

                text = json.dumps(results)
            else:
                # Single row retry
                text = f"Processed_{rows_in_call[0]}"

            return LLMResponse(
                text=text,
                tokens_in=50,
                tokens_out=20,
                model="test",
                cost=Decimal("0.0001"),
                latency_ms=100.0,
            )

        mock_client = Mock()
        mock_client.invoke = Mock(side_effect=mock_invoke)
        mock_client.spec = Mock(
            model="test",
            input_cost_per_1k_tokens=Decimal("0.0001"),
            output_cost_per_1k_tokens=Decimal("0.0001"),
        )
        mock_client.estimate_tokens = Mock(return_value=50)
        mock_client.calculate_cost = Mock(return_value=Decimal("0.0001"))
        mock_client_class = Mock(return_value=mock_client)
        mock_get.return_value = mock_client_class

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {{ text }}")
            .with_llm(provider="openai", model="test", temperature=0.0)
            .with_jinja2(True)
            .with_batch_size(10)  # All 10 rows in 1 batch
            .build()
        )

        pipeline.specifications.processing.auto_retry_failed = True
        pipeline.specifications.processing.max_retry_attempts = 1

        result = pipeline.execute()

        # Verify: 1 initial call (10 rows) + 1 retry call (1 row) = 2 calls
        assert len(invocations) == 2, f"Expected 2 calls, got {len(invocations)}"
        assert len(invocations[0]) == 10, "First call should process all 10 rows"
        assert invocations[1] == [5], "Retry should only process the 1 failed row"

        # Verify all rows are valid
        assert result.data["result"].notna().sum() == 10

    @patch("ondine.adapters.provider_registry.ProviderRegistry.get")
    def test_auto_retry_respects_max_attempts_with_batching(self, mock_get):
        """
        Verify max_retry_attempts is respected when using batching.

        Scenario:
        - Row keeps failing even after retries
        - Should stop after max_retry_attempts
        """
        df = pd.DataFrame({"text": [f"Item {i}" for i in range(5)]})

        attempt_count = [0]

        def mock_invoke(prompt, **kwargs):
            """Always fail row 2, no matter how many retries."""
            attempt_count[0] += 1

            rows_in_call = []
            for i in range(5):
                if f"Item {i}" in prompt:
                    rows_in_call.append(i)

            if len(rows_in_call) > 1:
                # Batch
                results = []
                for idx, row_num in enumerate(rows_in_call):
                    if row_num == 2:
                        result = None  # Always fail row 2
                    else:
                        result = f"OK_{row_num}"
                    results.append({"id": idx + 1, "result": result})

                import json

                text = json.dumps(results)
            else:
                # Single row (retry) - still fail row 2
                if rows_in_call[0] == 2:
                    text = json.dumps(None)
                else:
                    text = f"OK_{rows_in_call[0]}"

            return LLMResponse(
                text=text,
                tokens_in=30,
                tokens_out=10,
                model="test",
                cost=Decimal("0.0001"),
                latency_ms=50.0,
            )

        mock_client = Mock()
        mock_client.invoke = Mock(side_effect=mock_invoke)
        mock_client.spec = Mock(
            model="test",
            input_cost_per_1k_tokens=Decimal("0.0001"),
            output_cost_per_1k_tokens=Decimal("0.0001"),
        )
        mock_client.estimate_tokens = Mock(return_value=30)
        mock_client.calculate_cost = Mock(return_value=Decimal("0.0001"))
        mock_client_class = Mock(return_value=mock_client)
        mock_get.return_value = mock_client_class

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {{ text }}")
            .with_llm(provider="openai", model="test", temperature=0.0)
            .with_jinja2(True)
            .with_batch_size(5)
            .build()
        )

        pipeline.specifications.processing.auto_retry_failed = True
        pipeline.specifications.processing.max_retry_attempts = 2  # Try twice

        result = pipeline.execute()

        # Verify: 1 initial + 2 retries = 3 calls
        assert attempt_count[0] == 3, (
            f"Expected 3 calls (1 + 2 retries), got {attempt_count[0]}"
        )

        # Verify row 2 is still None (failed all attempts)
        assert pd.isna(result.data.loc[2, "result"]), (
            "Row 2 should remain None after max retries"
        )

        # Verify other rows succeeded
        assert result.data.loc[0, "result"] == "OK_0"
        assert result.data.loc[1, "result"] == "OK_1"
        assert result.data.loc[3, "result"] == "OK_3"
        assert result.data.loc[4, "result"] == "OK_4"
