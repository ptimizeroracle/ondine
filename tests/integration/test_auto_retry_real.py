"""
Real integration test for auto-retry functionality.

This test ACTUALLY executes the pipeline (not just config checks)
to verify retry processes only failed rows.
"""

from decimal import Decimal
from unittest.mock import Mock, patch

import pandas as pd

from ondine import PipelineBuilder
from ondine.core.models import LLMResponse


class TestAutoRetryActualExecution:
    """Integration tests that actually execute retry logic."""

    @patch("ondine.adapters.provider_registry.ProviderRegistry.get")
    def test_retry_processes_only_failed_rows_not_all(self, mock_get):
        """
        CRITICAL TEST: Verify retry processes ONLY failed rows.

        Regression test for bug where retry processed all 365 rows instead of 72.
        """
        # Create test data (10 rows)
        df = pd.DataFrame(
            {
                "pk": [f"row_{i}" for i in range(10)],
                "text": [f"PRODUCT {i}" for i in range(10)],
            }
        )

        # Track which rows are processed
        processed_rows = []

        def mock_invoke(prompt, **kwargs):
            """Mock LLM that tracks calls and fails specific rows."""
            # Extract row identifier from prompt
            for i in range(10):
                if f"PRODUCT {i}" in prompt:
                    processed_rows.append(i)

                    # First pass: Fail rows 2, 5, 8 (return empty)
                    # Second pass (retry): Succeed
                    if len([x for x in processed_rows if x == i]) == 1:
                        # First time seeing this row - fail these specific ones
                        if i in [2, 5, 8]:
                            text = ""  # Empty output
                        else:
                            text = f"Cleaned product {i}"
                    else:
                        # Second time (retry) - succeed
                        text = f"Retry cleaned product {i}"

                    return LLMResponse(
                        text=text,
                        tokens_in=10,
                        tokens_out=10,
                        model="test-model",
                        cost=Decimal("0.0001"),
                        latency_ms=100.0,
                    )

            return LLMResponse(
                text="Unknown",
                tokens_in=10,
                tokens_out=10,
                model="test-model",
                cost=Decimal("0.0001"),
                latency_ms=100.0,
            )

        # Setup mock
        mock_client = Mock()
        mock_client.invoke = Mock(side_effect=mock_invoke)
        mock_client.spec = Mock(
            model="test-model",
            input_cost_per_1k_tokens=Decimal("0.0001"),
            output_cost_per_1k_tokens=Decimal("0.0001"),
        )
        mock_client_class = Mock(return_value=mock_client)
        mock_get.return_value = mock_client_class

        # Build pipeline with auto-retry
        # Note: Using processing_batch_size for internal batching, NOT multi-row batching
        # Retry tests need row-by-row API calls to verify retry logic
        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["cleaned"])
            .with_prompt(template="Clean: {text}")
            .with_llm(provider="groq", model="test", temperature=0.0)
            .with_processing_batch_size(10)  # Internal batching only
            .with_concurrency(1)
            .build()
        )

        # Enable auto-retry
        pipeline.specifications.processing.auto_retry_failed = True
        pipeline.specifications.processing.max_retry_attempts = 1

        # Execute (should process all 10, then retry 3 failed)
        result = pipeline.execute()

        # Verify execution counts
        # First pass: 10 rows
        # Retry: 3 rows (indices 2, 5, 8)
        # Total: 13 LLM calls (not 20!)
        assert len(processed_rows) == 13, (
            f"Expected 13 calls (10 + 3 retry), got {len(processed_rows)}"
        )

        # Verify each failed row was retried exactly once
        assert processed_rows.count(2) == 2  # Once in pass 1, once in retry
        assert processed_rows.count(5) == 2
        assert processed_rows.count(8) == 2

        # Verify successful rows were NOT retried
        assert processed_rows.count(0) == 1
        assert processed_rows.count(1) == 1
        assert processed_rows.count(3) == 1

        # Verify final quality
        quality = result.validate_output_quality(["cleaned"])
        assert quality.valid_outputs == 10  # All should be valid after retry
        assert quality.success_rate == 100.0

    @patch("ondine.adapters.provider_registry.ProviderRegistry.get")
    def test_retry_respects_max_attempts(self, mock_get):
        """Should stop retrying after max_retry_attempts."""
        df = pd.DataFrame({"text": ["row1", "row2", "row3"]})

        call_count = {"count": 0}

        def mock_invoke(prompt, **kwargs):
            """Mock that always returns empty (forces retries)."""
            call_count["count"] += 1
            return LLMResponse(
                text="",  # Always empty
                tokens_in=10,
                tokens_out=10,
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
        mock_client_class = Mock(return_value=mock_client)
        mock_get.return_value = mock_client_class

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["cleaned"])
            .with_prompt(template="Clean: {text}")
            .with_llm(provider="groq", model="test", temperature=0.0)
            .with_processing_batch_size(3)  # Internal batching only, not multi-row
            .build()
        )

        # Set max_retry_attempts to 2
        pipeline.specifications.processing.auto_retry_failed = True
        pipeline.specifications.processing.max_retry_attempts = 2

        pipeline.execute()

        # Should call LLM exactly: 3 (initial) + 3 (retry 1) + 3 (retry 2) = 9 times
        # NOT infinite loop
        assert call_count["count"] == 9, (
            f"Expected 9 calls (3 + 3 + 3), got {call_count['count']}"
        )

    def test_retry_configuration_validates(self):
        """Should configure retry settings correctly."""
        df = pd.DataFrame({"text": ["test"]})

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["cleaned"])
            .with_prompt(template="Clean: {text}")
            .with_llm(provider="groq", model="test", temperature=0.0)
            .build()
        )

        pipeline.specifications.processing.auto_retry_failed = True
        pipeline.specifications.processing.max_retry_attempts = 2

        # Verify configuration
        assert pipeline.specifications.processing.auto_retry_failed is True
        assert pipeline.specifications.processing.max_retry_attempts == 2
