"""
Integration tests for auto-retry with multiple output columns.

Regression test for bug where retry only checked first output column,
missing failures in other columns.
"""

from decimal import Decimal
from unittest.mock import Mock, patch

import pandas as pd

from ondine.api import PipelineBuilder
from ondine.core.models import LLMResponse


class TestMultiColumnRetry:
    """Test auto-retry with multiple output columns."""

    @patch("ondine.adapters.provider_registry.ProviderRegistry.get")
    def test_retry_detects_failures_in_all_output_columns(self, mock_get):
        """
        CRITICAL TEST: Verify retry detects failures across ALL output columns.

        Regression test for bug where retry only checked output_cols[0],
        missing failures in other columns.

        Scenario:
        - 3 output columns: col1, col2, col3
        - Initial run: col1 succeeds, col2/col3 have failures
        - Retry should detect col2/col3 failures and retry those rows
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
            """Mock LLM that tracks calls and fails specific columns."""
            # Extract row identifier from prompt
            for i in range(10):
                if f"PRODUCT {i}" in prompt:
                    processed_rows.append(i)

                    # First pass: col1 always succeeds, col2/col3 fail for rows 2, 5, 8
                    # Second pass (retry): All succeed
                    if len([x for x in processed_rows if x == i]) == 1:
                        # First time seeing this row
                        if i in [2, 5, 8]:
                            # Fail col2 and col3 (return JSON with nulls)
                            text = (
                                f'{{"col1": "value_{i}", "col2": null, "col3": null}}'
                            )
                        else:
                            # All columns succeed
                            text = f'{{"col1": "value_{i}", "col2": "data_{i}", "col3": "info_{i}"}}'
                    else:
                        # Second time (retry) - all succeed
                        text = f'{{"col1": "value_{i}", "col2": "retry_data_{i}", "col3": "retry_info_{i}"}}'

                    return LLMResponse(
                        text=text,
                        tokens_in=10,
                        tokens_out=10,
                        model="test-model",
                        cost=Decimal("0.0001"),
                        latency_ms=100.0,
                    )

            return LLMResponse(
                text='{"col1": "unknown", "col2": "unknown", "col3": "unknown"}',
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

        # Build pipeline with auto-retry and MULTIPLE output columns
        from ondine.stages import JSONParser

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                df,
                input_columns=["text"],
                output_columns=["col1", "col2", "col3"],  # 3 output columns!
            )
            .with_prompt(template="Process: {text}")
            .with_llm(provider="groq", model="test", temperature=0.0)
            .with_parser(JSONParser(strict=False))
            .with_batch_size(10)
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
        # Retry: 3 rows (indices 2, 5, 8) - because col2/col3 had failures
        # Total: 13 LLM calls (not 10!)
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

        # Verify final quality - all should be valid after retry
        quality = result.validate_output_quality(["col1", "col2", "col3"])
        # valid_outputs counts valid rows, not cells
        # With 10 rows and 3 columns, if all are valid, we expect 10 valid rows
        assert quality.valid_outputs == 10
        assert quality.null_outputs == 0
        assert quality.success_rate == 100.0

    @patch("ondine.adapters.provider_registry.ProviderRegistry.get")
    def test_retry_continues_until_all_columns_valid_or_max_attempts(self, mock_get):
        """
        Test that retry continues for all columns until max attempts.

        Scenario:
        - Row 0: col1 fails, col2/col3 succeed
        - Row 1: col2 fails, col1/col3 succeed
        - Row 2: col3 fails, col1/col2 succeed
        - All should be retried
        """
        df = pd.DataFrame(
            {
                "text": ["A", "B", "C"],
            }
        )

        call_count = 0

        def mock_invoke(prompt, **kwargs):
            nonlocal call_count
            call_count += 1

            # Simulate different column failures for different rows
            if "A" in prompt:
                if call_count <= 3:  # First pass
                    text = '{"col1": null, "col2": "ok", "col3": "ok"}'
                else:  # Retry
                    text = '{"col1": "fixed", "col2": "ok", "col3": "ok"}'
            elif "B" in prompt:
                if call_count <= 3:
                    text = '{"col1": "ok", "col2": null, "col3": "ok"}'
                else:
                    text = '{"col1": "ok", "col2": "fixed", "col3": "ok"}'
            else:  # "C"
                if call_count <= 3:
                    text = '{"col1": "ok", "col2": "ok", "col3": null}'
                else:
                    text = '{"col1": "ok", "col2": "ok", "col3": "fixed"}'

            return LLMResponse(
                text=text,
                tokens_in=10,
                tokens_out=10,
                model="test-model",
                cost=Decimal("0.0001"),
                latency_ms=100.0,
            )

        mock_client = Mock()
        mock_client.invoke = Mock(side_effect=mock_invoke)
        mock_client.spec = Mock(
            model="test-model",
            input_cost_per_1k_tokens=Decimal("0.0001"),
            output_cost_per_1k_tokens=Decimal("0.0001"),
        )
        mock_client_class = Mock(return_value=mock_client)
        mock_get.return_value = mock_client_class

        from ondine.stages import JSONParser

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                df,
                input_columns=["text"],
                output_columns=["col1", "col2", "col3"],
            )
            .with_prompt(template="{text}")
            .with_llm(provider="groq", model="test", temperature=0.0)
            .with_parser(JSONParser(strict=False))
            .with_batch_size(3)
            .with_concurrency(1)
            .build()
        )

        pipeline.specifications.processing.auto_retry_failed = True
        pipeline.specifications.processing.max_retry_attempts = 1

        result = pipeline.execute()

        # Should process 3 rows initially, then retry all 3 (each had a failure)
        assert call_count == 6, f"Expected 6 calls (3 + 3 retry), got {call_count}"

        # All should be valid after retry
        quality = result.validate_output_quality(["col1", "col2", "col3"])
        assert quality.null_outputs == 0
        assert quality.success_rate == 100.0
