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
        mock_client.router = None  # No router in mock (single deployment)
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
