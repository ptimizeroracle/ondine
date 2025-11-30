"""
Integration tests for batch failure recovery.

Verifies that:
1. Empty/Malformed batches are caught by BatchDisaggregator
2. Rows are marked as failed (Nulls)
3. Pipeline._auto_retry_failed_rows picks them up and retries
"""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest
from pydantic import BaseModel

from ondine.api import PipelineBuilder
from ondine.core.models import LLMResponse
from ondine.orchestration import AsyncExecutor


class ResultModel(BaseModel):
    """Test model for structured output."""

    value: str


class BatchResponse(BaseModel):
    """Batch wrapper."""

    items: list[dict]


class TestBatchRecovery:
    """Test recovery from batch failures."""

    @patch("ondine.adapters.provider_registry.ProviderRegistry.get")
    @pytest.mark.asyncio
    async def test_empty_batch_triggers_retry(self, mock_get):
        """
        Test that an empty batch response triggers auto-retry.

        Scenario:
        - Batch size 5
        - First call returns {"items": []} (Empty batch failure)
        - Pipeline detects failure (all rows null)
        - Auto-retry runs 5 individual calls (or 1 batch) and succeeds
        """
        # Setup data
        df = pd.DataFrame({"id": range(5), "text": [f"Item {i}" for i in range(5)]})

        # Setup Mock LLM
        # Call 1: Empty batch (Failure)
        # Call 2: Retry (Success - returning individual items or valid batch)

        call_count = 0

        async def mock_invoke(prompt, output_cls=None, **kwargs):
            nonlocal call_count
            call_count += 1

            # First call: Return empty batch (simulating the bug)
            if call_count == 1:
                return LLMResponse(
                    text='{"items": []}',  # Empty list!
                    tokens_in=100,
                    tokens_out=10,
                    model="test-model",
                    cost=Decimal("0.001"),
                    latency_ms=100.0,
                )

            # Subsequent calls (Retry): Return valid data
            # The retry pipeline might run row-by-row or batch depending on config
            # Let's assume it runs as a batch of 5 again (since it's a new pipeline)
            # OR row-by-row if batch_size defaults to 1 in retry (it usually copies specs)

            # Construct valid response based on prompt
            # Simplified logic: just return a valid batch for simplicity
            # (Real logic would need to parse input prompt to match IDs, but let's assume batch retry)

            return LLMResponse(
                text='{"items": [{"id": 1, "result": {"value": "ok"}}, {"id": 2, "result": {"value": "ok"}}, {"id": 3, "result": {"value": "ok"}}, {"id": 4, "result": {"value": "ok"}}, {"id": 5, "result": {"value": "ok"}}]}',
                tokens_in=100,
                tokens_out=100,
                model="test-model",
                cost=Decimal("0.001"),
                latency_ms=100.0,
            )

        # Mock Client
        mock_client = Mock()
        # Mock async methods
        mock_client.structured_invoke_async = AsyncMock(side_effect=mock_invoke)
        mock_client.ainvoke = AsyncMock(side_effect=mock_invoke)  # Fallback
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()

        # We need to mock the attribute access for 'router'
        mock_client.router = None
        mock_client.spec = Mock(
            model="test-model",
            input_cost_per_1k_tokens=Decimal("0.001"),
            output_cost_per_1k_tokens=Decimal("0.001"),
        )

        # Configure Registry to return our mock class
        mock_client_class = Mock(return_value=mock_client)
        mock_get.return_value = mock_client_class

        # Build Pipeline
        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                df,
                input_columns=["text"],
                output_columns=["value"],
            )
            .with_prompt("Process {text}")
            .with_llm(provider="groq", model="test")
            .with_batch_size(5)  # Batch all 5
            .with_structured_output(BatchResponse)
            .with_executor(AsyncExecutor())
            .build()
        )

        # Enable Auto-Retry
        pipeline.specifications.processing.auto_retry_failed = True
        pipeline.specifications.processing.max_retry_attempts = 1

        # Execute
        result = await pipeline.execute_async()

        # Verification
        # 1. Should have succeeded
        assert result.success

        # 2. Should have processed all 5 rows
        assert len(result.data) == 5

        # 3. Should have called LLM at least twice (1 fail + 1 retry)
        assert call_count >= 2, f"Expected at least 2 calls, got {call_count}"

        # 4. Data should be valid (no nulls)
        assert result.data["value"].notnull().all()
        assert (result.data["value"] == "ok").all()

        print(f"\n✅ Batch Recovery Test Passed! Calls: {call_count}")
