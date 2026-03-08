"""Integration tests for budget enforcement behavior."""

from decimal import Decimal
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from ondine import PipelineBuilder
from ondine.core.models import LLMResponse
from ondine.utils.budget_controller import BudgetExceededError


def _row_from_prompt(prompt: str) -> int:
    for i in range(10):
        if f"Item {i}" in prompt:
            return i
    raise AssertionError(f"Unknown prompt: {prompt}")


@pytest.mark.integration
@patch("ondine.adapters.provider_registry.ProviderRegistry.get")
def test_budget_enforcement_stops_mid_run(mock_get):
    """
    Regression this catches:
    the pipeline must stop as soon as accumulated cost crosses the budget,
    not after finishing the whole dataset.
    """
    df = pd.DataFrame({"text": [f"Item {i}" for i in range(5)]})
    processed_rows: list[int] = []

    async def mock_invoke(prompt, **kwargs):
        row_num = _row_from_prompt(prompt)
        processed_rows.append(row_num)
        return LLMResponse(
            text=f"summary_{row_num}",
            tokens_in=10,
            tokens_out=10,
            model="test-model",
            cost=Decimal("0.40"),
            latency_ms=5.0,
        )

    mock_client = Mock()
    mock_client.ainvoke = AsyncMock(side_effect=mock_invoke)
    mock_client.start = AsyncMock()
    mock_client.stop = AsyncMock()
    mock_client.router = None
    mock_client.model = "test-model"
    mock_client.spec = Mock(model="test-model")
    mock_get.return_value = Mock(return_value=mock_client)

    with TemporaryDirectory() as tmpdir:
        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["summary"])
            .with_prompt("Summarize: {text}")
            .with_llm(provider="openai", model="test-model", temperature=0.0)
            .with_processing_batch_size(1)
            .with_concurrency(1)
            .with_max_budget(0.75)
            .with_checkpoint_dir(tmpdir)
            .build()
        )

        with pytest.raises(BudgetExceededError, match="Budget exceeded"):
            pipeline.execute()

        assert processed_rows == [0, 1]


@pytest.mark.integration
@patch("ondine.utils.budget_controller.logger.warning")
@patch("ondine.adapters.provider_registry.ProviderRegistry.get")
def test_budget_warning_threshold_emits_warning(mock_get, mock_warning):
    """
    Regression this catches:
    warning thresholds must be emitted before the hard budget limit is reached.
    """
    df = pd.DataFrame({"text": [f"Item {i}" for i in range(3)]})

    async def mock_invoke(prompt, **kwargs):
        row_num = _row_from_prompt(prompt)
        return LLMResponse(
            text=f"summary_{row_num}",
            tokens_in=10,
            tokens_out=10,
            model="test-model",
            cost=Decimal("0.16"),
            latency_ms=5.0,
        )

    mock_client = Mock()
    mock_client.ainvoke = AsyncMock(side_effect=mock_invoke)
    mock_client.start = AsyncMock()
    mock_client.stop = AsyncMock()
    mock_client.router = None
    mock_client.model = "test-model"
    mock_client.spec = Mock(model="test-model")
    mock_get.return_value = Mock(return_value=mock_client)

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["text"], output_columns=["summary"])
        .with_prompt("Summarize: {text}")
        .with_llm(provider="openai", model="test-model", temperature=0.0)
        .with_processing_batch_size(1)
        .with_concurrency(1)
        .with_max_budget(0.50)
        .build()
    )

    result = pipeline.execute()

    assert result.success
    assert result.costs.total_cost == Decimal("0.48")
    warning_messages = [call.args[0] for call in mock_warning.call_args_list]
    assert any("75% used" in message for message in warning_messages)
    assert any("90% used" in message for message in warning_messages)
