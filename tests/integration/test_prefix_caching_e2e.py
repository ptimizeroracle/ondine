"""Integration tests for the prefix-caching request path."""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from ondine import PipelineBuilder


def _fake_completion_response(text: str, cost: Decimal, cached_tokens: int):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
            cache_read_input_tokens=cached_tokens,
        ),
        model="gpt-4o-mini",
        _hidden_params={"response_cost": float(cost)},
    )


@pytest.mark.integration
@patch("ondine.adapters.unified_litellm_client.logger.debug")
@patch("ondine.adapters.unified_litellm_client.litellm.acompletion")
def test_prefix_caching_separates_system_message_and_detects_cache_hits(
    mock_acompletion, mock_debug
):
    """
    Regression this catches:
    system prompts must be sent separately and provider-reported cache hits
    must flow through the unified client path.
    """
    captured_messages = []

    async def fake_acompletion(**kwargs):
        captured_messages.append(kwargs["messages"])
        if len(captured_messages) == 1:
            return _fake_completion_response("positive", Decimal("0.010"), 0)
        return _fake_completion_response("neutral", Decimal("0.005"), 80)

    mock_acompletion.side_effect = fake_acompletion

    system_message = "Classify sentiment. Return one short label only."
    df = pd.DataFrame({"text": ["I love this", "It is okay"]})
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["text"], output_columns=["sentiment"])
        .with_prompt("Analyze: {text}", system_message=system_message)
        .with_llm(
            model="openai/gpt-4o-mini", temperature=0.0, enable_prefix_caching=True
        )
        .with_processing_batch_size(1)
        .with_concurrency(1)
        .build()
    )

    result = pipeline.execute()
    output = result.to_pandas()

    assert result.success
    assert output["sentiment"].tolist() == ["positive", "neutral"]
    assert result.costs.total_cost == Decimal("0.015")
    assert result.costs.input_tokens == 200
    assert len(captured_messages) == 2
    assert all(messages[0]["role"] == "system" for messages in captured_messages)
    assert all(
        messages[0]["content"] == system_message for messages in captured_messages
    )
    assert all(messages[1]["role"] == "user" for messages in captured_messages)
    assert any("Cache hit!" in call.args[0] for call in mock_debug.call_args_list)


@pytest.mark.integration
@patch("ondine.adapters.unified_litellm_client.litellm.acompletion")
def test_prefix_caching_cost_reduction_is_reflected_in_pipeline_totals(
    mock_acompletion,
):
    """
    Regression this catches:
    provider-reported lower cached-call cost must be reflected in final totals.
    """
    costs = [Decimal("0.012"), Decimal("0.004")]

    async def fake_acompletion(**kwargs):
        idx = fake_acompletion.call_index
        fake_acompletion.call_index += 1
        cached_tokens = 0 if idx == 0 else 90
        return _fake_completion_response(f"answer_{idx}", costs[idx], cached_tokens)

    fake_acompletion.call_index = 0
    mock_acompletion.side_effect = fake_acompletion

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            pd.DataFrame({"question": ["Q1", "Q2"]}),
            input_columns=["question"],
            output_columns=["answer"],
        )
        .with_prompt("{question}", system_message="Answer succinctly.")
        .with_llm(
            model="openai/gpt-4o-mini", temperature=0.0, enable_prefix_caching=True
        )
        .with_processing_batch_size(1)
        .with_concurrency(1)
        .build()
    )

    result = pipeline.execute()

    assert result.success
    assert result.costs.total_cost == sum(costs, Decimal("0"))
    assert result.to_pandas()["answer"].tolist() == ["answer_0", "answer_1"]
