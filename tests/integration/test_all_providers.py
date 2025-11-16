"""
Integration tests for all supported providers.

Tests that all providers work with the same interface.
"""

import os

import pandas as pd
import pytest

from ondine import PipelineBuilder


def get_provider_configs():
    """Get available provider configurations."""
    configs = []

    if os.getenv("GROQ_API_KEY"):
        configs.append(("groq", "llama-3.3-70b-versatile", "GROQ_API_KEY"))

    if os.getenv("OPENAI_API_KEY"):
        configs.append(("openai", "gpt-4o-mini", "OPENAI_API_KEY"))

    if os.getenv("ANTHROPIC_API_KEY"):
        configs.append(("anthropic", "claude-3-haiku-20240307", "ANTHROPIC_API_KEY"))

    return configs


@pytest.mark.integration
@pytest.mark.parametrize(("provider", "model", "key_name"), get_provider_configs())
class TestAllProviders:
    """Test all providers with same interface."""

    def test_provider_execution(self, provider, model, key_name):
        """Test that provider can execute pipeline."""
        df = pd.DataFrame({"text": ["Say hello"]})

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                df,
                input_columns=["text"],
                output_columns=["response"],
            )
            .with_prompt("{text}")
            .with_llm(provider=provider, model=model, temperature=0.0)
            .build()
        )

        result = pipeline.execute()

        # Verify result
        assert len(result.data) == 1
        assert "response" in result.data.columns
        assert result.metrics.processed_rows == 1
        assert result.costs.total_cost >= 0

    def test_provider_cost_estimation(self, provider, model, key_name):
        """Test cost estimation for provider."""
        df = pd.DataFrame({"text": ["Test"]})

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                df,
                input_columns=["text"],
                output_columns=["result"],
            )
            .with_prompt("Echo: {text}")
            .with_llm(provider=provider, model=model)
            .build()
        )

        estimate = pipeline.estimate_cost()

        assert estimate.total_cost >= 0
        assert estimate.total_tokens > 0
        assert estimate.rows == 1
