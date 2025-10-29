"""
Integration tests for Groq provider.

These tests make real API calls and require GROQ_API_KEY.
"""

import os

import pandas as pd
import pytest

from ondine import PipelineBuilder


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="Set GROQ_API_KEY to run Groq integration tests",
)
class TestGroqIntegration:
    """Integration tests for Groq LLM provider."""

    def test_simple_completion(self):
        """Test basic completion with Groq."""
        df = pd.DataFrame({"text": ["What is 2+2?"]})

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                df,
                input_columns=["text"],
                output_columns=["answer"],
            )
            .with_prompt("{text}")
            .with_llm(
                provider="groq",
                model="llama-3.3-70b-versatile",  # Updated to current Groq model
                temperature=0.0,
            )
            .build()
        )

        result = pipeline.execute()

        assert len(result.data) == 1
        assert result.metrics.processed_rows == 1
        # Groq API returns $0 cost but does track tokens
        assert result.costs.total_cost >= 0
        assert result.costs.total_tokens > 0
        # Verify we got a valid answer (not skipped)
        assert result.data["answer"].iloc[0] != "[SKIPPED]"
        assert len(result.data["answer"].iloc[0]) > 0

    def test_batch_processing(self):
        """Test batch processing with Groq."""
        df = pd.DataFrame(
            {
                "question": [
                    "What is the capital of France?",
                    "What is 5+3?",
                    "Name a color",
                ]
            }
        )

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                df,
                input_columns=["question"],
                output_columns=["answer"],
            )
            .with_prompt("Answer briefly: {question}")
            .with_llm(
                provider="groq",
                model="llama-3.3-70b-versatile",  # Updated to current Groq model
                temperature=0.0,
            )
            .with_batch_size(10)
            .with_concurrency(2)
            .build()
        )

        result = pipeline.execute()

        assert len(result.data) == 3
        assert result.metrics.processed_rows == 3
        assert "answer" in result.data.columns

    def test_cost_tracking(self):
        """Test cost tracking with Groq."""
        df = pd.DataFrame({"text": ["Hello world"]})

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                df,
                input_columns=["text"],
                output_columns=["response"],
            )
            .with_prompt("Echo: {text}")
            .with_llm(
                provider="groq", model="llama-3.3-70b-versatile"
            )  # Updated to current Groq model
            .build()
        )

        # Get estimate
        estimate = pipeline.estimate_cost()
        assert estimate.total_cost >= 0

        # Execute
        result = pipeline.execute()

        # Groq API returns $0 cost but does track tokens
        assert result.costs.total_cost >= 0
        assert result.costs.total_tokens > 0
        # Verify we got valid output
        assert result.data["response"].iloc[0] != "[SKIPPED]"
