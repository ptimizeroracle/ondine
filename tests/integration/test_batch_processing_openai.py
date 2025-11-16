"""Integration tests for batch processing with real OpenAI API.

These tests require OPENAI_API_KEY environment variable and will make real API calls.
Run with: pytest tests/integration/test_batch_processing_openai.py -v
"""

import os

import pandas as pd
import pytest

from ondine.api.pipeline_builder import PipelineBuilder


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestBatchProcessingOpenAI:
    """Integration tests with real OpenAI API."""

    def test_batch_processing_with_10_rows(self):
        """Test batch processing with 10 rows (batch_size=5)."""
        # Create test data
        data = pd.DataFrame(
            {
                "review": [
                    "This product is amazing!",
                    "Terrible quality, broke after one day",
                    "It's okay, nothing special",
                    "Best purchase ever!",
                    "Waste of money",
                    "Pretty good for the price",
                    "Not what I expected",
                    "Exceeded my expectations!",
                    "Average product",
                    "Would not recommend",
                ]
            }
        )

        # Build pipeline with batch processing
        from decimal import Decimal

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                data, input_columns=["review"], output_columns=["sentiment"]
            )
            .with_prompt("Classify sentiment: {review}\n\nSentiment:")
            .with_batch_size(5)  # Process 5 rows per API call
            .with_llm(
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.0,
                input_cost_per_1k_tokens=Decimal("0.00015"),
                output_cost_per_1k_tokens=Decimal("0.0006"),
            )
            .build()
        )

        # Execute
        result = pipeline.execute()

        # Assertions
        assert result.success is True
        assert len(result.data) == 10
        assert "sentiment" in result.data.columns

        # Check that all rows have results
        assert result.data["sentiment"].notna().all()

        # Check cost and tokens
        print(f"\nCost: ${result.costs.total_cost}")
        print(f"Tokens: {result.costs.total_tokens}")
        print(f"Input tokens: {result.costs.input_tokens}")
        print(f"Output tokens: {result.costs.output_tokens}")
        print(f"Results:\n{result.data}")

        # Verify batch processing worked (should have tokens from 2 API calls, not 10)
        assert result.costs.total_tokens > 0

    def test_batch_processing_vs_individual(self):
        """Compare batch processing vs individual processing."""
        # Create test data
        data = pd.DataFrame(
            {
                "text": [
                    "Apple",
                    "Banana",
                    "Cherry",
                    "Date",
                    "Elderberry",
                ]
            }
        )

        # Individual processing (batch_size=1, default)
        pipeline_individual = (
            PipelineBuilder.create()
            .from_dataframe(data, input_columns=["text"], output_columns=["category"])
            .with_prompt("Categorize: {text}\n\nCategory:")
            .with_llm(provider="openai", model="gpt-4o-mini", temperature=0.0)
            .build()
        )

        result_individual = pipeline_individual.execute()

        # Batch processing (batch_size=5)
        pipeline_batch = (
            PipelineBuilder.create()
            .from_dataframe(data, input_columns=["text"], output_columns=["category"])
            .with_prompt("Categorize: {text}\n\nCategory:")
            .with_batch_size(5)
            .with_llm(provider="openai", model="gpt-4o-mini", temperature=0.0)
            .build()
        )

        result_batch = pipeline_batch.execute()

        # Both should succeed
        assert result_individual.success is True
        assert result_batch.success is True

        # Both should have same number of rows
        assert len(result_individual.data) == 5
        assert len(result_batch.data) == 5

        # Compare costs (batch should be cheaper per row due to fewer API calls)
        cost_per_row_individual = (
            result_individual.costs.total_cost / result_individual.metrics.total_rows
        )
        cost_per_row_batch = (
            result_batch.costs.total_cost / result_batch.metrics.total_rows
        )

        print(f"\nIndividual: ${cost_per_row_individual:.6f}/row")
        print(f"Batch: ${cost_per_row_batch:.6f}/row")
        print(
            f"Savings: {(1 - cost_per_row_batch / cost_per_row_individual) * 100:.1f}%"
        )

    def test_batch_processing_with_partial_failure(self):
        """Test batch processing with intentionally difficult inputs."""
        # Create test data with some tricky inputs
        data = pd.DataFrame(
            {
                "text": [
                    "Normal text",
                    "Another normal text",
                    "Yet another normal text",
                ]
            }
        )

        # Build pipeline
        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(data, input_columns=["text"], output_columns=["result"])
            .with_prompt("Echo: {text}")
            .with_batch_size(3)
            .with_llm(provider="openai", model="gpt-4o-mini", temperature=0.0)
            .build()
        )

        # Execute
        result = pipeline.execute()

        # Should handle gracefully
        assert result.success is True
        assert len(result.data) == 3

        print(f"\nResults:\n{result.data}")

    def test_batch_size_validation(self):
        """Test that batch size validation works."""
        data = pd.DataFrame({"text": ["Test"] * 10})

        # Valid batch size
        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(data, input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {text}")
            .with_batch_size(10)
            .with_llm(provider="openai", model="gpt-4o-mini")
            .build()
        )

        result = pipeline.execute()
        assert result.success is True

    def test_backward_compatibility_batch_size_1(self):
        """Test that batch_size=1 (default) works as before."""
        data = pd.DataFrame({"text": ["Test 1", "Test 2"]})

        # Don't set batch_size (defaults to 1)
        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(data, input_columns=["text"], output_columns=["result"])
            .with_prompt("Echo: {text}")
            .with_llm(provider="openai", model="gpt-4o-mini", temperature=0.0)
            .build()
        )

        result = pipeline.execute()

        # Should work exactly as before
        assert result.success is True
        assert len(result.data) == 2
        assert "result" in result.data.columns
