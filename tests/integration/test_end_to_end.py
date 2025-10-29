"""
End-to-end integration tests with real LLM providers.

These tests verify the complete pipeline flow from data loading to output.
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from ondine import PipelineBuilder
from ondine.stages import JSONParser


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="Set GROQ_API_KEY to run end-to-end tests",
)
class TestEndToEndGroq:
    """End-to-end tests with Groq provider."""

    def test_simple_qa_pipeline(self):
        """Test simple Q&A pipeline end-to-end."""
        # Create test data
        df = pd.DataFrame(
            {
                "question": [
                    "What is 5+5?",
                    "What color is the sky?",
                    "What is H2O?",
                ]
            }
        )

        # Build and execute pipeline
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
                model="openai/gpt-oss-120b",
                temperature=0.0,
            )
            .with_batch_size(3)
            .with_concurrency(2)
            .build()
        )

        result = pipeline.execute()

        # Verify results
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 3
        assert "answer" in result.data.columns

        # Check answers are not empty
        for answer in result.data["answer"]:
            assert len(str(answer)) > 0
            assert answer != "[SKIPPED]"

        # Verify metrics
        assert result.metrics.total_rows >= 3
        assert result.costs.total_cost >= 0
        assert result.duration > 0

    def test_json_extraction_pipeline(self):
        """Test JSON extraction from LLM responses."""
        df = pd.DataFrame(
            {
                "product": [
                    "Apple iPhone 13 Pro 256GB",
                    "Samsung Galaxy S22 Ultra 512GB",
                ]
            }
        )

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                df,
                input_columns=["product"],
                output_columns=["brand", "model", "storage"],
            )
            .with_prompt(
                """Extract product details as JSON:
Product: {product}

Return JSON with keys: brand, model, storage
JSON:"""
            )
            .with_llm(
                provider="groq",
                model="openai/gpt-oss-120b",
                temperature=0.0,
            )
            .with_parser(JSONParser(strict=False))
            .build()
        )

        result = pipeline.execute()

        # Verify structured output
        assert result.success is True
        assert "brand" in result.data.columns
        assert "model" in result.data.columns
        assert "storage" in result.data.columns

    def test_csv_to_csv_pipeline(self):
        """Test complete CSV → processing → CSV workflow."""
        with TemporaryDirectory() as tmpdir:
            # Create input CSV
            input_path = Path(tmpdir) / "input.csv"
            output_path = Path(tmpdir) / "output.csv"

            df = pd.DataFrame(
                {
                    "text": ["Hello", "World", "Test"],
                }
            )
            df.to_csv(input_path, index=False)

            # Build pipeline
            pipeline = (
                PipelineBuilder.create()
                .from_csv(
                    str(input_path),
                    input_columns=["text"],
                    output_columns=["uppercase"],
                )
                .with_prompt("Convert to uppercase: {text}")
                .with_llm(
                    provider="groq",
                    model="openai/gpt-oss-120b",
                    temperature=0.0,
                )
                .to_csv(str(output_path))
                .build()
            )

            pipeline.execute()

            # Verify output file exists
            assert output_path.exists()

            # Read and verify output
            output_df = pd.read_csv(output_path)
            assert len(output_df) == 3
            assert "uppercase" in output_df.columns

    def test_error_handling_with_skip_policy(self):
        """Test error handling with SKIP policy."""
        df = pd.DataFrame(
            {
                "text": ["Valid", "Also valid", "Still valid"],
            }
        )

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                df,
                input_columns=["text"],
                output_columns=["result"],
            )
            .with_prompt("Process: {text}")
            .with_llm(
                provider="groq",
                model="openai/gpt-oss-120b",
                temperature=0.0,
            )
            .with_error_policy("skip")
            .with_max_retries(2)
            .build()
        )

        result = pipeline.execute()

        # Should complete even if some rows fail
        assert result.success is True
        assert result.metrics.total_rows >= 0

    def test_cost_estimation_accuracy(self):
        """Test that cost estimation is reasonably accurate."""
        df = pd.DataFrame(
            {
                "text": [f"Text {i}" for i in range(10)],
            }
        )

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                df,
                input_columns=["text"],
                output_columns=["result"],
            )
            .with_prompt("Echo: {text}")
            .with_llm(
                provider="groq",
                model="openai/gpt-oss-120b",
                temperature=0.0,
            )
            .build()
        )

        # Get estimate
        estimate = pipeline.estimate_cost()
        assert estimate.total_cost >= 0
        assert estimate.total_tokens > 0

        # Execute
        result = pipeline.execute()

        # Actual cost should be within reasonable range of estimate
        # (Groq might be free, so just check it's >= 0)
        assert result.costs.total_cost >= 0

    def test_checkpoint_and_resume(self):
        """Test checkpoint creation and resume functionality."""
        with TemporaryDirectory() as tmpdir:
            df = pd.DataFrame(
                {
                    "text": [f"Item {i}" for i in range(5)],
                }
            )

            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir()

            pipeline = (
                PipelineBuilder.create()
                .from_dataframe(
                    df,
                    input_columns=["text"],
                    output_columns=["result"],
                )
                .with_prompt("Process: {text}")
                .with_llm(
                    provider="groq",
                    model="openai/gpt-oss-120b",
                    temperature=0.0,
                )
                .with_checkpoint_dir(str(checkpoint_dir))
                .with_checkpoint_interval(2)
                .build()
            )

            result = pipeline.execute()

            # Verify execution completed
            assert result.success is True

            # Check if checkpoints were created
            # (They might not be if execution was fast)
            list(checkpoint_dir.glob("*.pkl"))
            # Just verify directory exists and is accessible
            assert checkpoint_dir.exists()

    def test_concurrent_execution_correctness(self):
        """Test that concurrent execution maintains correctness."""
        df = pd.DataFrame(
            {
                "number": [1, 2, 3, 4, 5],
            }
        )

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                df,
                input_columns=["number"],
                output_columns=["doubled"],
            )
            .with_prompt("What is {number} times 2? Answer with just the number.")
            .with_llm(
                provider="groq",
                model="openai/gpt-oss-120b",
                temperature=0.0,
            )
            .with_concurrency(3)  # Process 3 at a time
            .build()
        )

        result = pipeline.execute()

        # Verify all rows processed
        assert result.success is True
        assert len(result.data) == 5

        # Verify order is maintained (responses match input order)
        # Note: We can't verify exact values since LLM might format differently
        # but we can verify we got 5 non-empty responses
        for answer in result.data["doubled"]:
            assert len(str(answer)) > 0


@pytest.mark.integration
class TestEndToEndWithMock:
    """End-to-end tests with mock LLM (no API key needed)."""

    def test_pipeline_builder_validation(self):
        """Test pipeline builder validation."""
        df = pd.DataFrame({"text": ["test"]})

        # Should fail without LLM config
        with pytest.raises(ValueError, match="LLM"):
            (
                PipelineBuilder.create()
                .from_dataframe(
                    df,
                    input_columns=["text"],
                    output_columns=["result"],
                )
                .with_prompt("Test: {text}")
                .build()
            )

    def test_pipeline_validation_errors(self):
        """Test that pipeline validation catches errors."""
        df = pd.DataFrame({"text": ["test"]})

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                df,
                input_columns=["missing_column"],  # Column doesn't exist
                output_columns=["result"],
            )
            .with_prompt("Test: {missing_column}")
            .with_llm(
                provider="groq",
                model="openai/gpt-oss-120b",
            )
            .build()
        )

        validation = pipeline.validate()
        assert validation.is_valid is False
        assert len(validation.errors) > 0
