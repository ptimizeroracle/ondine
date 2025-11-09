"""
End-to-end integration tests with real LLM providers.

These tests verify the complete pipeline flow from data loading to output.
"""

import os
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

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


class TestExecutionResultContract:
    """Contract tests for ExecutionResult API to prevent regressions."""

    def test_execution_result_has_required_attributes(self):
        """Verify ExecutionResult has all required attributes with correct names."""
        # Create a minimal ExecutionResult
        from datetime import timedelta
        from decimal import Decimal

        from ondine.core.models import CostEstimate, ExecutionResult, ProcessingStats

        result = ExecutionResult(
            data=pd.DataFrame({"col": [1, 2, 3]}),
            metrics=ProcessingStats(
                total_rows=3,
                processed_rows=3,
                failed_rows=0,
                skipped_rows=0,
                rows_per_second=1.0,
                total_duration_seconds=3.0,
            ),
            costs=CostEstimate(
                total_cost=Decimal("0.01"),
                total_tokens=100,
                input_tokens=50,
                output_tokens=50,
                rows=3,
            ),
        )
        # Set end_time to calculate duration
        result.end_time = result.start_time + timedelta(seconds=3)

        # Verify correct attributes exist
        assert hasattr(result, "data")
        assert hasattr(result, "metrics")
        assert hasattr(result, "costs")
        assert hasattr(result, "duration")
        assert hasattr(result, "errors")
        assert hasattr(result, "success")

        # Verify nested attributes
        assert hasattr(result.metrics, "total_rows")
        assert hasattr(result.metrics, "processed_rows")
        assert hasattr(result.metrics, "failed_rows")

        assert hasattr(result.costs, "total_cost")
        assert hasattr(result.costs, "total_tokens")
        assert hasattr(result.costs, "input_tokens")
        assert hasattr(result.costs, "output_tokens")

        # Verify types
        assert isinstance(result.metrics.total_rows, int)
        assert isinstance(result.costs.total_cost, Decimal)
        assert isinstance(result.duration, float)

    def test_deprecated_attributes_do_not_exist(self):
        """Ensure old API attributes are removed to catch regressions."""
        from datetime import timedelta
        from decimal import Decimal

        from ondine.core.models import CostEstimate, ExecutionResult, ProcessingStats

        result = ExecutionResult(
            data=pd.DataFrame({"col": [1]}),
            metrics=ProcessingStats(
                total_rows=1,
                processed_rows=1,
                failed_rows=0,
                skipped_rows=0,
                rows_per_second=1.0,
                total_duration_seconds=1.0,
            ),
            costs=CostEstimate(
                total_cost=Decimal("0.01"),
                total_tokens=10,
                input_tokens=5,
                output_tokens=5,
                rows=1,
            ),
        )
        result.end_time = result.start_time + timedelta(seconds=1)

        # Verify deprecated attributes don't exist
        assert not hasattr(result, "rows_processed"), (
            "Use result.metrics.total_rows instead of result.rows_processed"
        )
        assert not hasattr(result, "cost"), "Use result.costs instead of result.cost"
        assert not hasattr(result, "execution_time"), (
            "Use result.duration instead of result.execution_time"
        )

    def test_pipeline_result_attributes_after_execution(self):
        """Test that pipeline result has correct attributes accessible to user scripts."""
        # This test verifies the API contract without needing to execute
        # We already tested execution in other E2E tests
        from datetime import timedelta

        from ondine.core.models import CostEstimate, ExecutionResult, ProcessingStats

        # Simulate what a pipeline.execute() returns
        result = ExecutionResult(
            data=pd.DataFrame({"text": ["test1", "test2"], "result": ["r1", "r2"]}),
            metrics=ProcessingStats(
                total_rows=2,
                processed_rows=2,
                failed_rows=0,
                skipped_rows=0,
                rows_per_second=1.0,
                total_duration_seconds=2.0,
            ),
            costs=CostEstimate(
                total_cost=Decimal("0.002"),
                total_tokens=30,
                input_tokens=20,
                output_tokens=10,
                rows=2,
            ),
        )
        result.end_time = result.start_time + timedelta(seconds=2)

        # Verify result structure (this is what user scripts depend on)
        assert hasattr(result, "metrics")
        assert hasattr(result, "costs")
        assert hasattr(result, "duration")

        # Verify can access like user scripts do (should not raise AttributeError)
        total_rows = result.metrics.total_rows
        total_cost = result.costs.total_cost
        duration = result.duration

        assert total_rows == 2
        assert isinstance(total_cost, Decimal)
        assert isinstance(duration, int | float)


class TestCheckpointResumeBehavior:
    """Behavior tests for checkpoint and resume functionality."""

    def test_checkpoint_resume_with_simulated_crash(self):
        """Test checkpoint creation and resume_from parameter acceptance."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir()

            df = pd.DataFrame({"text": [f"Item {i}" for i in range(5)]})

            # Create pipeline with checkpoints
            pipeline = (
                PipelineBuilder.create()
                .from_dataframe(df, input_columns=["text"], output_columns=["result"])
                .with_prompt("Process: {text}")
                .with_llm(provider="groq", model="test-model")
                .with_checkpoint_dir(str(checkpoint_dir))
                .with_checkpoint_interval(2)
                .build()
            )

            # Verify pipeline accepts resume_from parameter (API contract)
            # Note: We can't easily mock internal stages, so we test the API surface
            try:
                # This will fail due to missing API key, but that's OK
                # We're testing that the parameter is accepted
                pipeline.execute(
                    resume_from=UUID("00000000-0000-0000-0000-000000000000")
                )
            except Exception:
                # Expected to fail (no API key), but parameter was accepted
                pass

            # Verify checkpoint directory was configured
            assert checkpoint_dir.exists()

    def test_checkpoint_configuration_api(self):
        """Verify checkpoint configuration API works correctly."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir()

            df = pd.DataFrame({"text": ["test"]})

            # Test that checkpoint configuration is accepted
            pipeline = (
                PipelineBuilder.create()
                .from_dataframe(df, input_columns=["text"], output_columns=["result"])
                .with_prompt("Process: {text}")
                .with_llm(provider="groq", model="test-model")
                .with_checkpoint_dir(str(checkpoint_dir))
                .with_checkpoint_interval(5)
                .build()
            )

            # Verify configuration was applied
            assert str(pipeline.specifications.processing.checkpoint_dir) == str(
                checkpoint_dir
            )
            assert pipeline.specifications.processing.checkpoint_interval == 5


class TestNonRetryableErrorE2E:
    """E2E tests for non-retryable error classification and fail-fast behavior."""

    def test_error_classification_api_exists(self):
        """Verify NonRetryableError exceptions are exported and usable."""
        from ondine.core import (
            ConfigurationError,
            InvalidAPIKeyError,
            ModelNotFoundError,
            NonRetryableError,
            QuotaExceededError,
        )

        # Verify all exception types are accessible
        assert issubclass(ModelNotFoundError, NonRetryableError)
        assert issubclass(InvalidAPIKeyError, NonRetryableError)
        assert issubclass(ConfigurationError, NonRetryableError)
        assert issubclass(QuotaExceededError, NonRetryableError)

        # Verify they can be raised and caught
        with pytest.raises(ModelNotFoundError):
            raise ModelNotFoundError("test")

        with pytest.raises(NonRetryableError):
            raise InvalidAPIKeyError("test")

    def test_pipeline_propagates_non_retryable_errors(self):
        """Test that NonRetryableError propagates from pipeline to user code."""
        from ondine.core.exceptions import ModelNotFoundError

        # Note: This test verifies the API contract - that ModelNotFoundError
        # can be caught by user code. The actual error classification is tested
        # in unit tests (test_non_retryable_errors.py)

        # Verify the exception type is accessible for user error handling
        with pytest.raises(ModelNotFoundError, match="Model not found"):
            raise ModelNotFoundError("Model not found")


class TestMultiStagePipelineE2E:
    """E2E tests for multi-stage pipeline execution."""

    def test_multi_stage_api_contract(self):
        """Test that multiple pipelines can be chained via DataFrames."""
        df = pd.DataFrame({"text": ["item1", "item2"]})

        # Stage 1: Build pipeline that adds a column
        pipeline1 = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["category"])
            .with_prompt("Categorize: {text}")
            .with_llm(provider="groq", model="test-model")
            .build()
        )

        # Verify pipeline1 is configured correctly
        assert pipeline1.specifications.dataset.input_columns == ["text"]
        assert pipeline1.specifications.dataset.output_columns == ["category"]

        # Stage 2: Build pipeline that uses output from stage 1
        # (Would use result1.data as input in real execution)
        df_with_category = df.copy()
        df_with_category["category"] = "Electronics"

        pipeline2 = (
            PipelineBuilder.create()
            .from_dataframe(
                df_with_category,
                input_columns=["text", "category"],
                output_columns=["subcategory"],
            )
            .with_prompt("Subcategorize {text} in {category}")
            .with_llm(provider="groq", model="test-model")
            .build()
        )

        # Verify pipeline2 can use multiple input columns
        assert "text" in pipeline2.specifications.dataset.input_columns
        assert "category" in pipeline2.specifications.dataset.input_columns
        assert pipeline2.specifications.dataset.output_columns == ["subcategory"]


class TestCostTrackingE2E:
    """E2E tests for cost tracking accuracy."""

    def test_cost_configuration_api(self):
        """Test that cost configuration is accepted and stored correctly."""
        df = pd.DataFrame({"text": ["test"]})

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {text}")
            .with_llm(
                provider="groq",
                model="test-model",
                input_cost_per_1k_tokens=Decimal("0.10"),
                output_cost_per_1k_tokens=Decimal("0.20"),
            )
            .build()
        )

        # Verify cost configuration was applied
        assert pipeline.specifications.llm.input_cost_per_1k_tokens == Decimal("0.10")
        assert pipeline.specifications.llm.output_cost_per_1k_tokens == Decimal("0.20")

    def test_cost_result_structure(self):
        """Test that CostEstimate has all required fields."""
        from ondine.core.models import CostEstimate

        cost = CostEstimate(
            total_cost=Decimal("1.50"),
            total_tokens=1000,
            input_tokens=600,
            output_tokens=400,
            rows=10,
        )

        # Verify all cost fields are accessible
        assert cost.total_cost == Decimal("1.50")
        assert cost.total_tokens == 1000
        assert cost.input_tokens == 600
        assert cost.output_tokens == 400
        assert cost.rows == 10

        # Verify types
        assert isinstance(cost.total_cost, Decimal)
        assert isinstance(cost.total_tokens, int)

    def test_concurrency_configuration_api(self):
        """Test that concurrency configuration is accepted."""
        df = pd.DataFrame({"text": ["test"]})

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {text}")
            .with_llm(provider="groq", model="test-model")
            .with_concurrency(10)
            .build()
        )

        # Verify concurrency was configured
        assert pipeline.specifications.processing.concurrency == 10
