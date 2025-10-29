"""Unit tests for Pipeline class."""

from decimal import Decimal

import pandas as pd

from ondine.api.pipeline import Pipeline
from ondine.core.specifications import (
    DatasetSpec,
    DataSourceType,
    LLMProvider,
    LLMSpec,
    PipelineSpecifications,
    PromptSpec,
)


class TestPipeline:
    """Test suite for Pipeline class."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization with specifications."""
        specs = PipelineSpecifications(
            dataset=DatasetSpec(
                source_type=DataSourceType.DATAFRAME,
                input_columns=["text"],
                output_columns=["result"],
            ),
            prompt=PromptSpec(template="Process: {text}"),
            llm=LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
        )

        pipeline = Pipeline(specs)

        assert pipeline.id is not None
        assert pipeline.specifications == specs
        assert pipeline.observers == []

    def test_pipeline_with_dataframe(self):
        """Test pipeline with pre-loaded DataFrame."""
        df = pd.DataFrame({"text": ["test"]})
        specs = PipelineSpecifications(
            dataset=DatasetSpec(
                source_type=DataSourceType.DATAFRAME,
                input_columns=["text"],
                output_columns=["result"],
            ),
            prompt=PromptSpec(template="{text}"),
            llm=LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
        )

        pipeline = Pipeline(specs, dataframe=df)

        assert pipeline.dataframe is not None
        assert len(pipeline.dataframe) == 1

    def test_validate_valid_config(self):
        """Test validation with valid configuration."""
        specs = PipelineSpecifications(
            dataset=DatasetSpec(
                source_type=DataSourceType.DATAFRAME,
                input_columns=["text"],
                output_columns=["result"],
            ),
            prompt=PromptSpec(template="{text}"),
            llm=LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
        )

        pipeline = Pipeline(specs)
        validation = pipeline.validate()

        assert validation.is_valid is True
        assert len(validation.errors) == 0

    def test_validate_missing_template_variable(self):
        """Test validation catches missing template variables."""
        specs = PipelineSpecifications(
            dataset=DatasetSpec(
                source_type=DataSourceType.DATAFRAME,
                input_columns=["text"],
                output_columns=["result"],
            ),
            prompt=PromptSpec(template="{missing_var}"),  # Variable not in input
            llm=LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
        )

        pipeline = Pipeline(specs)
        validation = pipeline.validate()

        assert validation.is_valid is False
        assert len(validation.errors) > 0

    def test_estimate_cost_with_sample(self):
        """Test cost estimation with sample data."""
        import os

        df = pd.DataFrame({"text": [f"Sample {i}" for i in range(100)]})

        # Set a dummy API key for cost estimation (doesn't need to be real)
        os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key-for-estimation"

        specs = PipelineSpecifications(
            dataset=DatasetSpec(
                source_type=DataSourceType.DATAFRAME,
                input_columns=["text"],
                output_columns=["result"],
            ),
            prompt=PromptSpec(template="{text}"),
            llm=LLMSpec(
                provider=LLMProvider.OPENAI,
                model="gpt-4o-mini",
                input_cost_per_1k_tokens=Decimal("0.00015"),
                output_cost_per_1k_tokens=Decimal("0.0006"),
            ),
        )

        try:
            pipeline = Pipeline(specs, dataframe=df)
            estimate = pipeline.estimate_cost()

            assert estimate.total_cost >= 0
            assert estimate.rows == 100
        finally:
            # Clean up
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

    def test_add_observer(self):
        """Test adding observers to pipeline."""
        from ondine.orchestration import LoggingObserver

        specs = PipelineSpecifications(
            dataset=DatasetSpec(
                source_type=DataSourceType.DATAFRAME,
                input_columns=["text"],
                output_columns=["result"],
            ),
            prompt=PromptSpec(template="{text}"),
            llm=LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
        )

        pipeline = Pipeline(specs)
        observer = LoggingObserver()
        pipeline.add_observer(observer)

        assert len(pipeline.observers) == 1
        assert pipeline.observers[0] == observer

    def test_pipeline_with_executor(self):
        """Test pipeline with custom executor."""
        from ondine.orchestration import SyncExecutor

        specs = PipelineSpecifications(
            dataset=DatasetSpec(
                source_type=DataSourceType.DATAFRAME,
                input_columns=["text"],
                output_columns=["result"],
            ),
            prompt=PromptSpec(template="{text}"),
            llm=LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
        )

        executor = SyncExecutor()
        pipeline = Pipeline(specs, executor=executor)

        assert pipeline.executor == executor
