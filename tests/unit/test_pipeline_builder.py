"""Unit tests for PipelineBuilder."""

import pandas as pd
import pytest

from ondine import PipelineBuilder
from ondine.core.specifications import (
    DataSourceType,
    LLMProvider,
)


class TestPipelineBuilder:
    """Test suite for PipelineBuilder."""

    def test_builder_creation(self):
        """Test creating a builder."""
        builder = PipelineBuilder.create()
        assert builder is not None

    def test_from_dataframe(self):
        """Test configuring DataFrame source."""
        df = pd.DataFrame({"text": ["test"]})

        builder = PipelineBuilder.create().from_dataframe(
            df,
            input_columns=["text"],
            output_columns=["result"],
        )

        assert builder._dataset_spec is not None
        assert builder._dataset_spec.source_type == DataSourceType.DATAFRAME
        assert builder._dataset_spec.input_columns == ["text"]

    def test_from_csv(self):
        """Test configuring CSV source."""
        builder = PipelineBuilder.create().from_csv(
            "data.csv",
            input_columns=["text"],
            output_columns=["result"],
        )

        assert builder._dataset_spec.source_type == DataSourceType.CSV
        assert str(builder._dataset_spec.source_path) == "data.csv"

    def test_with_prompt(self):
        """Test configuring prompt."""
        builder = PipelineBuilder.create().with_prompt("Process: {text}")

        assert builder._prompt_spec is not None
        assert builder._prompt_spec.template == "Process: {text}"

    def test_with_llm(self):
        """Test configuring LLM."""
        builder = PipelineBuilder.create().with_llm(
            provider="groq",
            model="llama-3.1-70b-versatile",
            temperature=0.5,
        )

        assert builder._llm_spec is not None
        assert builder._llm_spec.provider == LLMProvider.GROQ
        assert builder._llm_spec.model == "llama-3.1-70b-versatile"
        assert builder._llm_spec.temperature == 0.5

    def test_processing_config(self):
        """Test configuring processing parameters."""
        builder = (
            PipelineBuilder.create()
            .with_processing_batch_size(50)  # Internal batching (renamed)
            .with_concurrency(10)
            .with_checkpoint_interval(250)
            .with_rate_limit(30)
            .with_max_budget(5.0)
        )

        assert builder._processing_spec.batch_size == 50
        assert builder._processing_spec.concurrency == 10
        assert builder._processing_spec.checkpoint_interval == 250
        assert builder._processing_spec.rate_limit_rpm == 30
        assert builder._processing_spec.max_budget == pytest.approx(5.0)

    def test_build_complete_pipeline(self):
        """Test building complete pipeline."""
        df = pd.DataFrame({"text": ["test"]})

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                df,
                input_columns=["text"],
                output_columns=["result"],
            )
            .with_prompt("Process: {text}")
            .with_llm(provider="groq", model="llama-3.1-70b-versatile")
            .build()
        )

        assert pipeline is not None
        assert pipeline.specifications is not None

    def test_build_without_dataset_fails(self):
        """Test that building without dataset fails."""
        builder = (
            PipelineBuilder.create()
            .with_prompt("Process: {text}")
            .with_llm(provider="groq", model="llama-3.1-70b-versatile")
        )

        with pytest.raises(ValueError, match="Dataset specification required"):
            builder.build()

    def test_build_without_prompt_fails(self):
        """Test that building without prompt fails."""
        df = pd.DataFrame({"text": ["test"]})

        builder = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_llm(provider="groq", model="llama-3.1-70b-versatile")
        )

        with pytest.raises(ValueError, match="Prompt specification required"):
            builder.build()

    def test_build_without_llm_fails(self):
        """Test that building without LLM fails."""
        df = pd.DataFrame({"text": ["test"]})

        builder = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {text}")
        )

        with pytest.raises(
            ValueError, match="Either LLM specification or custom LLM client required"
        ):
            builder.build()

    def test_fluent_api_chaining(self):
        """Test that fluent API returns self for chaining."""
        df = pd.DataFrame({"text": ["test"]})

        builder = PipelineBuilder.create()
        assert builder.from_dataframe(df, ["text"], ["result"]) is builder
        assert builder.with_prompt("Test: {text}") is builder
        assert builder.with_llm("groq", "llama-3.1-70b-versatile") is builder
        assert builder.with_batch_size(100) is builder
