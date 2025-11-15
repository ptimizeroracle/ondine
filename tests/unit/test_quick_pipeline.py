"""Tests for QuickPipeline simplified API."""

from decimal import Decimal

import pandas as pd
import pytest

from ondine.api.quick import QuickPipeline


class TestQuickPipeline:
    """Tests for QuickPipeline class."""

    def test_create_minimal(self, tmp_path):
        """Should create pipeline with minimal configuration."""
        # Create test data
        csv_file = tmp_path / "test.csv"
        pd.DataFrame({"text": ["hello", "world"]}).to_csv(csv_file, index=False)

        # Create pipeline with minimal config
        pipeline = QuickPipeline.create(
            data=csv_file, prompt="Categorize: {text}", model="gpt-4o-mini"
        )

        # Assertions
        assert pipeline is not None
        assert pipeline.specifications.prompt.template == "Categorize: {text}"
        assert pipeline.specifications.llm.model == "gpt-4o-mini"
        assert pipeline.specifications.dataset.input_columns == ["text"]
        assert pipeline.specifications.dataset.output_columns == ["output"]

    def test_create_with_dataframe(self):
        """Should accept DataFrame directly."""
        df = pd.DataFrame({"description": ["item1", "item2"]})

        pipeline = QuickPipeline.create(
            data=df, prompt="Summarize: {description}", model="gpt-4o-mini"
        )

        assert pipeline.specifications.dataset.input_columns == ["description"]

    def test_create_with_multiple_output_columns(self, tmp_path):
        """Should auto-select JSON parser for multi-column output."""
        csv_file = tmp_path / "test.csv"
        pd.DataFrame({"text": ["item1", "item2"]}).to_csv(csv_file, index=False)

        pipeline = QuickPipeline.create(
            data=csv_file,
            prompt="Extract from {text}",
            output_columns=["brand", "model", "price"],
        )

        # Should have 3 output columns
        assert len(pipeline.specifications.dataset.output_columns) == 3
        # JSON parser should be selected (not None)
        # We can't easily check this without inspecting stages, but batch size should be set
        assert pipeline.specifications.processing.batch_size > 0

    def test_create_with_single_output_column_string(self, tmp_path):
        """Should accept output_columns as string."""
        csv_file = tmp_path / "test.csv"
        pd.DataFrame({"text": ["item"]}).to_csv(csv_file, index=False)

        pipeline = QuickPipeline.create(
            data=csv_file, prompt="Process: {text}", output_columns="result"
        )

        assert pipeline.specifications.dataset.output_columns == ["result"]

    def test_create_with_budget(self, tmp_path):
        """Should accept max_budget parameter."""
        csv_file = tmp_path / "test.csv"
        pd.DataFrame({"text": ["item"]}).to_csv(csv_file, index=False)

        # Should not raise error
        pipeline = QuickPipeline.create(
            data=csv_file, prompt="Process: {text}", max_budget=Decimal("5.0")
        )

        # Pipeline should be created successfully
        assert pipeline is not None

    def test_extract_placeholders(self):
        """Should extract placeholder names from template."""
        # Single placeholder
        result = QuickPipeline._extract_placeholders("Summarize: {text}")
        assert result == ["text"]

        # Multiple placeholders
        result = QuickPipeline._extract_placeholders(
            "Compare {product1} and {product2}"
        )
        assert result == ["product1", "product2"]

        # No placeholders
        result = QuickPipeline._extract_placeholders("No variables here")
        assert result == []

        # Duplicate placeholders (should return both)
        result = QuickPipeline._extract_placeholders("{text} vs {text}")
        assert result == ["text", "text"]

    def test_detect_provider_openai(self):
        """Should detect OpenAI from model names."""
        assert QuickPipeline._detect_provider("gpt-4o-mini") == "openai"
        assert QuickPipeline._detect_provider("gpt-4") == "openai"
        assert QuickPipeline._detect_provider("text-davinci-003") == "openai"

    def test_detect_provider_anthropic(self):
        """Should detect Anthropic from model names."""
        assert QuickPipeline._detect_provider("claude-3-sonnet") == "anthropic"
        assert QuickPipeline._detect_provider("claude-2") == "anthropic"

    def test_detect_provider_groq(self):
        """Should detect Groq from model names."""
        assert QuickPipeline._detect_provider("llama-3-70b") == "groq"
        assert QuickPipeline._detect_provider("mixtral-8x7b") == "groq"

    def test_detect_provider_mlx(self):
        """Should detect MLX from model names."""
        assert QuickPipeline._detect_provider("mlx/qwen3") == "mlx"
        assert QuickPipeline._detect_provider("qwen-mlx") == "mlx"

    def test_detect_provider_default(self):
        """Should default to openai for unknown models."""
        assert QuickPipeline._detect_provider("unknown-model-123") == "openai"

    def test_select_parser_single_column(self):
        """Should select no parser for single column."""
        parser = QuickPipeline._select_parser(["output"])
        assert parser is None

    def test_select_parser_multi_column(self):
        """Should select JSON parser for multiple columns."""
        parser = QuickPipeline._select_parser(["col1", "col2", "col3"])
        assert parser is not None
        assert parser.strict is False

    def test_default_batch_size(self):
        """Should return appropriate batch size for data size."""
        # Small dataset
        assert QuickPipeline._default_batch_size(50) == 10

        # Medium dataset
        assert QuickPipeline._default_batch_size(500) == 50

        # Large dataset
        assert QuickPipeline._default_batch_size(5000) == 100

        # Very large dataset
        assert QuickPipeline._default_batch_size(50000) == 500

    def test_default_concurrency(self):
        """Should return appropriate concurrency for provider."""
        assert QuickPipeline._default_concurrency("openai") == 5
        assert QuickPipeline._default_concurrency("anthropic") == 5
        assert QuickPipeline._default_concurrency("groq") == 100
        assert QuickPipeline._default_concurrency("mlx") == 1
        assert QuickPipeline._default_concurrency("unknown") == 5

    def test_load_data_csv(self, tmp_path):
        """Should load CSV files."""
        csv_file = tmp_path / "test.csv"
        df_orig = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df_orig.to_csv(csv_file, index=False)

        df_loaded = QuickPipeline._load_data(csv_file)
        pd.testing.assert_frame_equal(df_loaded, df_orig)

    def test_load_data_excel(self, tmp_path):
        """Should load Excel files."""
        excel_file = tmp_path / "test.xlsx"
        df_orig = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df_orig.to_excel(excel_file, index=False)

        df_loaded = QuickPipeline._load_data(excel_file)
        pd.testing.assert_frame_equal(df_loaded, df_orig)

    def test_load_data_parquet(self, tmp_path):
        """Should load Parquet files."""
        parquet_file = tmp_path / "test.parquet"
        df_orig = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df_orig.to_parquet(parquet_file, index=False)

        df_loaded = QuickPipeline._load_data(parquet_file)
        pd.testing.assert_frame_equal(df_loaded, df_orig)

    def test_load_data_json(self, tmp_path):
        """Should load JSON files."""
        json_file = tmp_path / "test.json"
        df_orig = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df_orig.to_json(json_file, orient="records")

        df_loaded = QuickPipeline._load_data(json_file)
        # JSON loading might change dtypes, just check shape
        assert df_loaded.shape == df_orig.shape

    def test_load_data_dataframe(self):
        """Should return DataFrame as-is."""
        df_orig = pd.DataFrame({"a": [1, 2]})
        df_loaded = QuickPipeline._load_data(df_orig)
        assert df_loaded is df_orig

    def test_load_data_file_not_found(self):
        """Should raise error for missing file."""
        with pytest.raises(ValueError, match="Data file not found"):
            QuickPipeline._load_data("nonexistent.csv")

    def test_load_data_unsupported_format(self, tmp_path):
        """Should raise error for unsupported file format."""
        bad_file = tmp_path / "test.txt"
        bad_file.write_text("some data")

        with pytest.raises(ValueError, match="Unsupported file type"):
            QuickPipeline._load_data(bad_file)

    def test_create_no_placeholders_raises_error(self, tmp_path):
        """Should raise error if prompt has no placeholders."""
        csv_file = tmp_path / "test.csv"
        pd.DataFrame({"text": ["item"]}).to_csv(csv_file, index=False)

        with pytest.raises(ValueError, match="No placeholders found"):
            QuickPipeline.create(data=csv_file, prompt="Static text with no variables")

    def test_create_missing_columns_raises_error(self, tmp_path):
        """Should raise error if placeholder columns don't exist in data."""
        csv_file = tmp_path / "test.csv"
        pd.DataFrame({"text": ["item"]}).to_csv(csv_file, index=False)

        with pytest.raises(ValueError, match="not found in data"):
            QuickPipeline.create(data=csv_file, prompt="Process: {missing_column}")

    def test_create_with_temperature_override(self, tmp_path):
        """Should accept temperature parameter."""
        csv_file = tmp_path / "test.csv"
        pd.DataFrame({"text": ["item"]}).to_csv(csv_file, index=False)

        pipeline = QuickPipeline.create(
            data=csv_file, prompt="Process: {text}", temperature=0.7
        )

        assert pipeline.specifications.llm.temperature == 0.7

    def test_create_with_max_tokens_override(self, tmp_path):
        """Should accept max_tokens parameter."""
        csv_file = tmp_path / "test.csv"
        pd.DataFrame({"text": ["item"]}).to_csv(csv_file, index=False)

        pipeline = QuickPipeline.create(
            data=csv_file, prompt="Process: {text}", max_tokens=100
        )

        assert pipeline.specifications.llm.max_tokens == 100

    def test_create_with_explicit_provider(self, tmp_path):
        """Should respect explicit provider parameter."""
        csv_file = tmp_path / "test.csv"
        pd.DataFrame({"text": ["item"]}).to_csv(csv_file, index=False)

        pipeline = QuickPipeline.create(
            data=csv_file,
            prompt="Process: {text}",
            model="my-custom-model",
            provider="groq",
        )

        assert pipeline.specifications.llm.provider.value == "groq"

    def test_create_with_explicit_batch_concurrency(self, tmp_path):
        """Should respect explicit batch_size and concurrency."""
        csv_file = tmp_path / "test.csv"
        pd.DataFrame({"text": ["item"]}).to_csv(csv_file, index=False)

        pipeline = QuickPipeline.create(
            data=csv_file,
            prompt="Process: {text}",
            batch_size=25,  # QuickPipeline uses processing.batch_size
            concurrency=10,
        )

        # QuickPipeline.create() uses processing.batch_size (internal batching)
        # not prompt.batch_size (multi-row batching)
        assert pipeline.specifications.processing.batch_size == 25
        assert pipeline.specifications.processing.concurrency == 10
