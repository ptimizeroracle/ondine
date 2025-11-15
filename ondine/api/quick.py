"""
Simplified API with smart defaults for quick pipeline creation.

This module provides a high-level API that reduces boilerplate and makes
common use cases trivial while still providing access to full functionality.

Design Philosophy:
- Convention over configuration
- Auto-detect from context when possible
- Provide sensible defaults for 80% of use cases
- Still allow full customization when needed
"""

import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from ondine.api.pipeline import Pipeline
from ondine.api.pipeline_builder import PipelineBuilder
from ondine.stages.parser_factory import JSONParser


class QuickPipeline:
    """
    Simplified pipeline API with smart defaults.

    Designed for rapid prototyping and common use cases. Automatically detects:
    - Input columns from prompt template placeholders
    - Provider from model name (e.g., gpt-4 → openai, claude → anthropic)
    - Parser type (JSON for multi-column, text for single column)
    - Reasonable defaults for batch size, concurrency, retries

    Examples:
        Minimal usage:
        >>> pipeline = QuickPipeline.create(
        ...     data="data.csv",
        ...     prompt="Categorize this text: {text}"
        ... )
        >>> result = pipeline.execute()

        With explicit outputs:
        >>> pipeline = QuickPipeline.create(
        ...     data="products.csv",
        ...     prompt="Extract: {description}",
        ...     output_columns=["brand", "model", "price"]
        ... )

        Override defaults:
        >>> pipeline = QuickPipeline.create(
        ...     data=df,
        ...     prompt="Summarize: {content}",
        ...     model="gpt-4o",
        ...     temperature=0.7,
        ...     max_budget=Decimal("5.0")
        ... )
    """

    # Model name patterns for provider auto-detection
    PROVIDER_PATTERNS = {
        "openai": [r"^gpt-", r"^text-", r"^davinci", r"^curie", r"^babbage"],
        "anthropic": [r"^claude-"],
        "groq": [r"^llama", r"^mixtral", r"^gemma"],
        "azure_openai": [r"^azure/"],
        "mlx": [r"^mlx/", r"-mlx$"],
    }

    @staticmethod
    def create(
        data: str | Path | pd.DataFrame,
        prompt: str,
        model: str = "gpt-4o-mini",
        output_columns: list[str] | str | None = None,
        provider: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        max_budget: Decimal | float | str | None = None,
        batch_size: int | None = None,
        concurrency: int | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        """
        Create a pipeline with smart defaults.

        The simplest way to process data with LLMs. Automatically detects input columns
        from prompt placeholders and configures optimal settings based on data size.

        Args:
            data: CSV/Excel/Parquet file path or DataFrame
            prompt: Prompt template with {placeholders} matching column names
            model: Model name (default: gpt-4o-mini). Provider auto-detected from model name.
            output_columns: Output column name(s). If None, uses ["output"]
            provider: LLM provider. If None, auto-detected (gpt-4 → openai, claude → anthropic)
            temperature: Sampling temperature (0.0-1.0, default: 0.0 for deterministic)
            max_tokens: Max output tokens (optional, uses provider default)
            max_budget: Maximum cost budget in USD (optional, no limit if not set)
            batch_size: Rows per batch (optional, auto-sized: 10-500 based on data size)
            concurrency: Parallel requests (optional, auto-sized: 5-100 based on provider)
            **kwargs: Additional arguments passed to PipelineBuilder

        Returns:
            Configured Pipeline ready to execute

        Raises:
            ValueError: If input data cannot be loaded or prompt is invalid

        Example:
            ```python
            from ondine import QuickPipeline

            # Minimal - auto-detects everything
            pipeline = QuickPipeline.create(
                data="products.csv",
                prompt="Categorize: {description}"
            )
            result = pipeline.execute()

            # With budget control
            pipeline = QuickPipeline.create(
                data="reviews.csv",
                prompt="Sentiment: {review_text}",
                model="gpt-4o-mini",
                max_budget=5.0
            )

            # Multi-column output
            pipeline = QuickPipeline.create(
                data="products.csv",
                prompt="Extract from {title}: brand, price, category as JSON",
                output_columns=["brand", "price", "category"]
            )

            # Custom provider
            pipeline = QuickPipeline.create(
                data=df,
                prompt="Summarize: {text}",
                model="llama-3.3-70b-versatile",
                provider="groq"
            )
            ```

        Note:
            Input columns are automatically detected from {placeholders} in the prompt.
            Provider is auto-detected from model name (gpt-4 → openai, claude → anthropic, llama → groq).
        """
        # 1. Load data
        df = QuickPipeline._load_data(data)

        # 2. Auto-detect input columns from prompt template
        input_columns = QuickPipeline._extract_placeholders(prompt)
        if not input_columns:
            raise ValueError(
                f"No placeholders found in prompt: {prompt}\n"
                "Expected format: 'Your prompt with {{column_name}} placeholders'"
            )

        # Validate input columns exist in data
        missing = [col for col in input_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"Input columns {missing} not found in data. "
                f"Available columns: {list(df.columns)}"
            )

        # 3. Normalize output columns
        if output_columns is None:
            output_columns = ["output"]
        elif isinstance(output_columns, str):
            output_columns = [output_columns]

        # 4. Auto-detect provider from model name
        if provider is None:
            provider = QuickPipeline._detect_provider(model)

        # 5. Auto-select parser (JSON for multi-column, text for single)
        parser = QuickPipeline._select_parser(output_columns)

        # 6. Smart defaults for batch_size and concurrency
        if batch_size is None:
            batch_size = QuickPipeline._default_batch_size(len(df))
        if concurrency is None:
            concurrency = QuickPipeline._default_concurrency(provider)

        # 7. Build pipeline
        builder = (
            PipelineBuilder.create()
            .from_dataframe(
                df, input_columns=input_columns, output_columns=output_columns
            )
            .with_prompt(template=prompt)
            .with_llm(
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        )

        # Add optional parser if multi-column
        if parser:
            builder = builder.with_parser(parser)

        # Add batch/concurrency settings
        # Note: QuickPipeline uses processing_batch_size (internal batching)
        # not with_batch_size (multi-row batching)
        builder = builder.with_processing_batch_size(batch_size).with_concurrency(
            concurrency
        )

        # Add budget if specified
        if max_budget is not None:
            # Convert to float for PipelineBuilder (it expects float)
            if isinstance(max_budget, Decimal | str):
                max_budget = float(max_budget)
            builder = builder.with_max_budget(budget=max_budget)

        # Add sensible retry defaults
        builder = builder.with_max_retries(3)

        return builder.build()

    @staticmethod
    def _load_data(data: str | Path | pd.DataFrame) -> pd.DataFrame:
        """Load data from file or return DataFrame as-is."""
        if isinstance(data, pd.DataFrame):
            return data

        # Convert to Path for easier handling
        path = Path(data)
        if not path.exists():
            raise ValueError(f"Data file not found: {path}")

        # Load based on extension
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".json":
            return pd.read_json(path)
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            "Supported: .csv, .xlsx, .xls, .parquet, .json"
        )

    @staticmethod
    def _extract_placeholders(template: str) -> list[str]:
        """
        Extract placeholder names from prompt template.

        Examples:
            "Categorize: {text}" -> ["text"]
            "Compare {product1} vs {product2}" -> ["product1", "product2"]
        """
        return re.findall(r"\{(\w+)\}", template)

    @staticmethod
    def _detect_provider(model: str) -> str:
        """
        Auto-detect provider from model name.

        Examples:
            "gpt-4o-mini" -> "openai"
            "claude-3-sonnet" -> "anthropic"
            "llama-3-70b" -> "groq"
        """
        model_lower = model.lower()

        for provider, patterns in QuickPipeline.PROVIDER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, model_lower):
                    return provider

        # Default to openai if can't detect
        return "openai"

    @staticmethod
    def _select_parser(output_columns: list[str]) -> JSONParser | None:
        """
        Select parser based on output columns.

        Single column -> No parser needed (raw text)
        Multiple columns -> JSON parser
        """
        if len(output_columns) > 1:
            return JSONParser(strict=False)
        return None

    @staticmethod
    def _default_batch_size(data_size: int) -> int:
        """
        Smart batch size based on data size.

        Small datasets: larger batches (faster)
        Large datasets: smaller batches (checkpoint more often)
        """
        if data_size <= 100:
            return 10
        if data_size <= 1000:
            return 50
        if data_size <= 10000:
            return 100
        return 500

    @staticmethod
    def _default_concurrency(provider: str) -> int:
        """
        Smart concurrency based on provider typical rate limits.

        OpenAI: 60 RPM (tier 1) -> 5 concurrent
        Anthropic: 50 RPM -> 5 concurrent
        Groq: 6000 RPM -> 100 concurrent
        """
        defaults = {
            "openai": 5,
            "anthropic": 5,
            "azure_openai": 10,
            "groq": 100,
            "mlx": 1,  # Local, no rate limit but CPU-bound
        }
        return defaults.get(provider, 5)
