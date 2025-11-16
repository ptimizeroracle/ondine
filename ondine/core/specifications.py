"""
Core specification models for pipeline configuration.

These Pydantic models define the configuration contracts for all pipeline
components, following the principle of separation between configuration
(what to do) and execution (how to do it).
"""

from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DataSourceType(str, Enum):
    """Supported data source types."""

    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"
    DATAFRAME = "dataframe"


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OPENAI_COMPATIBLE = "openai_compatible"
    MLX = "mlx"


class ErrorPolicy(str, Enum):
    """Error handling policies for processing failures."""

    RETRY = "retry"
    SKIP = "skip"
    FAIL = "fail"
    USE_DEFAULT = "use_default"


class MergeStrategy(str, Enum):
    """Output merge strategies."""

    REPLACE = "replace"
    APPEND = "append"
    UPDATE = "update"


class DatasetSpec(BaseModel):
    """
    Specification for data source configuration.

    Defines how to load input data and which columns to process. Supports CSV,
    Excel, Parquet, and in-memory DataFrames.

    Attributes:
        source_type: Type of data source (csv, excel, parquet, dataframe)
        source_path: Path to file (None for DataFrame sources)
        input_columns: Column names to use in prompts (must exist in data)
        output_columns: Column names for LLM results (must not overlap with input)
        filters: Optional filters to apply when loading data
        sheet_name: Sheet name or index for Excel files (default: 0)
        delimiter: CSV delimiter character (default: ",")
        encoding: File encoding (default: "utf-8")

    Example:
        ```python
        spec = DatasetSpec(
            source_type=DataSourceType.CSV,
            source_path="products.csv",
            input_columns=["title", "description"],
            output_columns=["category", "price_range"]
        )
        ```
    """

    source_type: DataSourceType
    source_path: str | Path | None = None
    input_columns: list[str] = Field(
        ..., min_length=1, description="Columns to use as input"
    )
    output_columns: list[str] = Field(
        ..., min_length=1, description="Columns to store results"
    )
    filters: dict[str, Any] | None = Field(
        default=None, description="Optional data filters"
    )
    sheet_name: str | int | None = Field(
        default=0, description="Sheet name for Excel files"
    )
    delimiter: str = Field(default=",", description="CSV delimiter")
    encoding: str = Field(default="utf-8", description="File encoding")

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, v: str | Path | None) -> Path | None:
        """Convert string paths to Path objects."""
        if v is None:
            return None
        return Path(v) if isinstance(v, str) else v

    @field_validator("output_columns")
    @classmethod
    def validate_no_overlap(cls, v: list[str], info: Any) -> list[str]:
        """Ensure output columns don't overlap with input columns."""
        if "input_columns" in info.data:
            input_cols = set(info.data["input_columns"])
            output_cols = set(v)
            overlap = input_cols & output_cols
            if overlap:
                raise ValueError(f"Output columns overlap with input: {overlap}")
        return v


class PromptSpec(BaseModel):
    """Specification for prompt template configuration."""

    template: str = Field(..., min_length=1, description="Prompt template")
    system_message: str | None = Field(
        default=None, description="System message for LLM"
    )
    few_shot_examples: list[dict[str, str]] | None = Field(
        default=None, description="Few-shot learning examples"
    )
    template_variables: list[str] | None = Field(
        default=None, description="Expected template variables"
    )
    response_format: str = Field(
        default="raw", description="Response parsing format: 'raw', 'json', or 'regex'"
    )
    json_fields: list[str] | None = Field(
        default=None,
        description="Expected JSON field names (for response_format='json')",
    )
    regex_patterns: dict[str, str] | None = Field(
        default=None,
        description="Regex patterns for field extraction (for response_format='regex')",
    )
    batch_size: int = Field(
        default=1,
        description="Number of rows to process in a single API call (1 = no batching)",
        ge=1,
    )
    batch_strategy: str = Field(
        default="json",
        description="Batch formatting strategy: 'json' or 'csv'",
    )

    @field_validator("template")
    @classmethod
    def validate_template(cls, v: str) -> str:
        """Validate template has at least one variable."""
        if "{" not in v or "}" not in v:
            raise ValueError(
                "Template must contain at least one variable in {var} format"
            )
        return v

    @field_validator("response_format")
    @classmethod
    def validate_response_format(cls, v: str) -> str:
        """Validate response format is supported."""
        allowed = ["raw", "json", "regex"]
        if v not in allowed:
            raise ValueError(f"response_format must be one of {allowed}, got '{v}'")
        return v

    @field_validator("batch_strategy")
    @classmethod
    def validate_batch_strategy(cls, v: str) -> str:
        """Validate batch strategy is supported."""
        allowed = ["json", "csv"]
        if v not in allowed:
            raise ValueError(f"batch_strategy must be one of {allowed}, got '{v}'")
        return v


class LLMSpec(BaseModel):
    """
    Specification for LLM provider configuration.

    Defines which LLM provider to use, model settings, and authentication. Supports
    OpenAI, Azure OpenAI, Anthropic, Groq, MLX, and custom OpenAI-compatible APIs.

    Attributes:
        provider: LLM provider (openai, azure_openai, anthropic, groq, mlx, openai_compatible)
        model: Model identifier (e.g., "gpt-4o-mini", "claude-sonnet-4", "llama-3.3-70b-versatile")
        api_key: API key (optional, reads from environment if not provided)
        temperature: Sampling temperature (0.0-2.0, default: 0.0 for deterministic output)
        max_tokens: Maximum output tokens (optional, uses model default)
        top_p: Nucleus sampling parameter (0.0-1.0, default: 1.0)

    Example:
        ```python
        # OpenAI
        spec = LLMSpec(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            temperature=0.3
        )

        # Groq (fast and affordable)
        spec = LLMSpec(
            provider=LLMProvider.GROQ,
            model="llama-3.3-70b-versatile",
            temperature=0.0
        )

        # Azure with Managed Identity (no API key needed)
        spec = LLMSpec(
            provider=LLMProvider.AZURE_OPENAI,
            model="gpt-4",
            azure_endpoint="https://your-resource.openai.azure.com/",
            azure_deployment="gpt-4-deployment",
            use_managed_identity=True
        )
        ```

    Note:
        Use LLMProviderPresets for pre-configured common providers.
    """

    provider: LLMProvider
    model: str = Field(..., min_length=1, description="Model identifier")
    api_key: str | None = Field(default=None, description="API key (or from env)")
    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int | None = Field(default=None, gt=0, description="Max output tokens")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling")
    enable_prefix_caching: bool = Field(
        default=True,
        description="Enable prompt caching (OpenAI/Anthropic auto-cache system messages)",
    )

    # Azure-specific
    azure_endpoint: str | None = Field(
        default=None, description="Azure OpenAI endpoint"
    )
    azure_deployment: str | None = Field(
        default=None, description="Azure deployment name"
    )
    api_version: str | None = Field(
        default="2024-02-15-preview", description="Azure API version"
    )
    use_managed_identity: bool = Field(
        default=False,
        description="Use Azure Managed Identity for authentication (no API key needed)",
    )
    azure_ad_token: str | None = Field(
        default=None,
        description="Pre-fetched Azure AD token (alternative to use_managed_identity)",
    )

    # Custom/OpenAI-compatible provider fields
    base_url: str | None = Field(
        default=None,
        description="Base URL for OpenAI-compatible APIs (Ollama, vLLM, Together.ai, etc.)",
    )
    provider_name: str | None = Field(
        default=None,
        description="Custom provider name for logging/metrics",
    )

    # Cost tracking
    input_cost_per_1k_tokens: Decimal | None = Field(
        default=None, description="Input token cost"
    )
    output_cost_per_1k_tokens: Decimal | None = Field(
        default=None, description="Output token cost"
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url_format(cls, v: str | None) -> str | None:
        """Validate base_url is a valid HTTP(S) URL with a host."""
        if v is None:
            return v
        from urllib.parse import urlparse

        parsed = urlparse(v)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("base_url must start with http:// or https://")
        if not parsed.netloc:
            raise ValueError(
                "base_url must include a host (e.g., localhost, api.example.com)"
            )
        return v

    @field_validator("azure_endpoint", "azure_deployment")
    @classmethod
    def validate_azure_config(cls, v: str | None, info: Any) -> str | None:
        """Validate Azure-specific configuration."""
        if info.data.get("provider") == LLMProvider.AZURE_OPENAI and v is None:
            field_name = info.field_name
            raise ValueError(f"{field_name} required for Azure OpenAI provider")
        return v

    @model_validator(mode="after")
    def validate_provider_requirements(self) -> "LLMSpec":
        """Validate provider-specific requirements."""
        # Check openai_compatible requires base_url
        if self.provider == LLMProvider.OPENAI_COMPATIBLE and self.base_url is None:
            raise ValueError("base_url required for openai_compatible provider")
        return self


class ProcessingSpec(BaseModel):
    """
    Specification for processing parameters.

    Controls how the pipeline executes: batch sizes, concurrency, error handling,
    rate limiting, and budget constraints.

    Attributes:
        batch_size: Number of rows per batch (1-1000, default: 100)
        concurrency: Number of parallel LLM requests (1-20, default: 5)
        checkpoint_interval: Save checkpoint every N rows (default: 500)
        max_retries: Maximum retry attempts for failed requests (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 1.0)
        error_policy: How to handle errors (retry, skip, fail, use_default)
        rate_limit_rpm: Requests per minute limit (optional, no limit if None)
        max_budget: Maximum cost in USD (optional, no limit if None)

    Example:
        ```python
        # Conservative settings (free tier)
        spec = ProcessingSpec(
            batch_size=50,
            concurrency=5,
            rate_limit_rpm=25,
            max_budget=Decimal("5.0")
        )

        # Aggressive settings (paid tier)
        spec = ProcessingSpec(
            batch_size=200,
            concurrency=20,
            rate_limit_rpm=100,
            max_budget=Decimal("50.0")
        )

        # Fault-tolerant settings
        spec = ProcessingSpec(
            max_retries=5,
            error_policy=ErrorPolicy.RETRY,
            checkpoint_interval=100
        )
        ```
    """

    batch_size: int = Field(default=100, gt=0, le=1000, description="Rows per batch")
    concurrency: int = Field(default=5, gt=0, le=20, description="Parallel requests")
    checkpoint_interval: int = Field(
        default=500, gt=0, description="Checkpoint frequency"
    )
    max_retries: int = Field(default=3, ge=0, description="Max retry attempts")
    retry_delay: float = Field(
        default=1.0, ge=0.0, description="Initial retry delay (seconds)"
    )
    error_policy: ErrorPolicy = Field(
        default=ErrorPolicy.SKIP, description="Error handling policy"
    )
    rate_limit_rpm: int | None = Field(
        default=None, gt=0, description="Requests per minute limit"
    )
    max_budget: Decimal | None = Field(
        default=None, gt=0, description="Maximum budget in USD"
    )
    checkpoint_dir: Path = Field(
        default=Path(".checkpoints"), description="Checkpoint directory"
    )

    # Input preprocessing
    enable_preprocessing: bool = Field(
        default=False, description="Enable input preprocessing"
    )
    preprocessing_max_length: int = Field(
        default=500, gt=0, description="Max chars after preprocessing"
    )

    # Auto-retry failed rows
    auto_retry_failed: bool = Field(
        default=False, description="Auto-retry rows with null outputs"
    )
    max_retry_attempts: int = Field(
        default=1, ge=1, le=3, description="Max retry attempts for failed rows"
    )

    # Progress tracking
    progress_mode: str = Field(
        default="auto",
        description="Progress tracking mode: auto, rich, tqdm, logging, none",
    )

    @field_validator("checkpoint_dir")
    @classmethod
    def validate_checkpoint_dir(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v


class OutputSpec(BaseModel):
    """Specification for output configuration."""

    destination_type: DataSourceType
    destination_path: Path | None = None
    merge_strategy: MergeStrategy = Field(
        default=MergeStrategy.REPLACE, description="Output merge strategy"
    )
    atomic_write: bool = Field(default=True, description="Use atomic writes")

    @field_validator("destination_path")
    @classmethod
    def validate_destination_path(cls, v: str | Path | None) -> Path | None:
        """Convert string paths to Path objects."""
        if v is None:
            return None
        return Path(v) if isinstance(v, str) else v


class PipelineSpecifications(BaseModel):
    """Container for all pipeline specifications."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    dataset: DatasetSpec
    prompt: PromptSpec
    llm: LLMSpec
    processing: ProcessingSpec = Field(default_factory=ProcessingSpec)
    output: OutputSpec | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata"
    )


class LLMProviderPresets:
    """
    Pre-configured LLM provider specifications for common use cases.

    These presets provide convenient access to popular LLM providers with
    correct base URLs, pricing, and configuration. API keys must be provided
    at runtime via environment variables or explicit overrides.

    Example:
        # Use preset with env var API key
        from ondine.core.specifications import LLMProviderPresets

        pipeline = (
            PipelineBuilder.create()
            .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {text}")
            .with_llm_spec(LLMProviderPresets.TOGETHER_AI_LLAMA_70B)
            .build()
        )

        # Override API key
        spec = LLMProviderPresets.TOGETHER_AI_LLAMA_70B.model_copy(
            update={"api_key": "your-key"}  # pragma: allowlist secret
        )
        pipeline.with_llm_spec(spec)

    Security Note:
        All presets have api_key=None by default. You must provide API keys
        at runtime via environment variables or explicit overrides.
    """

    # OpenAI Presets
    GPT4O_MINI = LLMSpec(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        temperature=0.0,
        input_cost_per_1k_tokens=Decimal("0.00015"),
        output_cost_per_1k_tokens=Decimal("0.0006"),
    )

    GPT4O = LLMSpec(
        provider=LLMProvider.OPENAI,
        model="gpt-4o",
        temperature=0.0,
        input_cost_per_1k_tokens=Decimal("0.0025"),
        output_cost_per_1k_tokens=Decimal("0.01"),
    )

    # Together.AI Presets
    TOGETHER_AI_LLAMA_70B = LLMSpec(
        provider=LLMProvider.OPENAI_COMPATIBLE,
        provider_name="Together.AI",
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        base_url="https://api.together.xyz/v1",
        temperature=0.0,
        input_cost_per_1k_tokens=Decimal("0.0006"),
        output_cost_per_1k_tokens=Decimal("0.0006"),
    )

    TOGETHER_AI_LLAMA_8B = LLMSpec(
        provider=LLMProvider.OPENAI_COMPATIBLE,
        provider_name="Together.AI",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        base_url="https://api.together.xyz/v1",
        temperature=0.0,
        input_cost_per_1k_tokens=Decimal("0.0001"),
        output_cost_per_1k_tokens=Decimal("0.0001"),
    )

    # Ollama Local Presets
    OLLAMA_LLAMA_70B = LLMSpec(
        provider=LLMProvider.OPENAI_COMPATIBLE,
        provider_name="Ollama-Local",
        model="llama3.1:70b",
        base_url="http://localhost:11434/v1",
        temperature=0.0,
        input_cost_per_1k_tokens=Decimal("0.0"),
        output_cost_per_1k_tokens=Decimal("0.0"),
    )

    OLLAMA_LLAMA_8B = LLMSpec(
        provider=LLMProvider.OPENAI_COMPATIBLE,
        provider_name="Ollama-Local",
        model="llama3.1:8b",
        base_url="http://localhost:11434/v1",
        temperature=0.0,
        input_cost_per_1k_tokens=Decimal("0.0"),
        output_cost_per_1k_tokens=Decimal("0.0"),
    )

    # Groq Presets
    GROQ_LLAMA_70B = LLMSpec(
        provider=LLMProvider.GROQ,
        model="llama-3.1-70b-versatile",
        temperature=0.0,
        input_cost_per_1k_tokens=Decimal("0.00059"),
        output_cost_per_1k_tokens=Decimal("0.00079"),
    )

    # Anthropic Presets
    CLAUDE_SONNET_4 = LLMSpec(
        provider=LLMProvider.ANTHROPIC,
        model="claude-sonnet-4-20250514",
        temperature=0.0,
        max_tokens=8192,
        input_cost_per_1k_tokens=Decimal("0.003"),
        output_cost_per_1k_tokens=Decimal("0.015"),
    )

    @classmethod
    def create_custom_openai_compatible(
        cls,
        provider_name: str,
        model: str,
        base_url: str,
        input_cost_per_1k: float = 0.0,
        output_cost_per_1k: float = 0.0,
        **kwargs,
    ) -> LLMSpec:
        """
        Factory method for custom OpenAI-compatible providers.

        Use this for providers like vLLM, LocalAI, Anyscale, or any custom
        OpenAI-compatible API endpoint.

        Args:
            provider_name: Display name for the provider (for logging/metrics)
            model: Model identifier
            base_url: API endpoint URL (e.g., http://localhost:8000/v1)
            input_cost_per_1k: Input token cost per 1K tokens (default: 0.0)
            output_cost_per_1k: Output token cost per 1K tokens (default: 0.0)
            **kwargs: Additional LLMSpec parameters (temperature, max_tokens, etc.)

        Returns:
            Configured LLMSpec for the custom provider

        Example:
            spec = LLMProviderPresets.create_custom_openai_compatible(
                provider_name="My vLLM Server",
                model="mistral-7b-instruct",
                base_url="http://my-server:8000/v1",
                temperature=0.7
            )
        """
        return LLMSpec(
            provider=LLMProvider.OPENAI_COMPATIBLE,
            provider_name=provider_name,
            model=model,
            base_url=base_url,
            input_cost_per_1k_tokens=Decimal(str(input_cost_per_1k)),
            output_cost_per_1k_tokens=Decimal(str(output_cost_per_1k)),
            **kwargs,
        )
