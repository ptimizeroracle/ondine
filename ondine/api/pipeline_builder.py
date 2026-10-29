"""
Pipeline Builder - Fluent API for constructing pipelines.

Implements Builder pattern for intuitive pipeline creation.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from ondine.context.protocol import ContextStore

import pandas as pd  # noqa: TC002

from ondine.api.pipeline import Pipeline
from ondine.core.components import PipelineComponents
from ondine.core.specifications import (
    DatasetSpec,
    DataSourceType,
    LLMProvider,
    LLMSpec,
    MergeStrategy,
    OutputSpec,
    PipelineSpecifications,
    ProcessingSpec,
    PromptSpec,
)
from ondine.harness.contract import (
    ContractError,
    OutputContract,
    OutputField,
    Relation,
    RunPolicy,
    build_contract,
)
from ondine.orchestration import (
    AsyncExecutor,
    ExecutionStrategy,
    StreamingExecutor,
)


def _row_model(schema: Any) -> Any:
    """The model one row is checked against.

    A batch wrapper, a model whose only field is ``items: list[Row]`` (the
    shape ``with_batch_size(>1)`` asks for), gives ``Row``: each item is one
    row, and ``items`` is no output column. Any other schema is the row model
    itself.
    """
    fields = getattr(schema, "model_fields", None)
    if not isinstance(fields, dict) or list(fields) != ["items"]:
        return schema
    annotation = fields["items"].annotation
    element = get_args(annotation)[0] if get_origin(annotation) is list else None
    return element if hasattr(element, "model_fields") else schema


class PipelineBuilder:
    """
    Fluent builder for constructing pipelines.

    Provides an intuitive, chainable API for common use cases.

    Example:
        pipeline = (
            PipelineBuilder.create()
            .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {text}")
            .with_llm(provider="openai", model="gpt-4o-mini")
            .build()
        )
    """

    def __init__(self):
        """Initialize builder with None values."""
        self._dataset_spec: DatasetSpec | None = None
        self._prompt_spec: PromptSpec | None = None
        self._llm_spec: LLMSpec | None = None
        self._processing_spec: ProcessingSpec = ProcessingSpec()
        self._output_spec: OutputSpec | None = None
        self._dataframe: pd.DataFrame | None = None
        self._executor: ExecutionStrategy | None = None
        self._custom_parser: Any | None = None
        self._custom_llm_client: Any | None = None
        # Live collaborators, kept off specifications.metadata so the specs
        # stay serializable — see ondine/core/components.py (#232).
        self._knowledge_store: Any | None = None
        self._context_store: Any | None = None
        self._structured_output_model: Any | None = None
        self._custom_stages: list[dict] = []  # For custom stage injection
        self._observers: list[tuple[str, dict]] = []  # For observability
        self._custom_metadata: dict[str, Any] = {}  # For arbitrary metadata
        # Streaming configuration
        self._streaming_enabled: bool = False
        self._streaming_chunk_size: int = 10000
        self._streaming_max_pending: int = 3
        # Output contract declarations. They are compiled and validated together
        # in build(), so every problem is reported at once (ondine.harness).
        self._contract_fields: list[OutputField] = []
        self._contract_relations: list[Relation] = []
        self._contract_policy: RunPolicy | None = None
        self._schema_as_contract: bool = False

    @staticmethod
    def create() -> PipelineBuilder:
        """
        Start builder chain.

        Returns:
            New PipelineBuilder instance
        """
        return PipelineBuilder()

    @staticmethod
    def from_specifications(
        specs: PipelineSpecifications,
        dataframe: pd.DataFrame | None = None,
    ) -> PipelineBuilder:
        """
        Create builder from existing specifications.

        Useful for loading from YAML and modifying programmatically.

        Args:
            specs: Complete pipeline specifications
            dataframe: Optional in-memory DataFrame to re-attach. Specifications
                describe *how* to run a pipeline, not the data itself, so a
                caller rebuilding a pipeline that was constructed from an
                in-memory frame must pass it back explicitly or the rebuilt
                pipeline will have no data to load.

        Returns:
            PipelineBuilder pre-configured with specs

        Example:
            specs = load_pipeline_config("config.yaml")
            builder = PipelineBuilder.from_specifications(specs)
            pipeline = builder.build()
        """
        builder = PipelineBuilder()
        builder._dataset_spec = specs.dataset
        builder._prompt_spec = specs.prompt
        builder._llm_spec = specs.llm
        builder._processing_spec = specs.processing
        builder._output_spec = specs.output
        builder._dataframe = dataframe
        return builder

    def from_csv(
        self,
        path: str,
        input_columns: list[str],
        output_columns: list[str],
        delimiter: str = ",",
        encoding: str = "utf-8",
    ) -> PipelineBuilder:
        """
        Configure CSV data source.

        Args:
            path: Path to CSV file
            input_columns: Input column names to use in prompts
            output_columns: Output column names to generate
            delimiter: CSV delimiter (default: comma)
            encoding: File encoding (default: utf-8)

        Returns:
            Self for chaining

        Example:
            ```python
            builder = (
                PipelineBuilder.create()
                .from_csv(
                    "products.csv",
                    input_columns=["description"],
                    output_columns=["category"]
                )
            )
            ```
        """
        self._dataset_spec = DatasetSpec(
            source_type=DataSourceType.CSV,
            source_path=Path(path),
            input_columns=input_columns,
            output_columns=output_columns,
            delimiter=delimiter,
            encoding=encoding,
        )
        return self

    def from_excel(
        self,
        path: str,
        input_columns: list[str],
        output_columns: list[str],
        sheet_name: str | int = 0,
    ) -> PipelineBuilder:
        """
        Configure Excel data source.

        Args:
            path: Path to Excel file
            input_columns: Input column names
            output_columns: Output column names
            sheet_name: Sheet name or index

        Returns:
            Self for chaining
        """
        self._dataset_spec = DatasetSpec(
            source_type=DataSourceType.EXCEL,
            source_path=Path(path),
            input_columns=input_columns,
            output_columns=output_columns,
            sheet_name=sheet_name,
        )
        return self

    def from_parquet(
        self,
        path: str,
        input_columns: list[str],
        output_columns: list[str],
    ) -> PipelineBuilder:
        """
        Configure Parquet data source.

        Args:
            path: Path to Parquet file
            input_columns: Input column names
            output_columns: Output column names

        Returns:
            Self for chaining
        """
        self._dataset_spec = DatasetSpec(
            source_type=DataSourceType.PARQUET,
            source_path=Path(path),
            input_columns=input_columns,
            output_columns=output_columns,
        )
        return self

    def from_dataframe(
        self,
        df: pd.DataFrame,
        input_columns: list[str],
        output_columns: list[str],
    ) -> PipelineBuilder:
        """
        Configure DataFrame source.

        Useful for processing in-memory data or chaining pipelines.

        Args:
            df: Pandas DataFrame with input data
            input_columns: Column names to use in prompts
            output_columns: Column names to generate

        Returns:
            Self for chaining

        Example:
            ```python
            import pandas as pd

            df = pd.DataFrame({"text": ["Hello", "World"]})
            builder = (
                PipelineBuilder.create()
                .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            )
            ```
        """
        self._dataset_spec = DatasetSpec(
            source_type=DataSourceType.DATAFRAME,
            input_columns=input_columns,
            output_columns=output_columns,
        )
        self._dataframe = df
        return self

    def with_prompt(
        self,
        template: str,
        system_message: str | None = None,
    ) -> PipelineBuilder:
        """
        Configure prompt template.

        Use {variable} syntax to reference input columns. The LLM will process
        each row using this template.

        Args:
            template: Prompt template with {variable} placeholders matching input_columns
            system_message: Optional system message for chat models

        Returns:
            Self for chaining

        Example:
            ```python
            builder.with_prompt(
                "Categorize this product: {title}\\n\\nCategory:",
                system_message="You are a product categorization expert."
            )
            ```
        """
        self._prompt_spec = PromptSpec(
            template=template,
            system_message=system_message,
        )
        return self

    def with_system_prompt(self, system_prompt: str) -> PipelineBuilder:
        """
        Set system prompt for caching optimization.

        System prompts are cached by providers (OpenAI, Anthropic) and reused
        across all rows, reducing costs by 50-90% for the cached portion.

        This method should be called after with_prompt() to set or update the
        system message separately from the user prompt template.

        Args:
            system_prompt: Static instructions/context (will be cached)

        Returns:
            Self for chaining

        Example:
            ```python
            pipeline = (
                PipelineBuilder.create()
                .from_csv("data.csv", input_columns=["text"])
                .with_prompt("Review: {text}")  # Dynamic per row
                .with_system_prompt('''
                    You are a sentiment classifier.
                    Classify reviews as: positive, negative, or neutral.
                    Return only the label, nothing else.
                ''')  # Cached across all rows
                .with_llm(provider="openai", model="gpt-4o-mini")
                .build()
            )
            ```

        Note:
            Can also be set via with_prompt(template, system_message=...).
            This method provides a more explicit API for caching optimization.
        """
        if not self._prompt_spec:
            raise ValueError("Call with_prompt() before with_system_prompt()")

        self._prompt_spec.system_message = system_prompt
        return self

    def with_batch_size(self, batch_size: int) -> PipelineBuilder:
        """
        Set batch size for multi-row processing.

        Process multiple rows in a single API call to reduce costs and latency
        by up to 100×. Batch size of 1 (default) processes rows individually.

        Args:
            batch_size: Number of rows to process per API call (1-500)

        Returns:
            Self for chaining

        Raises:
            ValueError: If batch_size < 1 or prompt not set

        Example:
            ```python
            # Process 100 rows per API call (100× fewer calls)
            pipeline = (
                PipelineBuilder.create()
                .from_csv("data.csv", input_columns=["text"])
                .with_prompt("Classify: {text}")
                .with_batch_size(100)  # 5M rows = 50K API calls!
                .with_llm(provider="openai", model="gpt-4o-mini")
                .build()
            )
            ```

        Note:
            - Batch size is limited by model context window
            - Larger batches = fewer API calls but higher risk of partial failures
            - Recommended: Start with 10-50, increase based on results
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        if not self._prompt_spec:
            raise ValueError("Call with_prompt() before with_batch_size()")

        # Update batch_size directly (consistent with with_system_prompt)
        self._prompt_spec.batch_size = batch_size
        return self

    def with_batch_strategy(self, strategy: str) -> PipelineBuilder:
        """
        Set batch formatting strategy.

        Choose how multiple rows are formatted in a single prompt.

        Args:
            strategy: Strategy name ("json" or "csv")

        Returns:
            Self for chaining

        Raises:
            ValueError: If strategy is not supported or prompt not set

        Example:
            ```python
            # Use JSON array format (default, most reliable)
            pipeline.with_batch_size(100).with_batch_strategy("json")

            # Use CSV format (more compact, experimental)
            pipeline.with_batch_size(100).with_batch_strategy("csv")
            ```

        Supported Strategies:
            - "json": JSON array format (default, most reliable)
            - "csv": CSV format (more compact, experimental)
        """
        allowed = ["json", "csv"]
        if strategy not in allowed:
            raise ValueError(
                f"batch_strategy must be one of {allowed}, got '{strategy}'"
            )

        if not self._prompt_spec:
            raise ValueError("Call with_prompt() before with_batch_strategy()")

        # Update batch_strategy directly (consistent with with_system_prompt)
        self._prompt_spec.batch_strategy = strategy
        return self

    def with_jinja2(self, enabled: bool | None = True) -> PipelineBuilder:
        """
        Control Jinja2 template rendering mode.

        Jinja2 provides powerful features for dynamic prompts:
        - Conditionals: {% if condition %}...{% endif %}
        - Loops: {% for item in items %}...{% endfor %}
        - Filters: {{ text | upper | truncate(100) }}

        Args:
            enabled: Template rendering mode
                - True: Force Jinja2 rendering (for advanced features)
                - False: Force Python .format() (for compatibility)
                - None: Auto-detect based on syntax (default behavior)

        Returns:
            Self for chaining

        Note:
            By default (if not called), Ondine auto-detects the template syntax:
            - {{ variable }} → Jinja2 (auto-enabled)
            - {variable} → Python .format() (auto-enabled)

            Only call this method if you need to override auto-detection.

        Example:
            ```python
            # Auto-detection (recommended - no .with_jinja2() call needed)
            template = 'Product: {{ name }}'  # Jinja2 auto-detected

            # Explicit control (advanced use cases)
            template = '''Extract from: {{ description }}
            {% if category == "beverage" %}
            Focus on volume units (oz, ml, L)
            {% endif %}'''

            pipeline = (
                PipelineBuilder.create()
                .from_csv("products.csv", input_columns=["description", "category"])
                .with_prompt(template)
                .with_jinja2(True)  # Force Jinja2 (optional, auto-detected anyway)
                .with_llm("openai", "gpt-4o-mini")
                .build()
            )
            ```
        """
        self._processing_spec.use_jinja2 = enabled
        return self

    def with_llm(
        self,
        model: str,
        provider: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        input_cost_per_1k_tokens: Decimal | None = None,
        output_cost_per_1k_tokens: Decimal | None = None,
        **kwargs: Any,
    ) -> PipelineBuilder:
        """
        Configure LLM provider.

        Supports OpenAI, Azure OpenAI, Anthropic, Groq, MLX, and custom providers.
        API keys can be provided explicitly or via environment variables.

        Args:
            model: Model identifier. Can be:
                  - LiteLLM format: "provider/model" (e.g., "groq/openai/gpt-oss-120b")
                  - Short format: "model" (requires provider parameter)
            provider: Provider name (optional if model includes provider prefix)
                     Examples: "openai", "groq", "anthropic", or custom provider ID
            api_key: API key (optional, reads from environment if not provided)
            temperature: Sampling temperature (0.0-1.0, default: 0.0 for deterministic)
            max_tokens: Maximum output tokens (optional, uses model default)
            input_cost_per_1k_tokens: Input cost per 1K tokens (optional, auto-detected from LiteLLM)
            output_cost_per_1k_tokens: Output cost per 1K tokens (optional, auto-detected from LiteLLM)
            **kwargs: Provider-specific parameters (e.g., azure_endpoint, azure_deployment)

        Returns:
            Self for chaining

        Note:
            **Automatic Cost Tracking**: If you don't provide cost parameters, Ondine automatically
            detects pricing from LiteLLM's database (1800+ models). Only specify costs manually
            for custom/unknown models.

        Example:
            ```python
            # NEW: LiteLLM format (provider auto-detected from model string)
            builder.with_llm(model="groq/openai/gpt-oss-120b")
            builder.with_llm(model="openai/gpt-4o-mini")

            # OLD: Explicit provider (still supported)
            builder.with_llm(provider="openai", model="gpt-4o-mini")
            builder.with_llm(provider="groq", model="openai/gpt-oss-120b")

            # Any OpenAI-compatible endpoint (OpenRouter, Together, vLLM,
            # LM Studio, ...). The model name is passed through as-is — no
            # "openai/" prefix needed, the provider already says the endpoint
            # speaks OpenAI.
            builder.with_llm(
                provider="openai_compatible",
                model="inclusionai/ling-2.6-flash",
                base_url="https://openrouter.ai/api/v1",
            )

            # Manual cost override (custom models LiteLLM has no pricing for)
            builder.with_llm(
                model="my-custom-model",
                provider="openai_compatible",
                base_url="https://my-api.com/v1",
                input_cost_per_1k_tokens=Decimal("0.001"),
                output_cost_per_1k_tokens=Decimal("0.002")
            )
            ```
        """
        from ondine.adapters.provider_registry import ProviderRegistry

        # Auto-detect provider from model string if not explicitly provided
        if provider is None:
            if "/" in model:
                # LiteLLM format: "provider/model"
                provider_enum = LLMProvider.LITELLM
            else:
                raise ValueError(
                    "Provider parameter required when model doesn't include provider prefix.\n"
                    "Either use LiteLLM format (e.g., 'groq/openai/gpt-oss-120b') or specify provider explicitly."
                )
        else:
            # Try to convert to enum for built-in providers
            try:
                provider_enum = LLMProvider(provider.lower())
            except ValueError:
                # Not a built-in provider - check if it's a custom provider
                if ProviderRegistry.is_registered(provider):
                    # Use a dummy enum value for validation, but store the actual provider string
                    provider_enum = LLMProvider.OPENAI  # Dummy for Pydantic validation
                    kwargs["custom_provider_id"] = provider
                else:
                    raise ValueError(
                        f"Unknown provider: {provider}. "
                        f"Available providers: {', '.join(ProviderRegistry.list_providers())}"
                    )

        self._llm_spec = LLMSpec(
            provider=provider_enum,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            input_cost_per_1k_tokens=input_cost_per_1k_tokens,
            output_cost_per_1k_tokens=output_cost_per_1k_tokens,
            **kwargs,
        )
        return self

    def with_llm_spec(self, spec: LLMSpec) -> PipelineBuilder:
        """
        Configure LLM using a pre-built LLMSpec object.

        This method allows using LLMSpec objects directly, enabling:
        - Reusable provider configurations
        - Use of LLMProviderPresets for common providers
        - Custom LLMSpec instances for advanced use cases

        Args:
            spec: LLM specification object

        Returns:
            Self for chaining

        Raises:
            TypeError: If spec is not an LLMSpec instance

        Example:
            # Use preset
            from ondine.core.specifications import LLMProviderPresets

            pipeline = (
                PipelineBuilder.create()
                .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
                .with_prompt("Process: {text}")
                .with_llm_spec(LLMProviderPresets.TOGETHER_AI_LLAMA_70B)
                .build()
            )

            # Custom spec
            custom = LLMSpec(
                provider=LLMProvider.OPENAI,
                model="gpt-4o-mini",
                temperature=0.7
            )
            pipeline.with_llm_spec(custom)

            # Override preset
            spec = LLMProviderPresets.GPT4O_MINI.model_copy(
                update={"temperature": 0.9}
            )
            pipeline.with_llm_spec(spec)
        """
        if not isinstance(spec, LLMSpec):
            raise TypeError(
                f"Expected LLMSpec, got {type(spec).__name__}. "
                f"Use with_llm() for parameter-based configuration."
            )

        self._llm_spec = spec
        return self

    def with_custom_llm_client(self, client: Any) -> PipelineBuilder:
        """
        Provide a custom LLM client instance directly.

        This allows advanced users to create their own LLM client implementations
        by extending the LLMClient base class. The custom client will be used
        instead of the factory-created client.

        Args:
            client: Custom LLM client instance (must inherit from LLMClient)

        Returns:
            Self for chaining

        Example:
            class MyCustomClient(LLMClient):
                def invoke(self, prompt: str, **kwargs) -> LLMResponse:
                    # Custom implementation
                    ...

            pipeline = (
                PipelineBuilder.create()
                .from_dataframe(df, ...)
                .with_prompt("...")
                .with_custom_llm_client(MyCustomClient(spec))
                .build()
            )
        """
        from ondine.adapters.llm_client import LLMClient

        if not isinstance(client, LLMClient):
            raise TypeError(
                f"Custom client must inherit from LLMClient, got {type(client).__name__}"
            )

        self._custom_llm_client = client
        return self

    def with_concurrency(self, threads: int) -> PipelineBuilder:
        """
        Configure concurrent requests.

        Higher concurrency = faster processing but more API load. Adjust based on
        your provider's rate limits.

        Args:
            threads: Number of concurrent threads (1-100, typical: 5-20)

        Returns:
            Self for chaining

        Example:
            ```python
            # Conservative (free tier)
            builder.with_concurrency(5)

            # Aggressive (paid tier)
            builder.with_concurrency(20)

            # Maximum (enterprise)
            builder.with_concurrency(50)
            ```

        Note:
            Groq supports high concurrency (~100), while OpenAI free tier is limited to ~5.
        """
        self._processing_spec.concurrency = threads
        return self

    def with_execution_mode(
        self, mode: Literal["live", "provider_batch"]
    ) -> PipelineBuilder:
        """Select the pipeline's execution mode (§3, §5).

        * ``"live"`` (default) — one HTTP call per prompt through the
          asyncio engine. Byte-identical to every prior version.
        * ``"provider_batch"`` — compile prompts to JSONL, submit a
          provider-native Batch API job (OpenAI/Anthropic), and collect
          results later via :meth:`Pipeline.submit` /
          :meth:`Pipeline.attach`. Non-blocking; designed for very large
          or offline workloads where 50% batch discounts matter.

        The scope guard in :meth:`build` rejects ``provider_batch`` for
        providers without Batch API support (v1: OpenAI + Anthropic
        only) so misconfiguration is caught at build time, not mid-run.

        Args:
            mode: ``"live"`` or ``"provider_batch"``.

        Returns:
            Self for chaining.
        """
        self._processing_spec.execution_mode = mode
        return self

    def with_adaptive_concurrency(self, enabled: bool = True) -> PipelineBuilder:
        """Opt-in adaptive in-flight cap (Gradient2 algorithm).

        When enabled, the effective concurrency shrinks on 429 / RTT
        inflation and grows on saturation with near-baseline RTT. The
        ceiling is ``with_concurrency(...)``; the floor is 1. Use this
        when the provider's rate limits are opaque or variable and
        you'd rather let Ondine probe for the sweet spot than pick
        a fixed number.

        Default is off — behaviour is byte-identical to prior versions.

        Args:
            enabled: ``True`` to turn on, ``False`` to turn off. Passing
                ``False`` is useful to undo an earlier call on the same
                builder.

        Returns:
            Self for chaining.

        Example:
            ```python
            # Let Ondine find the right concurrency under a flaky
            # third-party API — upper bound still 20.
            builder.with_concurrency(20).with_adaptive_concurrency()
            ```
        """
        self._processing_spec.adaptive_concurrency = enabled
        return self

    def with_checkpoint_interval(self, rows: int) -> PipelineBuilder:
        """
        Configure checkpoint frequency.

        Args:
            rows: Rows between checkpoints

        Returns:
            Self for chaining
        """
        self._processing_spec.checkpoint_interval = rows
        return self

    def with_checkpoint_cleanup(self, enabled: bool = True) -> PipelineBuilder:
        """
        Control whether checkpoints are deleted after successful execution.

        By default checkpoints are cleaned up on success. Set ``enabled=False``
        to keep them as a safety net — if downstream code (e.g. ``to_csv()``)
        crashes after the pipeline returns, you can resume from the checkpoint
        at zero LLM cost.

        Args:
            enabled: True (default) to delete checkpoints on success,
                     False to retain them.

        Returns:
            Self for chaining
        """
        self._processing_spec.cleanup_on_success = enabled
        return self

    def with_rate_limit(
        self,
        rpm: int,
        *,
        redis_url: str | None = None,
        scope: str | None = None,
    ) -> PipelineBuilder:
        """
        Configure rate limiting.

        Prevents hitting API rate limits by throttling requests using token bucket algorithm.
        Set this below your provider's actual limit for safety.

        Args:
            rpm: Requests per minute (typical: 20-60 for free tiers, 100+ for paid)
            redis_url: Optional ``redis://host:port/db`` URL. When set, workers
                sharing the same Redis + ``scope`` share one token bucket —
                the canonical solution when multiple Ondine jobs target the
                same provider API key. Falls back to an in-process limiter
                at the same ``rpm`` if Redis is unreachable.
            scope: Bucket scope for the distributed limiter. Workers that
                must share a budget use the same scope (e.g.
                ``"openai:gpt-4o"`` or ``"anthropic:sonnet:tier2"``).
                Ignored when ``redis_url`` is not set.

        Returns:
            Self for chaining

        Example:
            ```python
            # Single-process: in-memory token bucket
            builder.with_rate_limit(50)

            # Multi-process: one shared bucket across every worker
            # pointed at the same Redis + scope
            builder.with_rate_limit(
                100,
                redis_url="redis://localhost:6379/0",
                scope="openai:gpt-4o-mini",
            )
            ```

        Note:
            Without ``redis_url``, rate limiting is per-process. Each
            pipeline instance gets its own token bucket, so N parallel
            jobs effectively allow N×rpm against your provider.
        """
        self._processing_spec.rate_limit_rpm = rpm
        if redis_url is not None:
            self._processing_spec.rate_limit_redis_url = redis_url
        if scope is not None:
            self._processing_spec.rate_limit_scope = scope
        return self

    def with_max_retries(self, retries: int) -> PipelineBuilder:
        """
        Configure maximum retry attempts.

        Args:
            retries: Maximum number of retry attempts

        Returns:
            Self for chaining
        """
        self._processing_spec.max_retries = retries
        return self

    def with_max_budget(self, budget: float) -> PipelineBuilder:
        """
        Configure maximum budget.

        Pipeline will halt execution if cost exceeds this limit. Warnings are shown
        at 75% and 90% of budget.

        Args:
            budget: Maximum budget in USD (e.g., 5.0 for $5)

        Returns:
            Self for chaining

        Example:
            ```python
            # Set $5 budget limit
            builder.with_max_budget(5.0)

            # For testing with small budget
            builder.with_max_budget(0.50)

            # For production runs
            builder.with_max_budget(100.0)
            ```

        Note:
            Budget enforcement uses Decimal precision to avoid floating-point errors.
        """
        self._processing_spec.max_budget = Decimal(str(budget))
        return self

    def with_error_policy(self, policy: str) -> PipelineBuilder:
        """
        Configure error handling policy.

        Args:
            policy: Error policy ('skip', 'fail', 'retry', 'use_default')

        Returns:
            Self for chaining
        """
        from ondine.core.specifications import ErrorPolicy

        self._processing_spec.error_policy = ErrorPolicy(policy.lower())
        return self

    def with_checkpoint_dir(self, directory: str) -> PipelineBuilder:
        """
        Configure checkpoint directory.

        Args:
            directory: Path to checkpoint directory

        Returns:
            Self for chaining
        """
        self._processing_spec.checkpoint_dir = Path(directory)
        return self

    def with_parser(self, parser: Any) -> PipelineBuilder:
        """
        Configure response parser.

        This method allows setting a custom parser. The parser type
        determines the response_format in the prompt spec.

        Args:
            parser: Parser instance (JSONParser, RegexParser, PydanticParser, etc.)

        Returns:
            Self for chaining
        """
        # Store the parser for later use in the pipeline
        # We'll configure response_format based on parser type
        if hasattr(parser, "__class__"):
            parser_name = parser.__class__.__name__
            if "JSON" in parser_name:
                if not self._prompt_spec:
                    raise ValueError(
                        "with_prompt() must be called before with_parser()"
                    )
                # Update the existing prompt spec's response_format
                self._prompt_spec.response_format = "json"
            elif "Regex" in parser_name:
                if not self._prompt_spec:
                    raise ValueError(
                        "with_prompt() must be called before with_parser()"
                    )
                self._prompt_spec.response_format = "regex"
                if hasattr(parser, "patterns"):
                    self._prompt_spec.regex_patterns = parser.patterns

        # Store the parser instance in metadata for the pipeline to use
        if not hasattr(self, "_custom_parser"):
            self._custom_parser = parser

        return self

    def to_csv(self, path: str) -> PipelineBuilder:
        """
        Configure CSV output destination.

        Alias for with_output(path, format='csv').

        Args:
            path: Output CSV file path

        Returns:
            Self for chaining
        """
        return self.with_output(path, format="csv")

    def with_output(
        self,
        path: str,
        format: str = "csv",
        merge_strategy: str = "replace",
    ) -> PipelineBuilder:
        """
        Configure output destination.

        Args:
            path: Output file path
            format: Output format (csv, excel, parquet)
            merge_strategy: Merge strategy (replace, append, update)

        Returns:
            Self for chaining
        """
        format_map = {
            "csv": DataSourceType.CSV,
            "excel": DataSourceType.EXCEL,
            "parquet": DataSourceType.PARQUET,
        }

        merge_map = {
            "replace": MergeStrategy.REPLACE,
            "append": MergeStrategy.APPEND,
            "update": MergeStrategy.UPDATE,
        }

        self._output_spec = OutputSpec(
            destination_type=format_map[format.lower()],
            destination_path=Path(path),
            merge_strategy=merge_map[merge_strategy.lower()],
        )
        return self

    def with_executor(self, executor: ExecutionStrategy) -> PipelineBuilder:
        """
        Set custom execution strategy.

        Args:
            executor: ExecutionStrategy instance

        Returns:
            Self for chaining
        """
        self._executor = executor
        return self

    def with_async_execution(self, max_concurrency: int = 10) -> PipelineBuilder:
        """
        Use async execution strategy.

        Enables async/await for non-blocking execution.
        Ideal for FastAPI, aiohttp, and async frameworks.

        Args:
            max_concurrency: Maximum concurrent async tasks

        Returns:
            Self for chaining
        """
        self._executor = AsyncExecutor(max_concurrency=max_concurrency)
        return self

    def with_streaming(
        self,
        chunk_size: int = 10000,
        max_pending_chunks: int = 3,
    ) -> PipelineBuilder:
        """
        Enable streaming execution for large datasets.

        Processes data in chunks with bounded memory usage.
        Ideal for datasets that don't fit in memory (100K+ rows).

        Uses:
        - Polars LazyFrame for memory-efficient loading
        - Bounded async queue for backpressure
        - Incremental result writing

        Args:
            chunk_size: Number of rows per chunk (default: 10000)
            max_pending_chunks: Max chunks in processing queue (default: 3)

        Returns:
            Self for chaining

        Example:
            pipeline = (
                PipelineBuilder.create()
                .from_csv("large_file.csv", ...)
                .with_streaming(chunk_size=5000)
                .build()
            )

            # Use streaming execution
            async for result in pipeline.execute_stream_async():
                print(f"Processed {result.metrics.processed_rows} rows")
        """
        self._streaming_enabled = True
        self._streaming_chunk_size = chunk_size
        self._streaming_max_pending = max_pending_chunks
        # Also set executor for backward compatibility with execute_stream()
        self._executor = StreamingExecutor(chunk_size=chunk_size)
        return self

    def with_progress_mode(self, mode: str = "auto") -> PipelineBuilder:
        """
        Configure progress tracking mode.

        Args:
            mode: Progress tracking mode
                - "auto": Auto-detect (rich if TTY, else logging) [default]
                - "rich": Split-panel progress bars with ETA
                - "textual": Full TUI with interactive scrollable logs
                - "logging": Simple log messages
                - "none": Disable progress tracking

        Returns:
            Self for chaining

        Example:
            ```python
            # Use rich progress (beautiful UI)
            builder.with_progress_mode("rich")

            # Disable progress (faster, cleaner logs)
            builder.with_progress_mode("none")

            # Auto-detect (recommended)
            builder.with_progress_mode("auto")
            ```
        """
        self._processing_spec.progress_mode = mode
        return self

    def with_stage(
        self,
        stage_name: str,
        position: str = "before_prompt",
        **stage_kwargs,
    ) -> PipelineBuilder:
        """
        Add a custom pipeline stage by name.

        Enables injection of custom processing stages at specific points
        in the pipeline. Stages must be registered via StageRegistry.

        Args:
            stage_name: Registered stage name (e.g., "rag_retrieval")
            position: Where to inject the stage. Options:
                - "after_loader" / "before_prompt": After data loading, before prompt formatting
                - "after_prompt" / "before_llm": After prompt formatting, before LLM invocation
                - "after_llm" / "before_parser": After LLM invocation, before parsing
                - "after_parser": After response parsing
            **stage_kwargs: Arguments to pass to stage constructor

        Returns:
            Self for chaining

        Raises:
            ValueError: If stage_name not registered or position invalid

        Example:
            # RAG retrieval example
            pipeline = (
                PipelineBuilder.create()
                .from_csv("questions.csv", input_columns=["question"], output_columns=["answer"])
                .with_stage(
                    "rag_retrieval",
                    position="before_prompt",
                    vector_store="pinecone",
                    index_name="my-docs",
                    top_k=5
                )
                .with_prompt("Context: {retrieved_context}\\n\\nQuestion: {question}\\n\\nAnswer:")
                .with_llm(provider="openai", model="gpt-4o")
                .build()
            )

            # Content moderation example
            pipeline = (
                PipelineBuilder.create()
                .from_csv("content.csv", input_columns=["text"], output_columns=["moderated"])
                .with_stage(
                    "content_moderation",
                    position="before_llm",
                    block_patterns=["spam", "offensive"]
                )
                .with_prompt("Moderate: {text}")
                .with_llm(provider="openai", model="gpt-4o-mini")
                .build()
            )
        """
        from ondine.stages.stage_registry import StageRegistry

        # Validate position
        valid_positions = [
            "after_loader",
            "before_prompt",
            "after_prompt",
            "before_llm",
            "after_llm",
            "before_parser",
            "after_parser",
        ]
        if position not in valid_positions:
            raise ValueError(
                f"Invalid position '{position}'. Must be one of: {', '.join(valid_positions)}"
            )

        # Get stage class from registry (this will raise ValueError if not found)
        stage_class = StageRegistry.get(stage_name)

        # Store stage config for later instantiation
        self._custom_stages.append(
            {
                "name": stage_name,
                "class": stage_class,
                "position": position,
                "kwargs": stage_kwargs,
            }
        )

        return self

    def with_observer(
        self, name: str, config: dict[str, Any] | None = None
    ) -> PipelineBuilder:
        """
        Add observability observer to the pipeline.

        Observers receive events during pipeline execution for monitoring,
        logging, and tracing.

        Args:
            name: Observer identifier (e.g., "langfuse", "opentelemetry", "logging")
            config: Observer-specific configuration dictionary


        Returns:
            Self for chaining

        Raises:
            ValueError: If observer not registered

        Example:
            # OpenTelemetry for infrastructure monitoring
            pipeline = (
                PipelineBuilder.create()
                .from_csv("data.csv", ...)
                .with_prompt("...")
                .with_llm(provider="openai", model="gpt-4o-mini")
                .with_observer("opentelemetry", config={
                    "tracer_name": "my_pipeline",
                    "include_prompts": False
                })
                .build()
            )

            # Langfuse for LLM-specific observability
            pipeline = (
                PipelineBuilder.create()
                .from_csv("data.csv", ...)
                .with_prompt("...")
                .with_llm(provider="openai", model="gpt-4o-mini")
                .with_observer("langfuse", config={
                    "public_key": "pk-lf-...",
                    "secret_key": "sk-lf-..."  # pragma: allowlist secret
                })
                .build()
            )

            # Multiple observers
            pipeline = (
                PipelineBuilder.create()
                .from_csv("data.csv", ...)
                .with_prompt("...")
                .with_llm(provider="openai", model="gpt-4o-mini")
                .with_observer("langfuse", config={...})
                .with_observer("opentelemetry", config={...})
                .with_observer("logging", config={"log_level": "DEBUG"})
                .build()
            )
        """
        from ondine.observability.registry import ObserverRegistry

        # Validate observer is registered
        if not ObserverRegistry.is_registered(name):
            available = ", ".join(ObserverRegistry.list_observers())
            raise ValueError(
                f"Observer '{name}' not registered. "
                f"Available observers: {available or 'none'}"
            )

        # Store observer config for later instantiation
        self._observers.append((name, config or {}))

        return self

    def with_router(
        self,
        model_list: list[dict],
        routing_strategy: str = "simple-shuffle",
        timeout: int = 120,
        num_retries: int = 2,  # Default 2: retry failed requests with other providers
        redis_url: str | None = None,
        # Circuit breaker / resilience params (sensible defaults)
        allowed_fails: int = 3,  # Failures before provider cooldown
        cooldown_time: int = 60,  # Seconds to disable failed provider
        **router_kwargs,
    ) -> PipelineBuilder:
        """
        Enable LiteLLM Router for load balancing, failover, and resilience.

        Router provides:
        - Load balancing across multiple deployments
        - Automatic failover with health tracking
        - Built-in circuit breaker (cooldown on failures)
        - Built-in retries with exponential backoff
        - Redis for distributed state (optional)

        Resilience is ENABLED BY DEFAULT with sensible settings:
        - 3 failures → provider goes into cooldown
        - 60 second cooldown period
        - Automatic failover to healthy providers

        All LiteLLM Router parameters are supported via **router_kwargs!
        Just pass any parameter directly - it flows through to LiteLLM.

        Args:
            model_list: List of deployment configs (required)
            routing_strategy: "simple-shuffle", "weighted-pick", "latency-based-routing", etc.
            timeout: Request timeout in seconds (default: 120)
            num_retries: Retry attempts with other providers on failure (default: 2)
            redis_url: Redis URL for distributed state (optional)
            allowed_fails: Failures before cooldown (default: 3). Set to 0 to disable.
            cooldown_time: Cooldown duration in seconds (default: 60). Set to 0 to disable.
            **router_kwargs: ANY LiteLLM Router parameter! Examples:
                - fallbacks=["backup-model"]: Fallback model names
                - enable_pre_call_checks=False: Skip health checks (faster!)
                - set_verbose=True: Enable debug logging
                - debug=True: Enable provider tracking
                Full docs: https://docs.litellm.ai/docs/routing

        Returns:
            Self for chaining

        Example:
            ```python
            # Basic usage (resilience enabled by default)
            .with_router(
                model_list=[
                    {"model_name": "fast", "litellm_params": {"model": "groq/openai/gpt-oss-120b", "rpm": 30}},
                    {"model_name": "fast", "litellm_params": {"model": "openai/gpt-4o-mini", "rpm": 500}}
                ]
            )

            # Custom resilience settings
            .with_router(
                model_list=[...],
                allowed_fails=5,     # More lenient
                cooldown_time=120,   # Longer cooldown
            )

            # Disable circuit breaker (not recommended)
            .with_router(
                model_list=[...],
                allowed_fails=0,     # Never cooldown
            )
            ```

        See: https://docs.litellm.ai/docs/routing
        """
        if not self._llm_spec:
            self._llm_spec = LLMSpec(provider=LLMProvider.OPENAI, model="router")

        # Build router config with explicit params + flexible kwargs
        # Build config - explicit params + all kwargs flow through to LiteLLM!
        router_config = {
            "model_list": model_list,
            "routing_strategy": routing_strategy,
            "timeout": timeout,
            "num_retries": num_retries,
            # Circuit breaker / resilience params
            "allowed_fails": allowed_fails,
            "cooldown_time": cooldown_time,
            **router_kwargs,  # ALL other params pass directly to LiteLLM Router!
        }

        # Add Redis if provided
        if redis_url:
            router_config["redis_url"] = redis_url

        self._llm_spec.router_config = router_config
        return self

    def with_redis_cache(
        self, redis_url: str = "redis://localhost:6379", ttl: int = 3600
    ) -> PipelineBuilder:
        """
        Enable LiteLLM Redis caching.

        Caches responses automatically (identical requests = $0 cost!).

        Args:
            redis_url: Redis URL (default: localhost)
            ttl: Cache TTL in seconds (default: 1h)

        Returns:
            Self for chaining

        See: https://docs.litellm.ai/docs/caching
        """
        if not self._llm_spec:
            self._llm_spec = LLMSpec(provider=LLMProvider.OPENAI, model="cached")

        self._llm_spec.cache_config = {
            "type": "redis",
            "host": redis_url,
            "ttl": ttl,
        }
        return self

    def with_disk_cache(self, cache_dir: str = ".cache") -> PipelineBuilder:
        """
        Enable LiteLLM Disk caching.

        Caches responses to the local filesystem.

        Args:
            cache_dir: Directory to store cache files (default: .cache)

        Returns:
            Self for chaining
        """
        if not self._llm_spec:
            self._llm_spec = LLMSpec(provider=LLMProvider.OPENAI, model="cached")

        self._llm_spec.cache_config = {
            "type": "disk",
            "disk_cache_dir": cache_dir,
        }
        return self

    def with_structured_output(
        self,
        schema: Any,
        mode: str = "auto",
        *,
        as_contract: bool = False,
    ) -> PipelineBuilder:
        """
        Configure structured output using a Pydantic model.

        This enables native schema enforcement, parsing, and auto-retry logic
        using LlamaIndex's structured_predict capabilities.

        Automatically configures JSONParser to handle the structured JSON output,
        unless a custom parser was already configured.

        **Batch Processing Note:**
        When using with_batch_size() > 1, your Pydantic model should have an
        'items' field containing a list. Ondine adds a row id to each item and
        realigns the response by it, so a model that reorders or drops items is
        corrected or reported rather than silently misassigning answers (#255).
        You do not add the id yourself. Example:

            class MyBatch(BaseModel):
                items: list[MyResult]  # one per row; realigned by row id

        Args:
            schema: Pydantic model class defining the expected output structure
            mode: Instructor mode for structured output:
                - "auto" (default): Intelligent detection based on model capabilities
                  Uses LiteLLM's model info, provider registry, and safe fallbacks
                - "tools": Function calling mode (most reliable, not all models support)
                - "json": JSON mode (universal compatibility, works with all models)
                - "json_schema": OpenAI's native JSON schema mode
            as_contract: Also declare the schema as the output contract. Off
                by default: a schema shapes the answer but rejects nothing
                unless you opt in. Not yet enforced: the contract is validated
                at ``build()``, but running a pipeline with one raises
                NotImplementedError until delivery is wired (harness step 3).
                Once enforced, its types, ``Literal``/``Enum`` values, bounds,
                patterns, lengths and validators will be checks on every
                answer, and a validator's own message will be the reason a row
                fails. For a batch wrapper (a model whose only field is
                ``items: list[Row]``, as ``with_batch_size(>1)`` uses) the
                contract is read from ``Row``, since each item is one row.
                Combine with :meth:`with_output_field` for output columns the
                schema does not declare, and set the run policy with
                :meth:`with_contract_policy`.

        Returns:
            Self for chaining

        Example:
            ```python
            # Auto-detect best mode (recommended)
            .with_structured_output(MySchema)

            # Force JSON mode (for models that don't support function calling)
            .with_structured_output(MySchema, mode="json")

            # Force TOOLS mode (for OpenAI/Anthropic)
            .with_structured_output(MySchema, mode="tools")

            # Declare the schema's rules as the contract (not yet enforced)
            .with_structured_output(MySchema, as_contract=True)
            ```

        Note:
            Each call replaces the whole structured-output configuration,
            ``as_contract`` included.
        """
        if not hasattr(self, "_custom_metadata"):
            self._custom_metadata = {}
        self._structured_output_model = schema
        self._schema_as_contract = as_contract
        self._custom_metadata["instructor_mode"] = mode

        # Auto-inject JSONParser if no parser configured
        # Structured output always returns JSON, so we need a JSON parser
        if self._custom_parser is None:
            from ondine.stages.response_parser_stage import JSONParser

            self._custom_parser = JSONParser()

        return self

    # ── Output contract ────────────────────────────────────────────
    #
    # What a deliverable value is: per-column rules, field-to-field rules, the
    # opt-in schema switch (with_structured_output(..., as_contract=True)) and
    # the run policy. Nothing is checked here; build() compiles every
    # declaration into one OutputContract and reports all problems at once.
    # Not yet enforced: running a pipeline with a contract raises
    # NotImplementedError until delivery is wired (harness step 3).

    def with_output_field(
        self,
        name: str | OutputField,
        *,
        allowed: Sequence[Any] | None = None,
        aliases: Mapping[str, Any] | None = None,
        type: str | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        pattern: str | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        required: bool | None = None,
        absent: Any = None,
        normalise: Sequence[str] | None = None,
    ) -> PipelineBuilder:
        """
        Declare what a deliverable value of one output column is.

        Not yet enforced: the declaration is validated at ``build()``, but
        running a pipeline with a contract raises NotImplementedError until
        delivery is wired (harness step 3).

        Once enforced, every answer will be checked against the declaration.
        A row whose value breaks it will be rejected (blanked and quarantined)
        or, under ``with_contract_policy(on_failure="deliver_flagged")``,
        delivered flagged as unverified. Columns you do not declare are not
        checked.

        A keyword left out (or ``None``) declares no such rule.

        Args:
            name: The output column, or a prepared
                :class:`~ondine.harness.contract.OutputField` (then pass no
                other keyword).
            allowed: The only values the column may take.
            aliases: Other spellings, each mapped to one allowed value, which
                is what will be delivered once the contract is enforced.
            type: "text" (default), "integer", "number", "boolean" or
                "any". "any" declares no type rule: the value passes whatever
                it is, as an undeclared column does. It is the type a schema
                field Ondine cannot read (a nested model, a list) gets. The
                keyword is named ``type`` to match ``OutputField.type``.
            minimum: Inclusive lower bound, for integer and number columns.
            maximum: Inclusive upper bound, for integer and number columns.
            pattern: A regular expression a text value must contain a match
                for; anchor it with ``^...$`` to match the whole value.
            min_length: Fewest characters a text value may have.
            max_length: Most characters a text value may have.
            required: True (default): the column must hold a value. False: it
                may be absent (missing, null, empty, or equal to ``absent``).
            absent: For an optional column, the answer that means "no value"
                (for example "n/a"). Once enforced, it will be delivered empty.
            normalise: Cleanups applied before checking, from "strip",
                "collapse_whitespace", "strip_punctuation", "casefold". Once
                enforced, a text value will be delivered cleaned.

        Returns:
            Self for chaining

        Example:
            ```python
            # The common case: the labels a column may take
            builder.with_output_field("intent", allowed=["refund", "billing"])

            # Several rules on several columns
            (
                builder
                .with_output_field(
                    "intent",
                    allowed=["refund", "billing"],
                    aliases={"money back": "refund"},
                    normalise=["strip", "casefold"],
                )
                .with_output_field("amount", type="number", minimum=0)
                .with_output_field("sku", required=False, absent="n/a")
            )
            ```

        Raises:
            TypeError: An ``OutputField`` was passed together with keywords.

        Note:
            Calls for different columns add up. Declare all of a column's
            rules in one call: a second, different declaration for the same
            column fails ``build()`` (as does a column the enforced schema
            already declares), because silently merging or replacing would
            hide which one you meant. Repeating the same declaration is
            harmless. Problems in the rules themselves (a minimum above the
            maximum, an unknown column, an unsafe pattern) are reported by
            ``build()``, all at once.
        """
        rules = {
            key: value
            for key, value in {
                "allowed": allowed,
                "aliases": aliases,
                "type": type,
                "minimum": minimum,
                "maximum": maximum,
                "pattern": pattern,
                "min_length": min_length,
                "max_length": max_length,
                "required": required,
                "absent": absent,
                "normalise": normalise,
            }.items()
            if value is not None
        }
        if isinstance(name, OutputField):
            if rules:
                raise TypeError(
                    "with_output_field() takes either an OutputField or rule "
                    f"keywords, not both; got an OutputField and {', '.join(rules)}"
                )
            declaration = name
        else:
            declaration = OutputField(name, **rules)
        if declaration not in self._contract_fields:
            self._contract_fields.append(declaration)
        return self

    def with_output_relation(
        self,
        left: str,
        op: str,
        right: str | None = None,
        *,
        input_column: str | None = None,
    ) -> PipelineBuilder:
        """
        Declare a rule between two output columns, or an output and an input.

        Not yet enforced: the rule is validated at ``build()``, but running a
        pipeline with a contract raises NotImplementedError until delivery is
        wired (harness step 3).

        The rule reads ``left op right``, with ``op`` one of
        ``< <= == != >= >``. Once enforced, it will be checked only when both
        values passed their own column rules, so one bad value gives one
        reason, not two.

        Args:
            left: An output column.
            op: The comparison.
            right: Another output column. Give this or ``input_column``.
            input_column: An input column of the same row, instead of
                ``right``.

        Returns:
            Self for chaining

        Example:
            ```python
            # An end date never before its start date
            builder.with_output_relation("end", ">=", "start")

            # A refund never above the order total in the input
            builder.with_output_relation("refund", "<=", input_column="total")
            ```

        Note:
            Calls add up; repeating the same rule is harmless. Unknown
            columns and comparisons of incompatible types are reported by
            ``build()``.
        """
        relation = Relation(left, op, right, input_column)
        if relation not in self._contract_relations:
            self._contract_relations.append(relation)
        return self

    def with_contract_policy(
        self,
        on_failure: Literal["reject", "deliver_flagged"] = "reject",
        partial_rows: Literal["whole_row", "keep_valid_fields"] = "whole_row",
        early_stop_fraction: float | None = RunPolicy.early_stop_fraction,
        early_stop_window: int = RunPolicy.early_stop_window,
    ) -> PipelineBuilder:
        """
        Set how a run will treat rows that fail the output contract.

        Not yet enforced: the policy is validated at ``build()``, but running
        a pipeline with a contract raises NotImplementedError until delivery
        is wired (harness step 3). The options below describe what each will
        do once it is.

        Only meaningful with a contract: declare one with
        :meth:`with_output_field`, :meth:`with_output_relation` or
        ``with_structured_output(..., as_contract=True)``. Without this call
        the defaults below apply.

        Args:
            on_failure: "reject" (default) will blank a failing row and put it
                in the quarantine; "deliver_flagged" will deliver its value
                with the status unverified.
            partial_rows: "whole_row" (default) will fail the row when any
                required column fails; "keep_valid_fields" will deliver the
                columns that passed.
            early_stop_fraction: The run will stop when more than this share
                of the first ``early_stop_window`` rows (by input order) is
                rejected, which usually means a broken prompt or contract.
                Default 0.5. ``None`` turns early stop off.
            early_stop_window: How many rows the share is measured over.
                Default 50.

        Returns:
            Self for chaining

        Example:
            ```python
            # Keep every row, flag the ones that failed a check
            builder.with_contract_policy(on_failure="deliver_flagged")

            # Never stop early
            builder.with_contract_policy(early_stop_fraction=None)
            ```

        Note:
            Each call sets the whole policy, like the builder's other
            settings: an option left out of the last call takes its default.
            Invalid values are reported by ``build()``, together with every
            other contract problem. A policy with nothing declared to check
            fails ``build()``, since it could never apply.
        """
        self._contract_policy = RunPolicy(
            on_failure=on_failure,
            partial_rows=partial_rows,
            early_stop_fraction=early_stop_fraction,
            early_stop_window=early_stop_window,
        )
        return self

    def _compile_output_contract(
        self, dataset: DatasetSpec, prompt: PromptSpec
    ) -> OutputContract | None:
        """The declared output contract, validated; ``None`` if none was declared.

        Takes the final dataset and prompt specs, which give the output and
        input columns and the response format. Every problem, in the
        declarations or the policy, is raised in one ContractError.
        """
        schema = (
            _row_model(self._structured_output_model)
            if self._schema_as_contract
            else None
        )
        declared = self._contract_fields or self._contract_relations
        if not declared and schema is None:
            if self._contract_policy is not None:
                raise ContractError(
                    [
                        "with_contract_policy() was called but nothing is "
                        "declared to check: add with_output_field(), "
                        "with_output_relation() or "
                        "with_structured_output(..., as_contract=True)"
                    ]
                )
            return None
        return build_contract(
            dataset.output_columns,
            fields=self._contract_fields,
            relations=self._contract_relations,
            policy=self._contract_policy,
            schema=schema,
            input_columns=dataset.input_columns,
            response_format=prompt.response_format,
            regex_patterns=prompt.regex_patterns,
        )

    # ── Knowledge Base ─────────────────────────────────────────────

    def with_knowledge_base(
        self,
        store: Any = None,
        *,
        query_columns: list[str] | None = None,
        top_k: int = 3,
        rerank: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        query_transform: str | None = None,
        evaluate: bool = False,
        eval_model: str = "openai/gpt-4o-mini",
    ) -> PipelineBuilder:
        """Attach a knowledge base for retrieval-augmented generation.

        When a ``KnowledgeStore`` is provided, the pipeline inserts a
        retrieval stage between data loading and prompt formatting.
        Each row is augmented with a ``{_kb_context}`` variable that
        you can reference in your prompt template.

        Args:
            store: A pre-built ``KnowledgeStore`` (required).
            query_columns: Input columns to concatenate as the search query.
                If ``None``, uses all input columns from the dataset spec.
            top_k: Number of chunks to retrieve per row.
            rerank: Enable cross-encoder reranking of results.
            reranker_model: Cross-encoder model name (only used when
                ``rerank=True``).
            query_transform: Query expansion strategy. One of
                ``"multi-query"``, ``"hyde"``, ``"step-back"``, or
                ``None`` to disable.
            evaluate: When ``True``, run LLM-as-judge evaluation on
                RAG answers and add ``_kb_eval_*`` columns.
            eval_model: LLM model for the judge (only used when
                ``evaluate=True``).

        Returns:
            Self for chaining

        Example:
            ```python
            from ondine.knowledge import KnowledgeStore

            kb = KnowledgeStore("knowledge.db")
            kb.ingest("docs/")

            pipeline = (
                PipelineBuilder.create()
                .from_csv("questions.csv", input_columns=["question"], output_columns=["answer"])
                .with_knowledge_base(kb, top_k=5, rerank=True, query_transform="hyde")
                .with_prompt(
                    "Context:\\n{_kb_context}\\n\\nQuestion: {question}\\nAnswer:"
                )
                .with_llm(model="openai/gpt-4o-mini")
                .build()
            )
            ```
        """
        if store is None:
            raise ValueError(
                "A KnowledgeStore instance is required. "
                "Create one with: KnowledgeStore('path.db')"
            )

        self._knowledge_store = store
        self._custom_metadata["knowledge_config"] = {
            "query_columns": query_columns,
            "top_k": top_k,
            "rerank": rerank,
            "reranker_model": reranker_model,
            "query_transform": query_transform,
            "evaluate": evaluate,
            "eval_model": eval_model,
        }
        return self

    # ── Context Store / Anti-Hallucination ──────────────────────────

    def with_context_store(
        self,
        store: ContextStore | None = None,
    ) -> PipelineBuilder:
        """Configure the context store backend for evidence-based verification.

        When enabled, the pipeline stores each LLM answer that passes
        ``with_grounding`` as an "input → answer" evidence record. Answers no
        check passed are never stored, so evidence priming cannot copy an
        unchecked answer into a later prompt.

        Args:
            store: ContextStore instance, or None to auto-detect:
                   tries RustContextStore first, falls back to InMemoryContextStore.

        Returns:
            Self for chaining

        Example:
            ```python
            from ondine.context import RustContextStore

            pipeline = (
                PipelineBuilder.create()
                .from_csv("data.csv", ...)
                .with_context_store(RustContextStore("evidence.db"))
                .with_grounding(threshold=0.3)
                .build()
            )
            ```
        """
        if store is None:
            try:
                from ondine.context.rust_store import RustContextStore

                store = RustContextStore()
            except ImportError:
                from ondine.context.memory_store import InMemoryContextStore

                store = InMemoryContextStore()

        self._context_store = store
        return self

    def with_evidence_priming(
        self,
        query_columns: list[str] | None = None,
        *,
        top_k: int = 3,
        min_score: float = 0.1,
    ) -> PipelineBuilder:
        """Inject prior evidence into prompts before LLM inference.

        Enables pre-LLM evidence retrieval: for each row, searches the context
        store for earlier answers to similar inputs and appends them to the
        prompt, labelled as reference only. Results below ``min_score`` are
        discarded to avoid injecting noise.

        The store holds evidence you stored yourself, plus LLM answers that
        passed ``with_grounding`` in earlier runs. Grounding means the answer
        is similar to its source text, not that it is correct. Without
        grounding, pipeline runs add nothing, and priming finds only your own
        evidence.

        Requires a context store (calls ``with_context_store()`` automatically
        if not already set).

        Args:
            query_columns: Columns to use as the search query. Defaults to the
                pipeline's input columns.
            top_k: Maximum evidence records to retrieve per row.
            min_score: Minimum relevance score to include a result (0-1).

        Returns:
            Self for chaining

        Example:
            ```python
            pipeline = (
                PipelineBuilder.create()
                .from_csv("data.csv", input_columns=["product"], output_columns=["category"])
                .with_prompt("Classify: {product}")
                .with_context_store(RustContextStore("evidence.db"))
                .with_evidence_priming(query_columns=["product"], top_k=3, min_score=0.2)
                .build()
            )
            ```
        """
        if self._context_store is None:
            self.with_context_store()

        self._custom_metadata["evidence_priming"] = {
            "query_columns": query_columns,
            "top_k": top_k,
            "min_score": min_score,
        }
        return self

    def with_grounding(
        self,
        threshold: float = 0.3,
        action: str = "flag",
        embed_fn: Callable[..., Any] | None = None,
    ) -> PipelineBuilder:
        """Enable grounding verification on LLM responses.

        After the LLM produces a response, grounding compares it against
        source text via TF-IDF cosine similarity. When an embedding function
        is provided, dense cosine similarity is also computed and the final
        score is ``max(tfidf_score, embedding_score)``.

        Requires a context store to be configured (auto-enables if not set).

        Args:
            threshold: Minimum similarity score to accept (0.0-1.0).
            action: What to do with ungrounded responses:
                    "flag" (keep the answer, add a ``_grounding_score``
                    column; filter on it yourself),
                    "skip" (set the row's output cells to None; the row
                    stays in the result).
            embed_fn: Optional callable ``(list[str]) -> list[list[float]]``
                      that returns embedding vectors for the given texts.
                      When provided, embedding cosine similarity augments
                      the TF-IDF score (taking the max of both).

        Returns:
            Self for chaining
        """
        if action == "retry":
            # Documented in early releases but never implemented (#307): the
            # run kept every ungrounded answer and added no score column.
            raise ValueError(
                'with_grounding(action="retry") is not supported: Ondine does '
                'not re-prompt ungrounded rows. Use action="flag" and filter '
                'on the _grounding_score column, or action="skip" to blank '
                "ungrounded rows."
            )
        valid_actions = ("flag", "skip")
        if action not in valid_actions:
            raise ValueError(f"action must be one of {valid_actions}, got '{action}'")

        self._custom_metadata["grounding"] = {
            "threshold": threshold,
            "action": action,
            "embed_fn": embed_fn,
        }

        if self._context_store is None:
            self.with_context_store()

        return self

    def with_contradiction_detection(
        self,
        key_columns: list[str] | None = None,
        value_columns: list[str] | None = None,
        tolerance: int | float | None = None,
    ) -> PipelineBuilder:
        """Enable cross-row contradiction detection.

        After grounding, checks whether the LLM produced conflicting outputs
        for rows that share the same key columns but differ in value columns.

        Requires a context store to be configured (auto-enables if not set).

        Args:
            key_columns: Columns that identify the same entity (e.g., ["product_id"]).
                         If None, uses the input columns from the dataset spec.
            value_columns: Columns to compare for contradictions (e.g., ["category"]).
                           If None, uses the output columns from the dataset spec.
            tolerance: For numeric value columns, differences within this
                       tolerance are **not** flagged as contradictions.
                       ``None`` (default) uses exact string equality.
                       For example, ``tolerance=1`` on a 0-5 scale ignores
                       ±1 score differences.

        Returns:
            Self for chaining
        """
        self._custom_metadata["contradiction_detection"] = {
            "key_columns": key_columns,
            "value_columns": value_columns,
            "tolerance": tolerance,
        }

        if self._context_store is None:
            self.with_context_store()

        return self

    def with_confidence_scoring(
        self,
        include_in_output: bool = True,
        scoring_mode: str = "default",
    ) -> PipelineBuilder:
        """Enable confidence scoring on LLM responses.

        Adds a ``_confidence_score`` column computed from the grounding
        similarity (and, in ``"default"`` mode, the evidence support count).
        It is a heuristic, not a calibrated probability that the answer is
        correct. It needs ``with_grounding``: without a grounding score the
        column is empty.

        Args:
            include_in_output: Whether to add the ``_confidence_score`` column
                               to the output DataFrame.
            scoring_mode: Scoring formula to use. One of ``"default"``,
                          ``"sigmoid"``, or ``"grounding_only"``.

        Returns:
            Self for chaining
        """
        valid_modes = ("default", "sigmoid", "grounding_only")
        if scoring_mode not in valid_modes:
            raise ValueError(
                f"scoring_mode must be one of {valid_modes!r}, got {scoring_mode!r}"
            )
        self._custom_metadata["confidence_scoring"] = {
            "include_in_output": include_in_output,
            "scoring_mode": scoring_mode,
        }

        if self._context_store is None:
            self.with_context_store()

        return self

    # ── Build ─────────────────────────────────────────────────────

    def build(self) -> Pipeline:
        """
        Build final Pipeline.

        Validates all configurations and constructs the Pipeline object ready for execution.
        This is the final step in the builder chain.

        Returns:
            Configured Pipeline ready to execute

        Raises:
            ValueError: If required specifications missing (dataset, prompt, or LLM)
            ContractError: If the declared output contract has problems; it lists
                every one (see with_output_field, with_output_relation,
                with_contract_policy)

        Example:
            ```python
            pipeline = (
                PipelineBuilder.create()
                .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
                .with_prompt("Summarize: {text}")
                .with_llm(provider="openai", model="gpt-4o-mini")
                .with_concurrency(10)
                .with_max_budget(5.0)
                .build()  # Returns Pipeline object
            )

            # Execute the pipeline
            result = pipeline.execute()
            print(f"Processed {result.metrics.total_rows} rows")
            ```
        """
        # Validate required specs
        if not self._dataset_spec:
            raise ValueError("Dataset specification required")
        if not self._prompt_spec:
            raise ValueError("Prompt specification required")

        # LLM spec is optional if custom client is provided
        if not self._llm_spec and not self._custom_llm_client:
            raise ValueError("Either LLM specification or custom LLM client required")

        # Prepare metadata with custom parser, custom client, custom stages, and observers
        metadata = {}
        if self._custom_metadata:
            metadata.update(self._custom_metadata)
        if self._custom_stages:
            metadata["custom_stages"] = self._custom_stages
        if self._observers:
            metadata["observers"] = self._observers
        if self._streaming_enabled:
            metadata["streaming"] = {
                "enabled": True,
                "chunk_size": self._streaming_chunk_size,
                "max_pending_chunks": self._streaming_max_pending,
            }

        # Create specifications bundle
        # If custom client provided but no llm_spec, create a dummy spec
        llm_spec = self._llm_spec
        if llm_spec is None and self._custom_llm_client is not None:
            # Create minimal spec using custom client's attributes
            llm_spec = LLMSpec(
                provider=LLMProvider.OPENAI,  # Dummy provider
                model=self._custom_llm_client.model,
                temperature=self._custom_llm_client.temperature,
                max_tokens=self._custom_llm_client.max_tokens,
            )

        if llm_spec is None:
            raise ValueError(
                "LLM spec is required. Call with_llm() or with_custom_llm_client() before build()."
            )

        # Scope guard (§5): provider_batch mode is only supported for
        # providers with a Batch API backend (v1: OpenAI + Anthropic).
        # Catching it here — the single user-facing choke point — means a
        # misconfigured run fails immediately with a clear message rather
        # than crashing deep in the JSONL submit path later.
        if self._processing_spec.execution_mode == "provider_batch":
            from ondine.orchestration.backends.provider_batch import (
                SUPPORTED_BATCH_PROVIDERS,
                ProviderBatchBackend,
            )

            provider_key = ProviderBatchBackend._provider_key(llm_spec)
            if provider_key not in SUPPORTED_BATCH_PROVIDERS:
                provider_label = (
                    llm_spec.provider.value
                    if hasattr(llm_spec.provider, "value")
                    else llm_spec.provider
                )
                raise ValueError(
                    f"execution_mode='provider_batch' is only supported for "
                    f"{sorted(SUPPORTED_BATCH_PROVIDERS)} in v1, but the "
                    f"configured provider is '{provider_label}'. Either "
                    f"switch to an OpenAI/Anthropic provider or use "
                    f"execution_mode='live'."
                )

        # Last check before a pipeline exists, so a broken contract costs
        # nothing. Kept off the specs: a schema model does not serialize.
        output_contract = self._compile_output_contract(
            self._dataset_spec, self._prompt_spec
        )

        specifications = PipelineSpecifications(
            dataset=self._dataset_spec,
            prompt=self._prompt_spec,
            llm=llm_spec,
            processing=self._processing_spec,
            output=self._output_spec,
            metadata=metadata,
        )

        # Create and return pipeline
        return Pipeline(
            specifications,
            dataframe=self._dataframe,
            executor=self._executor,
            components=PipelineComponents(
                llm_client=self._custom_llm_client,
                knowledge_store=self._knowledge_store,
                context_store=self._context_store,
                structured_output_model=self._structured_output_model,
                custom_parser=self._custom_parser,
                output_contract=output_contract,
            ),
        )
