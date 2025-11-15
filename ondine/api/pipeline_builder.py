"""
Pipeline Builder - Fluent API for constructing pipelines.

Implements Builder pattern for intuitive pipeline creation.
"""

from decimal import Decimal
from pathlib import Path

import pandas as pd

from ondine.api.pipeline import Pipeline
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
from ondine.orchestration import (
    AsyncExecutor,
    ExecutionStrategy,
    StreamingExecutor,
)


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
        self._custom_parser: any | None = None
        self._custom_llm_client: any | None = None
        self._custom_stages: list[dict] = []  # For custom stage injection
        self._observers: list[tuple[str, dict]] = []  # For observability

    @staticmethod
    def create() -> "PipelineBuilder":
        """
        Start builder chain.

        Returns:
            New PipelineBuilder instance
        """
        return PipelineBuilder()

    @staticmethod
    def from_specifications(specs: PipelineSpecifications) -> "PipelineBuilder":
        """
        Create builder from existing specifications.

        Useful for loading from YAML and modifying programmatically.

        Args:
            specs: Complete pipeline specifications

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
        return builder

    def from_csv(
        self,
        path: str,
        input_columns: list[str],
        output_columns: list[str],
        delimiter: str = ",",
        encoding: str = "utf-8",
    ) -> "PipelineBuilder":
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
    ) -> "PipelineBuilder":
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
    ) -> "PipelineBuilder":
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
    ) -> "PipelineBuilder":
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
    ) -> "PipelineBuilder":
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

    def with_system_prompt(self, system_prompt: str) -> "PipelineBuilder":
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

    def with_batch_size(self, batch_size: int) -> "PipelineBuilder":
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

    def with_batch_strategy(self, strategy: str) -> "PipelineBuilder":
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

    def with_llm(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: any,
    ) -> "PipelineBuilder":
        """
        Configure LLM provider.

        Supports OpenAI, Azure OpenAI, Anthropic, Groq, MLX, and custom providers.
        API keys can be provided explicitly or via environment variables.

        Args:
            provider: Provider name (openai, azure_openai, anthropic, groq, mlx) or custom provider ID
            model: Model identifier (e.g., "gpt-4o-mini", "claude-sonnet-4")
            api_key: API key (optional, reads from environment if not provided)
            temperature: Sampling temperature (0.0-1.0, default: 0.0 for deterministic)
            max_tokens: Maximum output tokens (optional, uses model default)
            **kwargs: Provider-specific parameters (e.g., azure_endpoint, azure_deployment)

        Returns:
            Self for chaining

        Example:
            ```python
            # OpenAI
            builder.with_llm(provider="openai", model="gpt-4o-mini")

            # Groq (fast and affordable)
            builder.with_llm(provider="groq", model="llama-3.3-70b-versatile")

            # Azure OpenAI with Managed Identity
            builder.with_llm(
                provider="azure_openai",
                model="gpt-4",
                azure_endpoint="https://your-resource.openai.azure.com/",
                azure_deployment="gpt-4-deployment",
                use_managed_identity=True
            )
            ```
        """
        from ondine.adapters.provider_registry import ProviderRegistry

        # Try to convert to enum for built-in providers
        try:
            provider_enum = LLMProvider(provider.lower())
        except ValueError:
            # Not a built-in provider - check if it's a custom provider
            if ProviderRegistry.is_registered(provider):
                # Use a dummy enum value for validation, but store the actual provider string
                provider_enum = LLMProvider.OPENAI  # Dummy for Pydantic validation
                kwargs["_custom_provider_id"] = provider
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
            **kwargs,
        )
        return self

    def with_llm_spec(self, spec: LLMSpec) -> "PipelineBuilder":
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

    def with_custom_llm_client(self, client: any) -> "PipelineBuilder":
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

    def with_processing_batch_size(self, size: int) -> "PipelineBuilder":
        """
        Configure internal batch size for PromptFormatterStage.

        This is different from with_batch_size() which enables multi-row batching.
        This method controls how many prompts are grouped together internally
        for processing efficiency.

        Args:
            size: Rows per internal batch

        Returns:
            Self for chaining

        Note:
            This is an internal optimization parameter. Most users should use
            with_batch_size() for multi-row batching instead.
        """
        self._processing_spec.batch_size = size
        return self

    def with_concurrency(self, threads: int) -> "PipelineBuilder":
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

    def with_checkpoint_interval(self, rows: int) -> "PipelineBuilder":
        """
        Configure checkpoint frequency.

        Args:
            rows: Rows between checkpoints

        Returns:
            Self for chaining
        """
        self._processing_spec.checkpoint_interval = rows
        return self

    def with_rate_limit(self, rpm: int) -> "PipelineBuilder":
        """
        Configure rate limiting.

        Prevents hitting API rate limits by throttling requests using token bucket algorithm.
        Set this below your provider's actual limit for safety.

        Args:
            rpm: Requests per minute (typical: 20-60 for free tiers, 100+ for paid)

        Returns:
            Self for chaining

        Example:
            ```python
            # OpenAI free tier (60 RPM limit)
            builder.with_rate_limit(50)

            # Groq free tier (30 RPM limit)
            builder.with_rate_limit(25)

            # Paid tier with high limits
            builder.with_rate_limit(100)
            ```

        Note:
            Rate limiting is applied per pipeline execution, not globally.
        """
        self._processing_spec.rate_limit_rpm = rpm
        return self

    def with_max_retries(self, retries: int) -> "PipelineBuilder":
        """
        Configure maximum retry attempts.

        Args:
            retries: Maximum number of retry attempts

        Returns:
            Self for chaining
        """
        self._processing_spec.max_retries = retries
        return self

    def with_max_budget(self, budget: float) -> "PipelineBuilder":
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

    def with_error_policy(self, policy: str) -> "PipelineBuilder":
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

    def with_checkpoint_dir(self, directory: str) -> "PipelineBuilder":
        """
        Configure checkpoint directory.

        Args:
            directory: Path to checkpoint directory

        Returns:
            Self for chaining
        """
        self._processing_spec.checkpoint_dir = Path(directory)
        return self

    def with_parser(self, parser: any) -> "PipelineBuilder":
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

    def to_csv(self, path: str) -> "PipelineBuilder":
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
    ) -> "PipelineBuilder":
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

    def with_executor(self, executor: ExecutionStrategy) -> "PipelineBuilder":
        """
        Set custom execution strategy.

        Args:
            executor: ExecutionStrategy instance

        Returns:
            Self for chaining
        """
        self._executor = executor
        return self

    def with_async_execution(self, max_concurrency: int = 10) -> "PipelineBuilder":
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

    def with_streaming(self, chunk_size: int = 1000) -> "PipelineBuilder":
        """
        Use streaming execution strategy.

        Processes data in chunks for memory-efficient handling.
        Ideal for large datasets (100K+ rows).

        Args:
            chunk_size: Number of rows per chunk

        Returns:
            Self for chaining
        """
        self._executor = StreamingExecutor(chunk_size=chunk_size)
        return self

    def with_progress_mode(self, mode: str = "auto") -> "PipelineBuilder":
        """
        Configure progress tracking mode.

        Args:
            mode: Progress tracking mode
                - "auto": Auto-detect (rich if TTY, else logging) [default]
                - "rich": Beautiful progress bars with ETA
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
    ) -> "PipelineBuilder":
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
        self, name: str, config: dict[str, any] | None = None
    ) -> "PipelineBuilder":
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

    def build(self) -> Pipeline:
        """
        Build final Pipeline.

        Validates all configurations and constructs the Pipeline object ready for execution.
        This is the final step in the builder chain.

        Returns:
            Configured Pipeline ready to execute

        Raises:
            ValueError: If required specifications missing (dataset, prompt, or LLM)

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
        if self._custom_parser is not None:
            metadata["custom_parser"] = self._custom_parser
        if self._custom_llm_client is not None:
            metadata["custom_llm_client"] = self._custom_llm_client
        if self._custom_stages:
            metadata["custom_stages"] = self._custom_stages
        if self._observers:
            metadata["observers"] = self._observers

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
        )
