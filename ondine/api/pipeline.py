"""
Main Pipeline class - the Facade for the entire system.

This is the primary entry point that users interact with.
"""

from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from ondine.adapters import (
    LocalFileCheckpointStorage,
    create_llm_client,
)
from ondine.core.models import (
    CostEstimate,
    ExecutionResult,
    ProcessingStats,
    ValidationResult,
)
from ondine.core.specifications import (
    PipelineSpecifications,
)
from ondine.orchestration import (
    CostTrackingObserver,
    ExecutionContext,
    ExecutionObserver,
    ExecutionStrategy,
    LoggingObserver,
    ProgressBarObserver,
    StateManager,
    StreamingExecutor,
    SyncExecutor,
    create_progress_tracker,
)
from ondine.stages import (
    BatchAggregatorStage,
    BatchDisaggregatorStage,
    DataLoaderStage,
    LLMInvocationStage,
    PromptFormatterStage,
    ResponseParserStage,
    ResultWriterStage,
    create_response_parser,
)
from ondine.utils import RateLimiter, RetryHandler, get_logger

logger = get_logger(__name__)


class Pipeline:
    """
    Main pipeline class - Facade for dataset processing.

    Provides high-level interface for building and executing LLM-powered data
    transformations. Handles orchestration, state management, cost tracking,
    checkpointing, and error handling.

    This is typically created via PipelineBuilder or QuickPipeline, not directly.

    Example:
        ```python
        from ondine import PipelineBuilder

        # Create via builder (recommended)
        pipeline = (
            PipelineBuilder.create()
            .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
            .with_prompt("Summarize: {text}")
            .with_llm(provider="openai", model="gpt-4o-mini")
            .build()
        )

        # Execute
        result = pipeline.execute()
        print(f"Processed {result.metrics.total_rows} rows")
        print(f"Cost: ${result.costs.total_cost}")
        ```

    Note:
        Use PipelineBuilder for construction, not direct instantiation.
    """

    def __init__(
        self,
        specifications: PipelineSpecifications,
        dataframe: pd.DataFrame | None = None,
        executor: ExecutionStrategy | None = None,
    ):
        """
        Initialize pipeline with specifications.

        Args:
            specifications: Complete pipeline configuration
            dataframe: Optional pre-loaded DataFrame
            executor: Optional execution strategy (default: SyncExecutor)
        """
        self.id = uuid4()
        self.specifications = specifications
        self.dataframe = dataframe
        self.executor = executor or SyncExecutor()
        self.observers: list[ExecutionObserver] = []
        self.logger = get_logger(f"{__name__}.{self.id}")

    def add_observer(self, observer: ExecutionObserver) -> "Pipeline":
        """
        Add execution observer.

        Args:
            observer: Observer to add

        Returns:
            Self for chaining
        """
        self.observers.append(observer)
        return self

    def validate(self) -> ValidationResult:
        """
        Validate pipeline configuration.

        Returns:
            ValidationResult with any errors/warnings
        """
        result = ValidationResult(is_valid=True)

        # Validate dataset spec
        if not self.specifications.dataset.input_columns:
            result.add_error("No input columns specified")

        if not self.specifications.dataset.output_columns:
            result.add_error("No output columns specified")

        # Validate that input columns exist in dataframe (if dataframe is provided)
        if self.dataframe is not None and self.specifications.dataset.input_columns:
            df_cols = set(self.dataframe.columns)
            input_cols = set(self.specifications.dataset.input_columns)
            missing_cols = input_cols - df_cols
            if missing_cols:
                result.add_error(
                    f"Input columns not found in dataframe: {missing_cols}"
                )

        # Validate prompt spec
        if not self.specifications.prompt.template:
            result.add_error("No prompt template specified")
        else:
            # Check that template variables match input columns
            import re

            template_vars = set(
                re.findall(r"\{(\w+)\}", self.specifications.prompt.template)
            )
            input_cols = set(self.specifications.dataset.input_columns)
            missing_vars = template_vars - input_cols
            if missing_vars:
                result.add_error(
                    f"Template variables not in input columns: {missing_vars}"
                )

        # Validate LLM spec
        if not self.specifications.llm.model:
            result.add_error("No LLM model specified")

        return result

    def estimate_cost(self) -> CostEstimate:
        """
        Estimate total processing cost.

        Returns:
            Cost estimate
        """
        # Create stages
        loader = DataLoaderStage(self.dataframe)

        # Load first few rows for estimation
        df = loader.process(self.specifications.dataset, ExecutionContext())
        sample_size = min(10, len(df))
        sample_df = df.head(sample_size)

        # Create formatter and get prompts
        formatter = PromptFormatterStage(self.specifications.processing.batch_size)
        batches = formatter.process(
            (sample_df, self.specifications.prompt), ExecutionContext()
        )

        # Create LLM client and estimate
        llm_client = create_llm_client(self.specifications.llm)
        llm_stage = LLMInvocationStage(llm_client)

        sample_estimate = llm_stage.estimate_cost(batches)

        # Scale to full dataset
        scale_factor = Decimal(len(df)) / Decimal(sample_size)

        return CostEstimate(
            total_cost=sample_estimate.total_cost * scale_factor,
            total_tokens=int(sample_estimate.total_tokens * float(scale_factor)),
            input_tokens=int(sample_estimate.input_tokens * float(scale_factor)),
            output_tokens=int(sample_estimate.output_tokens * float(scale_factor)),
            rows=len(df),
            confidence="sample-based",
        )

    def execute(self, resume_from: UUID | None = None) -> ExecutionResult:
        """
        Execute pipeline end-to-end.

        Runs all stages: data loading, prompt formatting, LLM invocation, response parsing,
        and result writing. Handles checkpointing, cost tracking, and error recovery.

        Args:
            resume_from: Optional session ID to resume from checkpoint (for fault tolerance)

        Returns:
            ExecutionResult containing:
                - data: DataFrame with results
                - metrics: Processing statistics (total_rows, success_count, etc.)
                - costs: Cost breakdown (total_cost, input_tokens, output_tokens)
                - duration: Execution time in seconds
                - errors: List of any errors encountered

        Example:
            ```python
            # Execute pipeline
            result = pipeline.execute()

            # Access results
            print(f"Processed: {result.metrics.total_rows} rows")
            print(f"Successful: {result.metrics.success_count} rows")
            print(f"Cost: ${result.costs.total_cost}")
            print(f"Time: {result.duration:.2f}s")

            # Access output data
            result.data.to_csv("output.csv", index=False)

            # Resume from checkpoint (if pipeline was interrupted)
            result = pipeline.execute(resume_from=previous_session_id)
            ```

        Note:
            Progress is automatically saved via checkpoints. If execution fails,
            use resume_from to continue from the last checkpoint.
        """
        # Validate first
        validation = self.validate()
        if not validation.is_valid:
            raise ValueError(f"Pipeline validation failed: {validation.errors}")

        # Create or restore execution context
        state_manager = StateManager(
            storage=LocalFileCheckpointStorage(
                self.specifications.processing.checkpoint_dir
            ),
            checkpoint_interval=self.specifications.processing.checkpoint_interval,
        )

        if resume_from:
            # Resume from checkpoint
            context = state_manager.load_checkpoint(resume_from)
            if not context:
                raise ValueError(f"No checkpoint found for session {resume_from}")
            self.logger.info(
                f"Resuming from checkpoint at row {context.last_processed_row}"
            )
        else:
            # Create new context
            context = ExecutionContext(pipeline_id=self.id)

        # Add default observers if none specified
        if not self.observers:
            self.observers = [
                ProgressBarObserver(),
                LoggingObserver(),
                CostTrackingObserver(),
            ]

        # Attach observers to context for progress notifications
        context.observers = self.observers

        # Initialize new observability system if observers configured
        observer_configs = self.specifications.metadata.get("observers", [])
        if observer_configs:
            from ondine.observability.dispatcher import ObserverDispatcher
            from ondine.observability.registry import ObserverRegistry

            # Instantiate observers from configuration
            new_observers = []
            for observer_name, observer_config in observer_configs:
                try:
                    observer_class = ObserverRegistry.get(observer_name)
                    observer_instance = observer_class(config=observer_config)
                    new_observers.append(observer_instance)
                    self.logger.info(f"Initialized observer: {observer_name}")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to initialize observer '{observer_name}': {e}"
                    )

            # Create dispatcher and attach to context
            if new_observers:
                context.observer_dispatcher = ObserverDispatcher(new_observers)

                # Emit pipeline start event
                from ondine.observability.events import PipelineStartEvent

                start_event = PipelineStartEvent(
                    pipeline_id=self.id,
                    run_id=context.session_id,
                    timestamp=datetime.now(),
                    trace_id=context.trace_id,
                    span_id=context.span_id,
                    config={},
                    metadata=self.specifications.metadata,
                    total_rows=0,  # Will be updated after data loading
                )
                context.observer_dispatcher.dispatch("pipeline_start", start_event)

        # Notify legacy observers of start
        for observer in self.observers:
            observer.on_pipeline_start(self, context)

        try:
            # Execute stages (preprocessing happens inside if enabled)
            result_df = self._execute_stages(context, state_manager)

            # Mark completion
            context.end_time = datetime.now()

            # Create execution result
            # Extract token tracking from intermediate_data (populated by LLMInvocationStage)
            token_tracking = context.intermediate_data.get("token_tracking", {})
            input_tokens = token_tracking.get("input_tokens", 0)
            output_tokens = token_tracking.get("output_tokens", 0)

            result = ExecutionResult(
                data=result_df,
                metrics=context.get_stats(),
                costs=CostEstimate(
                    total_cost=context.accumulated_cost,
                    total_tokens=context.accumulated_tokens,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    rows=context.total_rows,
                    confidence="actual",
                ),
                execution_id=context.session_id,
                start_time=context.start_time,
                end_time=context.end_time,
                success=True,
            )

            # Optional: Auto-retry failed rows
            if self.specifications.processing.auto_retry_failed:
                # Get preprocessed data from context (or loaded data if no preprocessing)
                retry_source_df = context.intermediate_data.get("preprocessed_data")
                if retry_source_df is None:
                    retry_source_df = context.intermediate_data.get("loaded_data")
                result = self._auto_retry_failed_rows(result, retry_source_df)

            # Cleanup checkpoints on success
            state_manager.cleanup_checkpoints(context.session_id)

            # Notify legacy observers of completion
            for observer in self.observers:
                observer.on_pipeline_complete(context, result)

            # Emit pipeline end event for new observability system
            if context.observer_dispatcher:
                from ondine.observability.events import PipelineEndEvent

                end_event = PipelineEndEvent(
                    pipeline_id=self.id,
                    run_id=context.session_id,
                    success=True,
                    timestamp=datetime.now(),
                    trace_id=context.trace_id,
                    span_id=context.span_id,
                    total_duration_ms=(
                        (context.end_time - context.start_time).total_seconds() * 1000
                        if context.end_time
                        else 0
                    ),
                    rows_processed=result.metrics.processed_rows,
                    rows_succeeded=result.metrics.processed_rows
                    - result.metrics.failed_rows,
                    rows_failed=result.metrics.failed_rows,
                    rows_skipped=result.metrics.skipped_rows,
                    total_cost=result.costs.total_cost,
                    total_tokens=result.costs.total_tokens,
                    input_tokens=result.costs.input_tokens,
                    output_tokens=result.costs.output_tokens,
                )
                context.observer_dispatcher.dispatch("pipeline_end", end_event)

                # Flush and close observers
                context.observer_dispatcher.flush_all()
                context.observer_dispatcher.close_all()

            return result

        except Exception as e:
            # Save checkpoint on error
            state_manager.save_checkpoint(context)
            self.logger.error(
                f"Pipeline failed. Checkpoint saved. "
                f"Resume with: pipeline.execute(resume_from=UUID('{context.session_id}'))"
            )

            # Notify legacy observers of error
            for observer in self.observers:
                observer.on_pipeline_error(context, e)

            # Emit error event for new observability system
            if context.observer_dispatcher:
                from ondine.observability.events import ErrorEvent

                error_event = ErrorEvent(
                    pipeline_id=self.id,
                    run_id=context.session_id,
                    timestamp=datetime.now(),
                    trace_id=context.trace_id,
                    span_id=context.span_id,
                    error=e,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    stack_trace="",  # Could add full traceback if needed
                )
                context.observer_dispatcher.dispatch("error", error_event)

                # Flush and close observers even on error
                context.observer_dispatcher.flush_all()
                context.observer_dispatcher.close_all()

            raise

    def _execute_stages(
        self, context: ExecutionContext, state_manager: StateManager
    ) -> pd.DataFrame:
        """Execute all pipeline stages with checkpointing."""
        specs = self.specifications

        # Create progress tracker
        progress_tracker = create_progress_tracker(specs.processing.progress_mode)

        # Execute with progress tracking
        with progress_tracker:
            context.progress_tracker = progress_tracker  # Store in context for stages
            return self._execute_stages_with_tracking(context, state_manager)

    def _execute_stages_with_tracking(
        self, context: ExecutionContext, state_manager: StateManager
    ) -> pd.DataFrame:
        """Execute stages with progress tracking enabled."""
        specs = self.specifications

        # Create budget controller if max_budget specified
        budget_controller = None
        if specs.processing.max_budget:
            from ondine.utils import BudgetController

            budget_controller = BudgetController(
                max_budget=specs.processing.max_budget,
                warn_at_75=True,
                warn_at_90=True,
                fail_on_exceed=True,
            )

        # Stage 1: Load data
        loader = DataLoaderStage(self.dataframe)
        df = self._execute_stage(loader, specs.dataset, context)
        context.intermediate_data["loaded_data"] = df

        # Optional: Preprocess loaded data
        if specs.processing.enable_preprocessing:
            from ondine.utils.input_preprocessing import preprocess_dataframe

            self.logger.info("Preprocessing loaded data...")
            df, stats = preprocess_dataframe(
                df,
                input_columns=specs.dataset.input_columns,
                max_length=specs.processing.preprocessing_max_length,
            )
            self.logger.info(
                f"Preprocessing complete: "
                f"{stats.reduction_pct:.1f}% char reduction, "
                f"{stats.truncated_count} truncated"
            )
            # Store preprocessed data for retry
            context.intermediate_data["preprocessed_data"] = df

        # Stage 2: Format prompts
        formatter = PromptFormatterStage(specs.processing.batch_size)
        batches = self._execute_stage(formatter, (df, specs.prompt), context)
        context.intermediate_data["prompt_batches"] = batches

        # Stage 2.5: Aggregate into batch prompts (if batch_size > 1)
        if specs.prompt.batch_size > 1:
            from ondine.strategies.json_batch_strategy import JsonBatchStrategy

            # Select strategy
            if specs.prompt.batch_strategy == "json":
                strategy = JsonBatchStrategy()
            else:
                # CSV strategy will be implemented in Phase 4
                strategy = JsonBatchStrategy()  # Fallback to JSON

            aggregator = BatchAggregatorStage(
                batch_size=specs.prompt.batch_size,
                strategy=strategy,
                model=specs.llm.model,
                validate_context_window=False,  # Disabled for performance (slow on large datasets)
            )
            batches = self._execute_stage(aggregator, batches, context)
            context.intermediate_data["aggregated_batches"] = batches
            self.logger.info(
                f"Batch aggregation enabled: {specs.prompt.batch_size} rows per API call"
            )

        # Stage 3: Invoke LLM
        llm_client = create_llm_client(specs.llm)
        rate_limiter = (
            RateLimiter(
                specs.processing.rate_limit_rpm,
                burst_size=min(
                    20, specs.processing.concurrency
                ),  # Limit burst to prevent rate limit errors
            )
            if specs.processing.rate_limit_rpm
            else None
        )
        retry_handler = RetryHandler(
            max_attempts=specs.processing.max_retries,
            initial_delay=specs.processing.retry_delay,
        )

        llm_stage = LLMInvocationStage(
            llm_client,
            concurrency=specs.processing.concurrency,
            rate_limiter=rate_limiter,
            retry_handler=retry_handler,
            error_policy=specs.processing.error_policy,
            max_retries=specs.processing.max_retries,
        )
        # Stage 3: Execute LLM invocation
        response_batches = self._execute_stage(llm_stage, batches, context)
        context.intermediate_data["response_batches"] = response_batches

        # Check budget after LLM invocation
        if budget_controller:
            budget_controller.check_budget(context.accumulated_cost)

        # Save checkpoint after expensive LLM stage
        if state_manager.should_checkpoint(context.last_processed_row):
            state_manager.save_checkpoint(context)

        # Stage 3.5: Disaggregate batch responses (if batch_size > 1)
        if specs.prompt.batch_size > 1:
            from ondine.strategies.json_batch_strategy import JsonBatchStrategy

            # Select strategy (same as aggregator)
            if specs.prompt.batch_strategy == "json":
                strategy = JsonBatchStrategy()
            else:
                strategy = JsonBatchStrategy()  # Fallback to JSON

            disaggregator = BatchDisaggregatorStage(
                strategy=strategy,
                retry_failed_individually=True,
            )
            response_batches = self._execute_stage(
                disaggregator, response_batches, context
            )
            context.intermediate_data["disaggregated_responses"] = response_batches
            self.logger.info("Batch disaggregation complete")

        # Stage 4: Parse responses (using configured parser)
        # Check if custom parser provided in metadata
        custom_parser = specs.metadata.get("custom_parser") if specs.metadata else None
        if custom_parser:
            parser = custom_parser
        else:
            parser = create_response_parser(
                prompt_spec=specs.prompt,
                output_columns=specs.dataset.output_columns,
            )
        parser_stage = ResponseParserStage(
            parser=parser,
            output_columns=specs.dataset.output_columns,
        )
        results_df = self._execute_stage(
            parser_stage,
            (response_batches, specs.dataset.output_columns),
            context,
        )

        # Stage 5: Write results (if output spec provided)
        if specs.output:
            writer = ResultWriterStage()
            return self._execute_stage(writer, (df, results_df, specs.output), context)
        # Merge results with original
        for col in results_df.columns:
            df[col] = results_df[col]
        return df

    async def execute_async(self, resume_from: UUID | None = None) -> ExecutionResult:
        """
        Execute pipeline asynchronously.

        Uses AsyncExecutor for non-blocking execution. Ideal for integration
        with FastAPI, aiohttp, and other async frameworks.

        Args:
            resume_from: Optional session ID to resume from checkpoint

        Returns:
            ExecutionResult with data and metrics

        Raises:
            ValueError: If executor doesn't support async
        """
        if not self.executor.supports_async():
            raise ValueError(
                "Current executor doesn't support async. "
                "Use AsyncExecutor: Pipeline(specs, executor=AsyncExecutor())"
            )

        # For now, wrap synchronous execution in async
        # TODO: Implement fully async execution pipeline
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, resume_from)

    def execute_stream(
        self, chunk_size: int | None = None
    ) -> Iterator[ExecutionResult]:
        """
        Execute pipeline in streaming mode.

        Processes data in chunks for memory-efficient handling of large datasets.
        Ideal for datasets that don't fit in memory.

        Args:
            chunk_size: Number of rows per chunk (uses executor's chunk_size if None)

        Yields:
            ExecutionResult objects for each processed chunk

        Raises:
            ValueError: If executor doesn't support streaming
        """
        if not self.executor.supports_streaming():
            raise ValueError(
                "Current executor doesn't support streaming. "
                "Use StreamingExecutor: Pipeline(specs, executor=StreamingExecutor())"
            )

        # Use executor's chunk_size if not provided
        if chunk_size is None and isinstance(self.executor, StreamingExecutor):
            chunk_size = self.executor.chunk_size
        elif chunk_size is None:
            chunk_size = 1000  # Default fallback

        # For now, execute the full pipeline and split result into chunks
        # TODO: Implement proper streaming execution that processes chunks independently
        result = self.execute()

        # Split the result data into chunks and yield as separate ExecutionResults
        total_rows = len(result.data)
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_data = result.data.iloc[start_idx:end_idx].copy()

            # Create a chunk result with proportional metrics
            chunk_rows = len(chunk_data)
            chunk_result = ExecutionResult(
                data=chunk_data,
                metrics=ProcessingStats(
                    total_rows=chunk_rows,
                    processed_rows=chunk_rows,
                    failed_rows=0,
                    skipped_rows=0,
                    rows_per_second=result.metrics.rows_per_second,
                    total_duration_seconds=result.metrics.total_duration_seconds
                    * (chunk_rows / total_rows),
                    stage_durations=result.metrics.stage_durations,
                ),
                costs=CostEstimate(
                    total_cost=result.costs.total_cost
                    * Decimal(chunk_rows / total_rows),
                    total_tokens=int(
                        result.costs.total_tokens * (chunk_rows / total_rows)
                    ),
                    input_tokens=int(
                        result.costs.input_tokens * (chunk_rows / total_rows)
                    ),
                    output_tokens=int(
                        result.costs.output_tokens * (chunk_rows / total_rows)
                    ),
                    rows=chunk_rows,
                    confidence=result.costs.confidence,
                ),
                execution_id=result.execution_id,
                start_time=result.start_time,
                end_time=result.end_time,
                success=True,
            )
            yield chunk_result

    def _execute_stage(
        self, stage: Any, input_data: Any, context: ExecutionContext
    ) -> Any:
        """Execute a single stage with observer notifications."""
        # Notify observers of stage start
        for observer in self.observers:
            observer.on_stage_start(stage, context)

        try:
            # Execute stage
            result = stage.execute(input_data, context)

            # Notify observers of completion
            for observer in self.observers:
                observer.on_stage_complete(stage, context, result)

            return result

        except Exception as e:
            # Notify observers of error
            for observer in self.observers:
                observer.on_stage_error(stage, context, e)
            raise

    def _auto_retry_failed_rows(
        self, result: ExecutionResult, original_df: pd.DataFrame
    ) -> ExecutionResult:
        """
        Row-level quality retry (re-executes entire pipeline for failed rows).

        Scope: Output quality validation (null/empty detection across ALL columns)
        Use when: LLM returns null or empty outputs
        Different from: RetryHandler (this retries whole row/pipeline, not just LLM call)

        Algorithm:
        1. Check output quality across ALL output columns
        2. Identify rows with null OR empty values in ANY column
        3. Re-execute pipeline for only failed rows (up to max_retry_attempts)
        4. Merge successful retries back into original result
        5. Repeat until all rows valid or max attempts reached

        Design Decision:
        - Creates NEW pipeline instance for retry (isolation)
        - Uses original preprocessed data (not re-preprocessed)
        - Disables auto_retry on retry pipeline (prevents infinite loop)

        See Also:
        - RetryHandler: Request-level retry for transient errors
        - ErrorHandler: Policy-based error orchestration
        - docs/architecture/decisions/ADR-006-retry-levels.md

        Args:
            result: Initial execution result
            original_df: Original (preprocessed) input dataframe

        Returns:
            ExecutionResult with retried rows merged in
        """
        if original_df is None:
            self.logger.warning("Cannot retry: original dataframe is None")
            return result

        specs = self.specifications
        output_cols = specs.dataset.output_columns

        # Check quality
        quality = result.validate_output_quality(output_cols)

        # Count both nulls and empties as failures
        total_failed = quality.null_outputs + quality.empty_outputs

        if total_failed == 0:
            self.logger.info("No failed rows to retry")
            return result

        self.logger.info(
            f"Auto-retry enabled: {quality.null_outputs} null + "
            f"{quality.empty_outputs} empty = {total_failed} failed outputs"
        )

        # Try up to max_retry_attempts
        for attempt in range(1, specs.processing.max_retry_attempts + 1):
            # Find null OR empty rows across ALL output columns
            failed_mask = pd.Series([False] * len(result.data), index=result.data.index)

            for col in output_cols:
                if col in result.data.columns:
                    null_mask = result.data[col].isna()
                    empty_mask = result.data[col].astype(str).str.strip() == ""
                    failed_mask |= null_mask | empty_mask

            failed_indices = result.data[failed_mask].index.tolist()

            if not failed_indices:
                break

            self.logger.info(
                f"Retry attempt {attempt}/{specs.processing.max_retry_attempts}: "
                f"{len(failed_indices)} rows"
            )

            # Extract failed rows from original (preprocessed) data
            retry_df = original_df.loc[failed_indices].copy()

            # Store original indices for mapping back
            original_indices = retry_df.index.tolist()
            retry_df = retry_df.reset_index(drop=True)

            # Create modified specs for retry (use dataframe, not file)
            from ondine.core.specifications import DataSourceType

            retry_specs = self.specifications.model_copy(deep=True)
            retry_specs.dataset.source_type = DataSourceType.DATAFRAME
            retry_specs.dataset.source_path = None  # Force use of dataframe
            retry_specs.processing.enable_preprocessing = False  # Already preprocessed
            retry_specs.processing.auto_retry_failed = (
                False  # Prevent infinite retry loop
            )
            retry_specs.output = None  # Don't write to file during retry

            # Create new pipeline for retry
            retry_pipeline = Pipeline(
                retry_specs,
                dataframe=retry_df,
            )

            # Execute retry
            retry_result = retry_pipeline.execute()

            # Merge retry results back (map reset indices to original indices)
            for col in output_cols:
                for new_idx, original_idx in enumerate(original_indices):
                    result.data.loc[original_idx, col] = retry_result.data.loc[
                        new_idx, col
                    ]

            # Update costs
            result.costs.total_cost += retry_result.costs.total_cost
            result.costs.total_tokens += retry_result.costs.total_tokens

            # Check quality again
            quality = result.validate_output_quality(output_cols)
            self.logger.info(
                f"After retry {attempt}: "
                f"{quality.valid_outputs}/{quality.total_rows} valid "
                f"({quality.success_rate:.1f}%), "
                f"{quality.null_outputs + quality.empty_outputs} still failed"
            )

            # Continue to next attempt to maximize completeness
            # (Don't stop early - use all attempts to get closest to 100%)

        return result
