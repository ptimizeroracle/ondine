"""LLM invocation stage with concurrency and retry logic."""

import asyncio
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from ondine.adapters.llm_client import LLMClient
from ondine.core.error_handler import ErrorAction, ErrorHandler
from ondine.core.exceptions import (
    InvalidAPIKeyError,
    ModelNotFoundError,
    QuotaExceededError,
)
from ondine.core.models import (
    CostEstimate,
    LLMResponse,
    PromptBatch,
    ResponseBatch,
    ValidationResult,
)
from ondine.core.specifications import ErrorPolicy
from ondine.orchestration.concurrency_controller import ConcurrencyController
from ondine.orchestration.deployment_tracker import DeploymentTracker
from ondine.orchestration.progress_reporter import ProgressReporter
from ondine.stages.batch_processor import BatchProcessor
from ondine.stages.pipeline_stage import PipelineStage
from ondine.utils import (
    NetworkError,
    RateLimiter,
    RateLimitError,
    RetryHandler,
)


class LLMInvocationStage(PipelineStage[list[PromptBatch], list[ResponseBatch]]):
    """
    Invoke LLM with prompts using concurrency and retries.

    Responsibilities:
    - Execute LLM calls with rate limiting
    - Handle retries for transient failures
    - Track tokens and costs
    - Support concurrent processing

    Uses extracted components for clean separation:
    - ConcurrencyController: Semaphore + rate limiting
    - DeploymentTracker: Router deployment mapping
    - ProgressReporter: UI progress updates
    - BatchProcessor: Flatten/reconstruct batches
    """

    def __init__(
        self,
        llm_client: LLMClient,
        concurrency: int = 5,
        rate_limiter: RateLimiter | None = None,
        retry_handler: RetryHandler | None = None,
        error_policy: ErrorPolicy = ErrorPolicy.SKIP,
        max_retries: int = 3,
        output_cls: type[BaseModel] | None = None,
    ):
        """
        Initialize LLM invocation stage.

        Args:
            llm_client: LLM client instance
            concurrency: Max concurrent requests
            rate_limiter: Optional rate limiter
            retry_handler: Optional retry handler
            error_policy: Policy for handling errors
            max_retries: Maximum retry attempts
            output_cls: Optional Pydantic model for structured output
        """
        super().__init__("LLMInvocation")
        self.llm_client = llm_client
        self.concurrency = concurrency
        self.rate_limiter = rate_limiter
        self.retry_handler = retry_handler or RetryHandler()
        self.output_cls = output_cls
        self.error_handler = ErrorHandler(
            policy=error_policy,
            max_retries=max_retries,
            default_value_factory=lambda: LLMResponse(
                text="",
                tokens_in=0,
                tokens_out=0,
                model=llm_client.model,
                cost=Decimal("0.0"),
                latency_ms=0.0,
            ),
        )

        # Extracted components
        self._concurrency_ctrl = ConcurrencyController(concurrency, rate_limiter)
        self._batch_processor = BatchProcessor()

    def process(self, batches: list[PromptBatch], context: Any) -> list[ResponseBatch]:
        """Execute LLM calls for all prompt batches using flatten-then-concurrent pattern."""
        # Initialize token tracking
        if "token_tracking" not in context.intermediate_data:
            context.intermediate_data["token_tracking"] = {
                "input_tokens": 0,
                "output_tokens": 0,
            }

        # Execute async processing
        try:
            return asyncio.run(self._process_async(batches, context))
        except RuntimeError:
            # Loop already running (e.g. Jupyter)
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.run(self._process_async(batches, context))

    async def _process_async(
        self, batches: list[PromptBatch], context: Any
    ) -> list[ResponseBatch]:
        """Async implementation of process()."""
        # Initialize global connection pool
        await self.llm_client.start()

        try:
            # Setup deployment tracker from Router config
            model_list = self._get_router_model_list()
            deployment_tracker = DeploymentTracker(model_list)

            # Setup progress reporter
            progress_tracker = getattr(context, "progress_tracker", None)
            progress_reporter = ProgressReporter(progress_tracker, deployment_tracker)

            # Calculate total rows and start progress
            # Note: We don't pre-create deployment bars anymore - they're created
            # dynamically as deployments are actually used (more accurate)
            total_rows = BatchProcessor.calculate_total_rows(batches)
            if progress_tracker:
                # Pass empty list - deployment bars created dynamically
                progress_reporter.start(self.name, total_rows, deployments=[])

            # Store for access in processing loop
            self._deployment_tracker = deployment_tracker
            self._progress_reporter = progress_reporter
            self._total_rows = total_rows

            # Flatten batches for concurrent processing
            items, batch_map = BatchProcessor.flatten(batches)

            # Process all prompts concurrently
            all_responses = await self._process_all_concurrent(items, context)

            # Reconstruct batches from flat responses
            response_batches = BatchProcessor.reconstruct(
                all_responses, batches, batch_map
            )

            # Notify and finish progress
            if hasattr(context, "notify_progress"):
                context.notify_progress()
            progress_reporter.finish()

            return response_batches
        finally:
            # Cleanup global connection pool
            await self.llm_client.stop()

    def _get_router_model_list(self) -> list[dict] | None:
        """Get model list from Router if available."""
        if hasattr(self.llm_client, "router") and self.llm_client.router:
            return getattr(self.llm_client.router, "model_list", None)
        return None

    async def _process_all_concurrent(
        self, items: list, context: Any
    ) -> list[LLMResponse]:
        """Process all prompt items concurrently using asyncio."""
        semaphore = asyncio.Semaphore(self.concurrency)

        async def _process_one(idx: int, item) -> LLMResponse:
            async with semaphore:
                return await self._process_single_item(idx, item, context)

        tasks = [_process_one(idx, item) for idx, item in enumerate(items)]
        responses = await asyncio.gather(*tasks)

        # Log distribution summary
        self._log_distribution_summary()

        return list(responses)

    async def _process_single_item(self, idx: int, item, context: Any) -> LLMResponse:
        """Process a single prompt item with error handling."""
        try:
            response = await self._invoke_async(item.prompt, item.metadata)

            if response is None:
                raise RuntimeError("LLM invocation returned None (unexpected)")

            # Track deployment distribution
            deployment_id = self._extract_deployment_id(response)
            if deployment_id:
                self._deployment_tracker.record_request(deployment_id)

            # Update progress
            batch_size = BatchProcessor.get_batch_size(item.metadata)
            self._progress_reporter.update(
                rows_completed=batch_size,
                cost=response.cost,
                deployment_id=deployment_id,
            )

            # Update context
            self._update_context(context, idx, item.metadata, response)

            return response

        except Exception as e:
            return self._handle_error(
                e, idx, len(context.intermediate_data.get("items", []))
            )

    def _invoke_with_retry_and_ratelimit(
        self,
        prompt: str,
        row_metadata: Any = None,
        context: Any = None,
        row_index: int = 0,
    ) -> LLMResponse:
        """
        Invoke LLM with rate limiting and retries (sync version).

        This is a backward-compatible sync wrapper around the async implementation.
        Used by tests and legacy code paths.
        """
        # Extract system message from row metadata
        system_message = None
        if row_metadata and hasattr(row_metadata, "custom") and row_metadata.custom:
            system_message = row_metadata.custom.get("system_message")

        def _invoke() -> LLMResponse:
            # Acquire rate limit token
            if self.rate_limiter:
                self.rate_limiter.acquire()

            # Invoke LLM (sync)
            if self.output_cls:
                return self.llm_client.structured_invoke(
                    prompt, self.output_cls, system_message=system_message
                )
            return self.llm_client.invoke(prompt, system_message=system_message)

        # Execute with retry handler
        return self.retry_handler.execute(_invoke)

    async def _invoke_async(self, prompt: str, metadata: Any) -> LLMResponse:
        """Invoke LLM async with retries."""
        # Extract system message from metadata
        system_message = None
        if metadata and hasattr(metadata, "custom") and metadata.custom:
            system_message = metadata.custom.get("system_message")

        async def _invoke() -> LLMResponse:
            # Rate limiting (run in executor since it's blocking)
            if self.rate_limiter:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.rate_limiter.acquire)

            # Invoke LLM
            if self.output_cls:
                return await self.llm_client.structured_invoke_async(
                    prompt, self.output_cls, system_message=system_message
                )
            return await self.llm_client.ainvoke(prompt, system_message=system_message)

        return await self.retry_handler.execute_async(_invoke)

    def _extract_deployment_id(self, response: LLMResponse) -> str | None:
        """Extract deployment ID from LLM response."""
        try:
            if hasattr(response, "metadata") and isinstance(response.metadata, dict):
                if "model_id" in response.metadata:
                    return response.metadata["model_id"]
            return None
        except Exception:
            return None

    def _update_context(
        self, context: Any, idx: int, metadata: Any, response: LLMResponse
    ) -> None:
        """Update execution context with response metrics."""
        if not context:
            return

        # Update row index
        batch_size = BatchProcessor.get_batch_size(metadata)
        if metadata.custom and metadata.custom.get("is_batch"):
            if idx == 0:
                context.update_row(metadata.row_index + batch_size - 1)
            else:
                context.update_row(context.last_processed_row + batch_size)
        else:
            context.update_row(metadata.row_index)

        # Update cost and token tracking
        if hasattr(response, "cost") and hasattr(response, "tokens_in"):
            context.add_cost(response.cost, response.tokens_in + response.tokens_out)
            context.intermediate_data["token_tracking"]["input_tokens"] += (
                response.tokens_in
            )
            context.intermediate_data["token_tracking"]["output_tokens"] += (
                response.tokens_out
            )

    def _handle_error(self, error: Exception, idx: int, total: int) -> LLMResponse:
        """Handle processing error according to error policy."""
        decision = self.error_handler.handle_error(
            error,
            {"stage": self.name, "prompt_index": idx, "total_prompts": total},
        )

        if decision.action == ErrorAction.SKIP:
            return LLMResponse(
                text="[SKIPPED]",
                tokens_in=0,
                tokens_out=0,
                model=self.llm_client.model,
                cost=Decimal("0.0"),
                latency_ms=0.0,
                metadata={"error": str(error), "action": "skipped"},
            )
        if decision.action == ErrorAction.USE_DEFAULT:
            return decision.default_value
        # FAIL action
        raise error

    def _log_distribution_summary(self) -> None:
        """Log Router distribution summary."""
        if not (hasattr(self.llm_client, "router") and self.llm_client.router):
            return

        self.logger.info("=" * 70)
        distribution = self._deployment_tracker.get_distribution()

        if distribution:
            self.logger.info("Router Distribution Summary (ACTUAL):")
            total_requests = sum(distribution.values())

            for dep_id in sorted(distribution.keys()):
                count = distribution[dep_id]
                percentage = (count / total_requests * 100) if total_requests > 0 else 0
                friendly_id = self._deployment_tracker.get_friendly_id(dep_id) or dep_id
                self.logger.info(
                    f"  • {friendly_id}: {count}/{total_requests} requests ({percentage:.1f}%)"
                )

            self.logger.info(f"Total API calls: {total_requests}")
        else:
            self.logger.info("Router Distribution Summary:")
            self.logger.info(
                "  WARNING: Could not extract deployment info from responses"
            )

        self.logger.info("=" * 70)

    def validate_input(self, batches: list[PromptBatch]) -> ValidationResult:
        """Validate prompt batches."""
        result = ValidationResult(is_valid=True)

        if not batches:
            result.add_error("No prompt batches provided")

        for batch in batches:
            if not batch.prompts:
                result.add_error(f"Batch {batch.batch_id} has no prompts")
            if len(batch.prompts) != len(batch.metadata):
                result.add_error(f"Batch {batch.batch_id} prompt/metadata mismatch")

        return result

    def estimate_cost(self, batches: list[PromptBatch]) -> CostEstimate:
        """Estimate LLM invocation cost."""
        total_input_tokens = 0
        total_output_tokens = 0

        for batch in batches:
            for prompt in batch.prompts:
                input_tokens = self.llm_client.estimate_tokens(prompt)
                total_input_tokens += input_tokens
                total_output_tokens += int(input_tokens * 0.5)

        total_cost = self.llm_client.calculate_cost(
            total_input_tokens, total_output_tokens
        )

        return CostEstimate(
            total_cost=total_cost,
            total_tokens=total_input_tokens + total_output_tokens,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            rows=sum(len(b.prompts) for b in batches),
            confidence="estimate",
        )

    def _classify_error(self, error: Exception) -> Exception:
        """Classify error as retryable or non-retryable."""
        error_str = str(error).lower()

        # Try to unwrap Instructor retry exceptions
        try:
            from instructor.core.exceptions import InstructorRetryException

            if isinstance(error, InstructorRetryException):
                if hasattr(error, "last_attempt") and error.last_attempt:
                    if hasattr(error.last_attempt, "exception"):
                        inner_exc = error.last_attempt.exception()
                        if inner_exc:
                            error = inner_exc
                            error_str = str(error).lower()
        except ImportError:
            pass

        # Network errors (retryable)
        if any(
            p in error_str for p in ["network", "timeout", "connection", "503", "502"]
        ):
            return NetworkError(str(error))

        # Quota errors (non-retryable)
        quota_patterns = [
            "quota exceeded",
            "insufficient_quota",
            "billing",
            "limit exceeded",
        ]
        if any(p in error_str for p in quota_patterns):
            return QuotaExceededError(f"Quota error: {error}")

        # Rate limit (retryable)
        if "rate" in error_str or "429" in error_str:
            return RateLimitError(str(error))

        # Auth errors (non-retryable)
        auth_patterns = ["invalid api key", "401", "403", "unauthorized"]
        if any(p in error_str for p in auth_patterns):
            return InvalidAPIKeyError(f"Authentication error: {error}")

        # Model errors
        model_patterns = [
            "decommissioned",
            "not found",
            "does not exist",
            "invalid model",
        ]
        if any(p in error_str for p in model_patterns):
            if hasattr(self.llm_client, "router") and self.llm_client.router:
                return NetworkError(f"Router node failed (retryable): {error}")
            return ModelNotFoundError(f"Model error: {error}")

        return error
