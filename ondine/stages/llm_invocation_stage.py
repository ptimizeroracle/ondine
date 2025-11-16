"""LLM invocation stage with concurrency and retry logic."""

import concurrent.futures
import time
from decimal import Decimal
from typing import Any

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
    """

    def __init__(
        self,
        llm_client: LLMClient,
        concurrency: int = 5,
        rate_limiter: RateLimiter | None = None,
        retry_handler: RetryHandler | None = None,
        error_policy: ErrorPolicy = ErrorPolicy.SKIP,
        max_retries: int = 3,
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
        """
        super().__init__("LLMInvocation")
        self.llm_client = llm_client
        self.concurrency = concurrency
        self.rate_limiter = rate_limiter
        self.retry_handler = retry_handler or RetryHandler()
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

    def process(self, batches: list[PromptBatch], context: Any) -> list[ResponseBatch]:
        """Execute LLM calls for all prompt batches using flatten-then-concurrent pattern."""

        # Initialize token tracking in context.intermediate_data (leverage existing design)
        if "token_tracking" not in context.intermediate_data:
            context.intermediate_data["token_tracking"] = {
                "input_tokens": 0,
                "output_tokens": 0,
            }

        # Start progress tracking if available
        progress_tracker = getattr(context, "progress_tracker", None)
        progress_task = None
        if progress_tracker:
            total_prompts = sum(len(b.prompts) for b in batches)
            progress_task = progress_tracker.start_stage(
                f"{self.name}: {context.total_rows} rows",
                total_rows=total_prompts,
            )
            # Store for access in concurrent loop
            self._current_progress_task = progress_task

        # Flatten all prompts from all batches
        all_prompts, batch_map = self._flatten_batches(batches)

        # Calculate total rows (handle both aggregated and non-aggregated batches)
        total_rows = 0
        for batch in batches:
            if not batch.metadata:
                continue
            if (
                batch.metadata
                and batch.metadata[0].custom
                and batch.metadata[0].custom.get("is_batch")
            ):
                # Aggregated batch: use batch_size from metadata
                total_rows += batch.metadata[0].custom.get(
                    "batch_size", len(batch.metadata)
                )
            else:
                # Non-aggregated batch: count metadata entries
                total_rows += len(batch.metadata)

        self.logger.info(
            f"Processing {total_rows:,} rows in {len(batches)} API calls "
            f"({self.concurrency} concurrent)"
        )

        # Step 2: Process ALL prompts concurrently (ignore batch boundaries)
        all_responses = self._process_all_prompts_concurrent(
            all_prompts, context, batches
        )

        # Step 3: Reconstruct batches from flat responses
        response_batches = self._reconstruct_batches(all_responses, batches, batch_map)

        # Notify progress after processing
        if hasattr(context, "notify_progress"):
            context.notify_progress()

        # Finish progress tracking
        if progress_tracker and progress_task:
            progress_tracker.finish(progress_task)

        return response_batches

    def _flatten_batches(
        self, batches: list[PromptBatch]
    ) -> tuple[list[tuple], list[tuple]]:
        """Flatten all prompts from all batches, tracking batch membership.

        Args:
            batches: List of PromptBatch objects

        Returns:
            Tuple of (all_prompts, batch_map) where:
            - all_prompts: List of (prompt, metadata, batch_id) tuples
            - batch_map: List of (batch_idx, prompt_idx_in_batch) tuples
        """
        all_prompts = []
        batch_map = []  # Maps flat index to (batch_idx, prompt_idx_in_batch)

        for batch_idx, batch in enumerate(batches):
            for prompt_idx, (prompt, metadata) in enumerate(
                zip(batch.prompts, batch.metadata, strict=False)
            ):
                all_prompts.append((prompt, metadata, batch.batch_id))
                batch_map.append((batch_idx, prompt_idx))

        return all_prompts, batch_map

    def _process_all_prompts_concurrent(
        self,
        all_prompts: list[tuple],
        context: Any,
        original_batches: list[PromptBatch] = None,
    ) -> list[Any]:
        """Process all prompts concurrently, ignoring batch boundaries.

        Args:
            all_prompts: List of (prompt, metadata, batch_id) tuples
            context: Execution context

        Returns:
            List of LLMResponse objects in same order as all_prompts
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.concurrency
        ) as executor:
            futures = [
                executor.submit(
                    self._invoke_with_retry_and_ratelimit,
                    prompt,
                    metadata,
                    context,
                    idx,
                )
                for idx, (prompt, metadata, _) in enumerate(all_prompts)
            ]

            # Collect results with progress tracking
            responses = []
            progress_tracker = getattr(context, "progress_tracker", None)
            progress_task = getattr(self, "_current_progress_task", None)

            for idx, future in enumerate(futures):
                # Progress logging every 25% (only for large batches)
                if len(futures) > 20 and (idx + 1) % max(1, len(futures) // 4) == 0:
                    progress = ((idx + 1) / len(futures)) * 100
                    self.logger.info(
                        f"API calls: {progress:.0f}% complete ({idx + 1}/{len(futures)})"
                    )

                try:
                    response = future.result()
                    responses.append(response)

                    # Update progress tracker
                    if progress_tracker and progress_task:
                        progress_tracker.update(
                            progress_task, advance=1, cost=response.cost
                        )

                    # Update context with actual row count
                    # For aggregated batches, each prompt represents multiple rows
                    if context:
                        # Get the prompt metadata to check if it's an aggregated batch
                        prompt_tuple = all_prompts[idx]
                        _, metadata, _ = prompt_tuple

                        # Check if this is an aggregated batch
                        if metadata.custom and metadata.custom.get("is_batch"):
                            # Aggregated: count all rows in the batch
                            batch_size = metadata.custom.get("batch_size", 1)

                            # For first batch, start from row_index in metadata
                            # For subsequent batches, increment from last position
                            if idx == 0:
                                # First batch: set to last row index in this batch
                                first_row_idx = metadata.row_index
                                context.update_row(first_row_idx + batch_size - 1)
                            else:
                                # Subsequent batches: increment by batch_size
                                context.update_row(
                                    context.last_processed_row + batch_size
                                )
                        else:
                            # Non-aggregated: count 1 row
                            context.update_row(context.last_processed_row + 1)

                        if hasattr(response, "cost") and hasattr(response, "tokens_in"):
                            context.add_cost(
                                response.cost, response.tokens_in + response.tokens_out
                            )
                            # Track input/output tokens separately
                            context.intermediate_data["token_tracking"][
                                "input_tokens"
                            ] += response.tokens_in
                            context.intermediate_data["token_tracking"][
                                "output_tokens"
                            ] += response.tokens_out

                except Exception as e:
                    # Handle errors using existing error policy
                    decision = self.error_handler.handle_error(
                        e,
                        {
                            "stage": self.name,
                            "prompt_index": idx,
                            "total_prompts": len(all_prompts),
                        },
                    )

                    if decision.action == ErrorAction.SKIP:
                        # Create placeholder response
                        placeholder = LLMResponse(
                            text="[SKIPPED]",
                            tokens_in=0,
                            tokens_out=0,
                            model=self.llm_client.model,
                            cost=Decimal("0.0"),
                            latency_ms=0.0,
                            metadata={"error": str(e), "action": "skipped"},
                        )
                        responses.append(placeholder)
                    elif decision.action == ErrorAction.USE_DEFAULT:
                        responses.append(decision.default_value)
                    elif decision.action == ErrorAction.FAIL:
                        # Cancel remaining futures
                        for remaining_future in futures[idx + 1 :]:
                            remaining_future.cancel()
                        raise

            return responses

    def _reconstruct_batches(
        self,
        all_responses: list[Any],
        original_batches: list[PromptBatch],
        batch_map: list[tuple],
    ) -> list[ResponseBatch]:
        """Reconstruct batches from flat responses.

        Args:
            all_responses: Flat list of LLMResponse objects
            original_batches: Original PromptBatch objects
            batch_map: List of (batch_idx, prompt_idx_in_batch) tuples

        Returns:
            List of ResponseBatch objects in original batch order
        """
        # Group responses by batch
        batch_responses = {i: [] for i in range(len(original_batches))}

        for response_idx, (batch_idx, prompt_idx_in_batch) in enumerate(batch_map):
            batch_responses[batch_idx].append(
                (prompt_idx_in_batch, all_responses[response_idx])
            )

        # Create ResponseBatch objects in original order
        response_batches = []
        for batch_idx, original_batch in enumerate(original_batches):
            # Sort by prompt index to maintain order
            sorted_responses = sorted(batch_responses[batch_idx], key=lambda x: x[0])
            responses = [r for _, r in sorted_responses]

            # Calculate batch metrics
            total_tokens = sum(r.tokens_in + r.tokens_out for r in responses)
            total_cost = sum(r.cost for r in responses)
            latencies = [r.latency_ms for r in responses]

            response_batch = ResponseBatch(
                responses=[r.text for r in responses],
                metadata=original_batch.metadata,
                tokens_used=total_tokens,
                cost=total_cost,
                batch_id=original_batch.batch_id,
                latencies_ms=latencies,
            )
            response_batches.append(response_batch)

        return response_batches

    def _classify_error(self, error: Exception) -> Exception:
        """
        Classify error as retryable or non-retryable using LlamaIndex exceptions.

        Leverages LlamaIndex's native exception types to determine if an error
        is fatal (non-retryable) or transient (retryable).

        Args:
            error: The exception to classify

        Returns:
            Classified exception (NonRetryableError subclass or RetryableError)
        """
        error_str = str(error).lower()

        # Check for LlamaIndex/provider-specific exceptions first
        # Note: OpenAI exceptions cover most providers (Groq, Azure, Together.AI, vLLM, Ollama)
        # because they use OpenAI-compatible APIs. Anthropic has its own exception types.
        # Import here to avoid circular dependencies and handle missing providers.
        try:
            from openai import AuthenticationError as OpenAIAuthError
            from openai import BadRequestError as OpenAIBadRequestError

            if isinstance(error, OpenAIAuthError):
                return InvalidAPIKeyError(f"OpenAI authentication failed: {error}")
            if isinstance(error, OpenAIBadRequestError):
                # Check if it's a model error
                if "model" in error_str or "decommissioned" in error_str:
                    return ModelNotFoundError(f"OpenAI model error: {error}")
        except ImportError:
            pass

        try:
            from anthropic import AuthenticationError as AnthropicAuthError
            from anthropic import BadRequestError as AnthropicBadRequestError

            if isinstance(error, AnthropicAuthError):
                return InvalidAPIKeyError(f"Anthropic authentication failed: {error}")
            if isinstance(error, AnthropicBadRequestError):
                if "model" in error_str:
                    return ModelNotFoundError(f"Anthropic model error: {error}")
        except ImportError:
            pass

        # Fallback to pattern matching for other providers or generic errors
        # Model errors (decommissioned, not found)
        model_patterns = [
            "model",
            "decommissioned",
            "not found",
            "does not exist",
            "invalid model",
            "unknown model",
            "model_not_found",
        ]
        if any(p in error_str for p in model_patterns):
            return ModelNotFoundError(f"Model error: {error}")

        # Authentication errors
        auth_patterns = [
            "invalid api key",
            "invalid_api_key",
            "authentication failed",
            "401",
            "403",
            "unauthorized",
            "invalid credentials",
            "api key not found",
            "permission denied",
        ]
        if any(p in error_str for p in auth_patterns):
            return InvalidAPIKeyError(f"Authentication error: {error}")

        # Quota/billing errors (not rate limit)
        quota_patterns = [
            "quota exceeded",
            "insufficient_quota",
            "billing",
            "credits exhausted",
            "account suspended",
            "payment required",
        ]
        if any(p in error_str for p in quota_patterns):
            return QuotaExceededError(f"Quota error: {error}")

        # Rate limit (retryable)
        if "rate" in error_str or "429" in error_str:
            return RateLimitError(str(error))

        # Network errors (retryable)
        if (
            "network" in error_str
            or "timeout" in error_str
            or "connection" in error_str
        ):
            return NetworkError(str(error))

        # Default: return original error (will be retried conservatively)
        return error

    def _invoke_with_retry_and_ratelimit(
        self,
        prompt: str,
        row_metadata: Any = None,
        context: Any = None,
        row_index: int = 0,
    ) -> Any:
        """Invoke LLM with rate limiting and retries."""
        time.time()

        # Extract system message from row metadata
        system_message = None
        if row_metadata and hasattr(row_metadata, "custom") and row_metadata.custom:
            system_message = row_metadata.custom.get("system_message")

        def _invoke() -> Any:
            # Acquire rate limit token
            if self.rate_limiter:
                self.rate_limiter.acquire()

            # Invoke LLM with error classification
            try:
                # Pass system_message as kwarg for caching optimization
                return self.llm_client.invoke(prompt, system_message=system_message)
            except Exception as e:
                # Classify error to determine if retryable
                classified = self._classify_error(e)
                raise classified

        # Execute with retry handler (respects NonRetryableError)
        return self.retry_handler.execute(_invoke)

        # LlamaIndex automatically instruments the LLM call above!
        # No need to manually emit events - LlamaIndex's handlers capture:
        # - Prompt and completion
        # - Token usage and costs
        # - Latency metrics
        # - Model information

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

        # Estimate tokens for all prompts
        for batch in batches:
            for prompt in batch.prompts:
                input_tokens = self.llm_client.estimate_tokens(prompt)
                total_input_tokens += input_tokens

                # Assume average output length (can be made configurable)
                estimated_output = int(input_tokens * 0.5)
                total_output_tokens += estimated_output

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
