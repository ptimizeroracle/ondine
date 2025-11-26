"""LLM invocation stage with concurrency and retry logic."""

import concurrent.futures
import time
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
            # Calculate total rows (handle both aggregated and non-aggregated batches)
            total_rows_for_progress = 0
            for batch in batches:
                if not batch.metadata:
                    continue
                if (
                    batch.metadata
                    and batch.metadata[0].custom
                    and batch.metadata[0].custom.get("is_batch")
                ):
                    # Aggregated batch: use batch_size from metadata
                    total_rows_for_progress += batch.metadata[0].custom.get(
                        "batch_size", len(batch.metadata)
                    )
                else:
                    # Non-aggregated batch: count metadata entries
                    total_rows_for_progress += len(batch.metadata)

            # Start progress tracking if available
            progress_tracker = getattr(context, "progress_tracker", None)
            progress_task = None
            self._deployment_map = {}  # Map ID -> Task ID for dynamic tracking
            self._hash_to_friendly_id = {}  # Map LiteLLM's hash to our friendly ID
            self._hash_to_label = {}  # Map LiteLLM's hash to display label
            self._available_names = []  # Queue of friendly names to assign

            # Initialize available names from config
            if hasattr(self.llm_client, "router") and self.llm_client.router:
                model_list = getattr(self.llm_client.router, "model_list", [])
                for i, dep in enumerate(model_list):
                    # Get friendly name and model info
                    friendly_id = dep.get("model_id", dep.get("model_name", "unknown"))

                    # Ensure ID is unique for visualization even if model_name is shared
                    # (Router requires shared model_name for load balancing)
                    unique_id = f"{friendly_id}_{i}"

                    litellm_params = dep.get("litellm_params", {})
                    model = litellm_params.get("model", "unknown")

                    # Create a nice label info object to store
                    provider = model.split("/")[0] if "/" in model else ""
                    model_short = model.split("/")[1] if "/" in model else model
                    # Removed truncation to show full model name

                    self._available_names.append(
                        {
                            "id": unique_id,
                            "label": f"{provider}/{model_short} ({friendly_id})",
                        }
                    )

            if progress_tracker:
                # Prepare deployments list for UI initialization
                deployments_for_progress = []
                for item in self._available_names:
                    deployments_for_progress.append(
                        {
                            "model_id": item["id"],
                            "label": item["label"],
                            "weight": 1.0,  # Assume equal distribution for UI
                        }
                    )

                progress_task = progress_tracker.start_stage(
                    f"{self.name}: {total_rows_for_progress:,} rows",
                    total_rows=total_rows_for_progress,
                    deployments=deployments_for_progress,
                )
                # Store for access in concurrent loop
                self._current_progress_task = progress_task
                self._progress_tracker = progress_tracker
                self._total_rows_for_progress = total_rows_for_progress

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
        # Track actual Router distribution (deployment_id -> count)
        deployment_distribution: dict[str, int] = {}

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
                try:
                    response = future.result()
                    responses.append(response)

                    # Extract ACTUAL deployment used by Router (not round-robin estimate)
                    actual_deployment_id = self._extract_deployment_from_response(
                        response
                    )

                    # Track actual distribution
                    if actual_deployment_id:
                        deployment_distribution[actual_deployment_id] = (
                            deployment_distribution.get(actual_deployment_id, 0) + 1
                        )

                    # Get batch size for this request (for aggregated batches)
                    prompt_tuple = all_prompts[idx]
                    _, metadata, _ = prompt_tuple
                    batch_size = 1
                    if metadata.custom and metadata.custom.get("is_batch"):
                        batch_size = metadata.custom.get("batch_size", 1)

                    # Removed debug logging - distribution tracked in final summary

                    # Update progress tracker (including deployment tracking)
                    if progress_tracker and progress_task:
                        # Update dynamic deployment progress
                        display_id = None
                        if actual_deployment_id:
                            # DYNAMIC MAPPING: Bind unknown hash to next available friendly name
                            # This solves the "hash mismatch" problem by assigning friendly names
                            # in order of appearance (First-Seen-First-Assigned).
                            if actual_deployment_id not in self._hash_to_friendly_id:
                                if self._available_names:
                                    # Assign next available friendly name
                                    next_friendly = self._available_names.pop(0)
                                    self._hash_to_friendly_id[actual_deployment_id] = (
                                        next_friendly["id"]
                                    )
                                    self._hash_to_label[actual_deployment_id] = (
                                        next_friendly["label"]
                                    )

                            # Now look up the friendly info
                            display_id = self._hash_to_friendly_id.get(
                                actual_deployment_id, actual_deployment_id
                            )
                            label_info = self._hash_to_label.get(
                                actual_deployment_id, ""
                            )

                            # Ensure deployment task exists
                            progress_tracker.ensure_deployment_task(
                                progress_task,
                                display_id,
                                total_rows=self._total_rows_for_progress
                                // 3,  # Estimate
                                label_info=label_info,
                            )

                        # Update progress ONCE (updates both main bar and deployment bar)
                        progress_tracker.update(
                            progress_task,
                            advance=batch_size,
                            cost=response.cost,
                            deployment_id=display_id,
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
                            # Non-aggregated: set to current row index (not increment)
                            # last_processed_row is an INDEX (0-based), not a count
                            context.update_row(metadata.row_index)

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

            # Final distribution summary (INFO level - always visible)
            if hasattr(self.llm_client, "router") and self.llm_client.router:
                self.logger.info("=" * 70)

                if deployment_distribution:
                    # We have actual deployment data
                    self.logger.info("Router Distribution Summary (ACTUAL):")
                    total_requests = sum(deployment_distribution.values())

                    # Calculate total rows processed (accounting for batch aggregation)
                    total_rows_processed = 0
                    for idx, (_, metadata, _) in enumerate(all_prompts):
                        if metadata.custom and metadata.custom.get("is_batch"):
                            total_rows_processed += metadata.custom.get("batch_size", 1)
                        else:
                            total_rows_processed += 1

                    for dep_id in sorted(deployment_distribution.keys()):
                        count = deployment_distribution[dep_id]
                        percentage = (
                            (count / total_requests) * 100 if total_requests > 0 else 0
                        )
                        # Use friendly name if available
                        display_name = getattr(self, "_hash_to_friendly_id", {}).get(
                            dep_id, dep_id
                        )
                        self.logger.info(
                            f"  • {display_name}: {count}/{total_requests} requests ({percentage:.1f}%)"
                        )

                    self.logger.info(f"Total API calls: {total_requests}")
                    self.logger.info(f"Total rows processed: {total_rows_processed}")
                else:
                    # No deployment data available
                    self.logger.info("Router Distribution Summary:")
                    self.logger.info(
                        "  ⚠️  Could not extract deployment info from LiteLLM responses"
                    )
                    self.logger.info(f"  Total API calls: {len(all_prompts)}")
                    self.logger.info(
                        "  Note: Progress bars show estimated distribution (round-robin)"
                    )

                self.logger.info("=" * 70)

            return responses

    def _extract_deployment_from_response(self, response: Any) -> str | None:
        """
        Extract actual deployment ID from LiteLLM response.

        LiteLLM Router stores the chosen deployment in response metadata.
        We check multiple possible locations for maximum compatibility.

        Args:
            response: LLMResponse object from invoke

        Returns:
            Deployment ID string or None if not available
        """
        try:
            # Method 1: Check if response has Router metadata (LLMResponse.metadata)
            if hasattr(response, "metadata") and isinstance(response.metadata, dict):
                # Check for model_id in metadata (added by some Router strategies)
                if "model_id" in response.metadata:
                    return response.metadata["model_id"]

            # Method 2: Check model field (Router sometimes sets this to deployment ID)
            if hasattr(response, "model") and response.model:
                # If router is active, model might be deployment ID
                if hasattr(self.llm_client, "router") and self.llm_client.router:
                    # Check if model matches any deployment ID
                    if (
                        hasattr(self, "_deployment_ids")
                        and response.model in self._deployment_ids
                    ):
                        return response.model

            # Method 3: Fallback - no deployment info available
            # This happens when Router is not used or deployment tracking is disabled
            return None

        except Exception:
            # Defensive: never crash on metadata extraction
            return None

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

        # Network errors (retryable) - CHECK FIRST
        if (
            "network" in error_str
            or "timeout" in error_str
            or "connection" in error_str
            or "service unavailable" in error_str
            or "503" in error_str
            or "502" in error_str
        ):
            # Try to extract failing model/provider from exception attributes
            # LiteLLM often attaches 'model' or 'llm_provider' to the exception
            provider_info = ""
            if hasattr(error, "model") and error.model and error.model != "mixed-llm":
                provider_info = f" [Provider: {error.model}]"
            elif hasattr(error, "failed_model") and error.failed_model:
                provider_info = f" [Provider: {error.failed_model}]"
            elif hasattr(error, "llm_provider") and error.llm_provider:
                provider_info = f" [Provider: {error.llm_provider}]"
            
            return NetworkError(f"{str(error)}{provider_info}")

        # Rate limit (retryable)
        if "rate" in error_str or "429" in error_str:
            return RateLimitError(str(error))

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

        # Authentication errors (non-retryable)
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

        # Fallback to pattern matching for other providers or generic errors
        # Model errors (decommissioned, not found)
        model_patterns = [
            "decommissioned",
            "not found",
            "does not exist",
            "invalid model",
            "unknown model",
            "model_not_found",
        ]
        # Only check generic "model" if it's clearly a not found error
        if any(p in error_str for p in model_patterns) or ("model" in error_str and "found" in error_str):
            # CRITICAL ROUTER LOGIC:
            # If we are using a Router, a "Model Not Found" on one provider is just a node failure.
            # We should treat it as a TRANSIENT NetworkError so the retry loop runs again.
            # The Router (simple-shuffle) will likely pick a different provider next time.
            if hasattr(self.llm_client, "router") and self.llm_client.router:
                return NetworkError(f"Router node failed (retryable): {error}")
            
            return ModelNotFoundError(f"Model error: {error}")

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
                # Use structured invoke if output_cls is configured
                if self.output_cls:
                    return self.llm_client.structured_invoke(
                        prompt, self.output_cls, system_message=system_message
                    )

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
