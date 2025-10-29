"""LLM invocation stage with concurrency and retry logic."""

import concurrent.futures
import time
from decimal import Decimal
from typing import Any

from ondine.adapters.llm_client import LLMClient
from ondine.core.error_handler import ErrorAction, ErrorHandler
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
        """Execute LLM calls for all prompt batches."""
        response_batches: list[ResponseBatch] = []

        for _batch_idx, batch in enumerate(batches):
            self.logger.info(
                f"Processing batch {batch.batch_id} ({len(batch.prompts)} prompts)"
            )

            # Process batch with concurrency
            responses = self._process_batch_concurrent(batch.prompts, context)

            # Notify progress after each batch
            if hasattr(context, "notify_progress"):
                context.notify_progress()

            # Calculate batch metrics
            total_tokens = sum(r.tokens_in + r.tokens_out for r in responses)
            total_cost = sum(r.cost for r in responses)
            latencies = [r.latency_ms for r in responses]

            # Create response batch
            response_batch = ResponseBatch(
                responses=[r.text for r in responses],
                metadata=batch.metadata,
                tokens_used=total_tokens,
                cost=total_cost,
                batch_id=batch.batch_id,
                latencies_ms=latencies,
            )
            response_batches.append(response_batch)

            # Update context
            context.add_cost(total_cost, total_tokens)
            context.update_row(batch.metadata[-1].row_index if batch.metadata else 0)

        return response_batches

    def _process_batch_concurrent(self, prompts: list[str], context: Any) -> list[Any]:
        """Process prompts concurrently while maintaining order."""
        self.logger.info(
            f"Processing {len(prompts)} prompts with concurrency={self.concurrency}"
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.concurrency
        ) as executor:
            # Submit all tasks and keep them in order
            futures = [
                executor.submit(
                    self._invoke_with_retry_and_ratelimit,
                    prompt,
                    context,
                    context.last_processed_row + idx if context else idx,
                )
                for idx, prompt in enumerate(prompts)
            ]

            self.logger.info(f"Submitted {len(futures)} parallel tasks to executor")

            # Collect results in submission order
            responses = []
            for idx, future in enumerate(futures):
                # Update progress periodically
                if (idx + 1) % max(
                    1, len(futures) // 4
                ) == 0:  # Log at 25%, 50%, 75%, 100%
                    progress = ((idx + 1) / len(futures)) * 100
                    self.logger.info(
                        f"Batch progress: {idx + 1}/{len(futures)} requests completed ({progress:.1f}%)"
                    )

                try:
                    response = future.result()
                    responses.append(response)

                    # Update context with row progress and cost
                    if context:
                        context.update_row(context.last_processed_row + 1)
                        if hasattr(response, "cost") and hasattr(response, "tokens_in"):
                            context.add_cost(
                                response.cost, response.tokens_in + response.tokens_out
                            )
                except Exception as e:
                    prompt = prompts[idx]

                    # Apply error policy
                    decision = self.error_handler.handle_error(
                        e,
                        context={
                            "row_index": idx,
                            "stage": self.name,
                            "prompt": prompt[:100],
                        },
                    )

                    if decision.action == ErrorAction.SKIP:
                        # Create placeholder response for skipped row
                        from decimal import Decimal

                        from ondine.core.models import LLMResponse

                        placeholder = LLMResponse(
                            text="[SKIPPED]",
                            tokens_in=0,
                            tokens_out=0,
                            model=self.llm_client.spec.model,
                            cost=Decimal("0.0"),
                            latency_ms=0.0,
                            metadata={"error": str(e), "action": "skipped"},
                        )
                        responses.append(placeholder)
                    elif decision.action == ErrorAction.USE_DEFAULT:
                        # Use default response
                        responses.append(decision.default_value)
                    elif decision.action == ErrorAction.FAIL:
                        # Re-raise to fail pipeline
                        raise
                    # RETRY is handled by retry_handler already

        return responses

    def _invoke_with_retry_and_ratelimit(
        self, prompt: str, context: Any = None, row_index: int = 0
    ) -> Any:
        """Invoke LLM with rate limiting and retries."""
        time.time()

        def _invoke() -> Any:
            # Acquire rate limit token
            if self.rate_limiter:
                self.rate_limiter.acquire()

            # Invoke LLM
            try:
                return self.llm_client.invoke(prompt)
            except Exception as e:
                # Classify errors for retry logic
                if "rate" in str(e).lower():
                    raise RateLimitError(str(e))
                if "network" in str(e).lower() or "timeout" in str(e).lower():
                    raise NetworkError(str(e))
                raise

        # Execute with retry handler
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
