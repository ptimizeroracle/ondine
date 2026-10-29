"""LLM invocation stage with concurrency and retry logic."""

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ondine.adapters.llm_client import LLMClient
from ondine.core.error_handler import ErrorAction, ErrorHandler
from ondine.core.exceptions import (
    InvalidAPIKeyError,
    ModelNotFoundError,
    QuotaExceededError,
)
from ondine.core.models import (
    SKIPPED_OUTPUT_MARKER,
    CostEstimate,
    LLMResponse,
    PromptBatch,
    ResponseBatch,
    RowMetadata,
    ValidationResult,
)
from ondine.core.specifications import ErrorPolicy
from ondine.harness.ledger import RowLedger
from ondine.orchestration.concurrency_controller import ConcurrencyController
from ondine.orchestration.deployment_tracker import DeploymentTracker
from ondine.orchestration.progress_reporter import ProgressReporter
from ondine.stages.batch_processor import BatchProcessor
from ondine.stages.pipeline_stage import PipelineStage
from ondine.utils import (
    NetworkError,
    RateLimitError,
    RetryHandler,
)
from ondine.utils.budget_controller import BudgetExceededError
from ondine.utils.rate_limiter import RateLimiterProtocol

if TYPE_CHECKING:
    from ondine.orchestration.execution_context import ExecutionContext

# What the error policy did with a call the provider never answered, as the
# stage marks it in the response's metadata ("action"). The response cache
# keeps that metadata, so a resumed run reads the same mark.
_SKIPPED = "skipped"  # no value: the cell gets SKIPPED_OUTPUT_MARKER
_SUBSTITUTED = "substituted"  # the policy's default value stands in


def skipped_by_error_policy(response: Any) -> bool:
    """True when the error policy skipped this call: its rows are already lost.

    A later stage that cannot read the skip marker must not report those rows
    a second time (see :func:`report_response`).
    """
    return _policy_action(response) == _SKIPPED


def report_response(
    context: "ExecutionContext",
    response: LLMResponse,
    metadata: RowMetadata,
    *,
    stage: str,
) -> None:
    """Tell ``context`` what one model call's response means for its rows.

    The one place that reads a response for the row ledger. It knows:

    * **which rows** the call answered: one row, or every row a mega-batch
      prompt held (``BatchProcessor.row_indices``). A batch's rows need not be
      consecutive: a resume gap or a prompt that could not be built leaves
      holes, and a range from the first row would name the wrong rows;
    * **what the error policy did**: a skipped call is a provider failure of
      every row it held; a substituted default is a provider failure too,
      although the default is delivered. Anything else is the model's answer;
    * **whose cost it is**: the call's tokens and cost go to its rows, shared
      by the context's ledger, so the rows' costs sum to the call's.

    The live stage, resume (replaying a previous attempt's cached responses)
    and the batch splitter (through :func:`skipped_by_error_policy`) all use
    it, so the three can never read a response differently.
    """
    rows = BatchProcessor.row_indices(metadata)
    action = _policy_action(response)
    if action == _SKIPPED:
        message = _policy_error(response, "skipped by error policy")
        for first_row, count in _consecutive_runs(rows):
            context.record_skipped_rows(
                first_row_index=first_row, count=count, stage=stage, message=message
            )
    elif action == _SUBSTITUTED:
        _report_substitution(
            context,
            rows,
            stage=stage,
            message=_policy_error(response, "the provider gave no answer"),
        )
    else:
        context.record_answers(
            rows,
            model=str(response.model),
            tokens=response.tokens_in + response.tokens_out,
            cost=response.cost,
        )


def _policy_action(response: Any) -> str | None:
    details = getattr(response, "metadata", None)
    if isinstance(details, dict):
        action = details.get("action")
        if action in (_SKIPPED, _SUBSTITUTED):
            return str(action)
    return None


def _policy_error(response: LLMResponse, fallback: str) -> str:
    return str(response.metadata.get("error", fallback))


def _consecutive_runs(rows: Sequence[int]) -> list[tuple[int, int]]:
    """``rows`` as ``(first, count)`` runs of consecutive rows, in their order."""
    runs: list[list[int]] = []
    for row in rows:
        if runs and row == runs[-1][0] + runs[-1][1]:
            runs[-1][1] += 1
        else:
            runs.append([row, 1])
    return [(first, count) for first, count in runs]


def _report_substitution(
    context: "ExecutionContext", rows: list[int], *, stage: str, message: str
) -> None:
    """A provider failure whose rows the error policy filled with its default.

    The ledger must not call those rows answered: the value delivered is the
    policy's, not the model's. Today's error list and loss counters never
    named them, though, and must not start to in this migration step. Those
    views count only losses reported to the ledger directly
    (``RowLedger.reported_losses``), so the failure arrives the way a
    sub-pipeline's do: in a ledger of its own, merged.
    """
    ledger = context.ledger
    substitution = RowLedger(len(rows), ledger.policy)
    substitution.record_provider_failure(
        range(len(rows)),
        1,
        stage=stage,
        message=f"{message} (the error policy delivered its default value)",
    )
    ledger.widen(max(rows) + 1)
    ledger.merge(substitution, rows)


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
        rate_limiter: RateLimiterProtocol | None = None,
        retry_handler: RetryHandler | None = None,
        error_policy: ErrorPolicy = ErrorPolicy.SKIP,
        max_retries: int = 3,
        output_cls: type[BaseModel] | None = None,
        budget_controller: Any | None = None,
        adaptive_concurrency: bool = False,
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
            budget_controller: Optional budget controller for mid-run enforcement
            adaptive_concurrency: Opt-in Gradient2-based adaptive cap.
                When False (default), behaviour is byte-identical to
                prior versions; when True, the effective in-flight
                limit shrinks on 429/RTT inflation and grows on
                saturation + near-baseline RTT, bounded above by
                ``concurrency`` and below by 1.
        """
        super().__init__("LLMInvocation")
        self.llm_client = llm_client
        self.concurrency = concurrency
        self.rate_limiter = rate_limiter
        self.retry_handler = retry_handler or RetryHandler()
        self.output_cls = output_cls
        self.budget_controller = budget_controller
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
        self._concurrency_ctrl = ConcurrencyController(
            concurrency,
            rate_limiter,
            adaptive=adaptive_concurrency,
        )
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
        except RuntimeError as exc:
            if "asyncio.run() cannot be called from a running event loop" not in str(
                exc
            ):
                raise
            # Loop already running (e.g. Jupyter)
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.run(self._process_async(batches, context))

    async def process_async(
        self, batches: list[PromptBatch], context: Any
    ) -> list[ResponseBatch]:
        """Async entry point for pipelined execution.

        Unlike process(), this does NOT call asyncio.run() — it expects
        to be called from within an already-running event loop. Use this
        when the LLM stage is part of an async pipeline where other
        coroutines (e.g. CPU prep for the next chunk) must run concurrently.

        Behavioral contract: identical output to process() for the same input.
        """
        if "token_tracking" not in context.intermediate_data:
            context.intermediate_data["token_tracking"] = {
                "input_tokens": 0,
                "output_tokens": 0,
            }
        return await self._process_async(batches, context)

    async def _process_async(
        self, batches: list[PromptBatch], context: Any
    ) -> list[ResponseBatch]:
        """Internal async implementation — shared by process() and process_async()."""
        # Initialize global connection pool
        await self.llm_client.start()

        try:
            # Setup deployment tracker from Router config
            model_list = self._get_router_model_list()
            deployment_tracker = DeploymentTracker(model_list)

            # Setup progress reporter (backed by shared RunProgressState)
            progress_tracker = getattr(context, "progress_tracker", None)
            run_progress = getattr(context, "run_progress", None)
            progress_reporter = ProgressReporter(
                progress_tracker, deployment_tracker, run_progress=run_progress
            )

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
        """Process all prompt items concurrently using asyncio.

        Admission is gated through the concurrency controller so that
        adaptive mode actually limits in-flight work. In the default
        (non-adaptive) mode, throttle() is a bounded asyncio.Semaphore
        — behaviour identical to the original implementation; in
        adaptive mode, it routes through the Gradient2 limiter and
        feeds RTT observations back for shrink/grow decisions.
        """

        async def _process_one(idx: int, item) -> LLMResponse:
            async with self._concurrency_ctrl.throttle():
                return await self._process_single_item(idx, item, context)

        if self.concurrency <= 1:
            sequential_responses: list[LLMResponse] = []
            for idx, item in enumerate(items):
                sequential_responses.append(await _process_one(idx, item))

            self._log_distribution_summary()
            return sequential_responses

        task_map = {
            asyncio.create_task(_process_one(idx, item)): idx
            for idx, item in enumerate(items)
        }
        pending = set(task_map)
        responses: list[LLMResponse | None] = [None] * len(items)

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_EXCEPTION
            )

            first_error = None
            for task in done:
                idx = task_map[task]
                exc = task.exception()
                if exc is not None and first_error is None:
                    first_error = exc
                    continue
                responses[idx] = task.result()

            if first_error is not None:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise first_error

        # Log distribution summary
        self._log_distribution_summary()

        return [response for response in responses if response is not None]

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

            # Update shared progress state + tracker rendering
            batch_size = BatchProcessor.get_batch_size(item.metadata)
            self._progress_reporter.update(
                rows_completed=batch_size,
                cost=response.cost,
                tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
                deployment_id=deployment_id,
            )

            # Update context (legacy accumulated_cost kept in sync)
            self._update_context(context, idx, item.metadata, response)

            return response

        except BudgetExceededError:
            raise
        except Exception as e:
            response = self._handle_error(e, idx, self._total_rows)

            batch_size = BatchProcessor.get_batch_size(item.metadata)
            self._progress_reporter.update(
                rows_completed=batch_size,
                cost=response.cost,
                tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
            )
            self._update_context(context, idx, item.metadata, response)

            return response

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
            # Note: admission + rate limiting are both handled by the
            # concurrency controller's throttle() wrapper in
            # _process_all_concurrent. We intentionally do NOT acquire
            # the rate limiter again here — double-acquiring on
            # retries would cause the second attempt to wait on its
            # own already-drained bucket and time out.
            try:
                # Invoke LLM
                if self.output_cls:
                    return await self.llm_client.structured_invoke_async(  # type: ignore[attr-defined,no-any-return]
                        prompt, self.output_cls, system_message=system_message
                    )
                return await self.llm_client.ainvoke(  # type: ignore[attr-defined,no-any-return]
                    prompt, system_message=system_message
                )
            except RateLimitError as rl_err:
                # Server-sourced backoff: signal the concurrency
                # controller so every concurrent caller observes the
                # 429, then re-raise so the retry handler can
                # schedule the next attempt. See
                # ConcurrencyController.on_rate_limit for semantics.
                self._concurrency_ctrl.on_rate_limit(
                    getattr(rl_err, "retry_after_s", None)
                )
                raise

        return await self.retry_handler.execute_async(_invoke)

    def _extract_deployment_id(self, response: LLMResponse) -> str | None:
        """Extract deployment ID from LLM response."""
        try:
            if hasattr(response, "metadata") and isinstance(response.metadata, dict):
                if "model_id" in response.metadata:
                    return str(response.metadata["model_id"])
            return None
        except Exception:
            return None

    def _update_context(
        self, context: Any, idx: int, metadata: Any, response: LLMResponse
    ) -> None:
        """Update execution context with response metrics."""
        if not context:
            return

        # Record which rows are done. A mega-batch carries the index of its
        # first row and covers batch_size of them.
        #
        # Completions arrive out of order — that is the point of concurrency —
        # so this reports *which* rows finished and lets the context work out
        # how far the contiguous watermark can move. It used to assign the
        # watermark directly from whichever response happened to land last,
        # which pushed it past rows that were still in flight and made resume
        # skip work that had never been done.
        batch_size = BatchProcessor.get_batch_size(metadata)
        is_batch = bool(metadata.custom and metadata.custom.get("is_batch"))
        context.complete_rows(metadata.row_index, batch_size if is_batch else 1)

        # What the response means for its rows: an answer, or a loss the
        # error policy decided. Both the async and sync error paths funnel
        # through here, so a skip is counted once, and the skipped rows are
        # named, not only counted: a run that skips 3 rows out of 5M is a
        # success by every headline number, and without the indices the
        # caller has no way to find those 3 rows. Reported before the budget
        # check below, so the answer that crosses the budget still counts.
        report_response(context, response, metadata, stage=self.name)

        # Update cost and token tracking
        if hasattr(response, "cost") and hasattr(response, "tokens_in"):
            context.add_cost(response.cost, response.tokens_in + response.tokens_out)
            context.intermediate_data["token_tracking"]["input_tokens"] += (
                response.tokens_in
            )
            context.intermediate_data["token_tracking"]["output_tokens"] += (
                response.tokens_out
            )

            # Persist completed responses incrementally so checkpoints can
            # resume without redoing already finished LLM calls. Prefer the
            # durable ``response_cache`` (SQLite, row-level atomic) when the
            # pipeline attached one; fall back to the inline-list format for
            # configurations that did not opt into a checkpoint directory.
            if context.response_cache is not None:
                from ondine.adapters.response_cache import (
                    CachedResponse,
                    CachedRowMetadata,
                )

                context.response_cache.append(
                    context.session_id,
                    metadata.row_index,
                    CachedResponse(
                        text=response.text,
                        tokens_in=response.tokens_in,
                        tokens_out=response.tokens_out,
                        model=response.model,
                        cost=response.cost,
                        latency_ms=response.latency_ms,
                        metadata=response.metadata or {},
                    ),
                    CachedRowMetadata(
                        row_index=metadata.row_index,
                        row_id=metadata.row_id,
                        custom=metadata.custom or {},
                    ),
                )
            else:
                completed_responses = context.intermediate_data.setdefault(
                    "completed_responses", []
                )
                completed_responses.append(
                    {
                        "text": response.text,
                        "tokens_in": response.tokens_in,
                        "tokens_out": response.tokens_out,
                        "model": response.model,
                        "cost": str(response.cost),
                        "latency_ms": response.latency_ms,
                        "metadata": response.metadata or {},
                        "row_metadata": {
                            "row_index": metadata.row_index,
                            "row_id": metadata.row_id,
                            "custom": metadata.custom or {},
                        },
                        # Keep in-memory reference so the disaggregator can use the
                        # fast Pydantic path instead of fragile JSON text parsing.
                        # This field is NOT serialized to disk checkpoints.
                        "_structured_result": response.structured_result,
                    }
                )

            if self.budget_controller:
                self.budget_controller.check_budget(context.accumulated_cost)

    def _handle_error(self, error: Exception, idx: int, total: int) -> LLMResponse:
        """Handle processing error according to error policy."""
        decision = self.error_handler.handle_error(
            error,
            {"stage": self.name, "prompt_index": idx, "total_prompts": total},
        )

        if decision.action == ErrorAction.SKIP:
            return LLMResponse(
                text=SKIPPED_OUTPUT_MARKER,
                tokens_in=0,
                tokens_out=0,
                model=self.llm_client.model,
                cost=Decimal("0.0"),
                latency_ms=0.0,
                metadata={"error": str(error), "action": _SKIPPED},
            )
        if decision.action == ErrorAction.USE_DEFAULT:
            default = decision.default_value
            if not isinstance(default, LLMResponse):
                return default  # type: ignore[no-any-return]
            # The default is delivered as before; the mark says it is not the
            # model's answer, for the ledger (see report_response).
            return replace(
                default,
                metadata={
                    **(default.metadata or {}),
                    "error": str(error),
                    "action": _SUBSTITUTED,
                },
            )
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
