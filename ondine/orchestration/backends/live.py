"""LiveBackend — the asyncio engine behind the backend protocol.

This is the live-mode counterpart to :class:`ProviderBatchBackend`. Both
implement the same :class:`~ondine.orchestration.backends.base.ExecutionBackend`
job-lifecycle protocol (``submit`` → ``poll`` → ``collect``), so the
front and back halves of the pipeline — and the CLI's ``submit`` /
``status`` / ``collect`` commands — work identically whether a run
executes live or via a provider batch job.

Degenerate lifecycle (live-note):
    A live run makes real HTTP calls and completes in one process, so it
    has no genuine job-id-then-later-collect separation like a batch API
    does. ``LiveBackend`` expresses that as a *degenerate* lifecycle:

    * ``submit()`` runs the concurrent engine **synchronously** (it
      blocks until every response is in), flattens the results, caches
      them keyed by a synthetic ``live-*`` id, and returns the id.
    * ``poll()`` always reports ``is_terminal=True`` — the work is done.
    * ``collect()`` yields the cached :class:`~ondine.core.models.LLMResponse`
      objects without re-running anything.

    This deliberately relaxes the "submit is non-blocking" contract that
    holds for batch backends (see the protocol docstring). The relaxation
    is safe because live mode is inherently synchronous: the results
    cannot outlive the submitting process, so there is no async
    submit→collect handoff to protect. The ``live-`` prefix on the job id
    lets the CLI distinguish these in-memory-only jobs and refuse a
    cross-process ``collect`` on them (results would be gone).

Engine provenance:
    The engine wiring below (instructor-mode override, client creation,
    observer-dispatcher wiring, rate limiter, retry handler, observer
    lifecycle) is a verbatim move of the logic that previously lived
    inline in ``Pipeline._execute_stages_with_tracking`` (stage 3) and
    later in t3's ``invoke()``-shaped LiveBackend. Behaviour is
    byte-for-byte identical to the pre-extraction code path — the same
    :class:`~ondine.stages.llm_invocation_stage.LLMInvocationStage` is
    constructed with the same parameters and run through the same
    observer notifications.
"""

from __future__ import annotations

import itertools
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from ondine.adapters import create_llm_client
from ondine.orchestration.backends.base import BatchProgress
from ondine.orchestration.observers import LoggingObserver
from ondine.orchestration.progress_tracker import NoOpProgressTracker
from ondine.stages.llm_invocation_stage import LLMInvocationStage
from ondine.utils import RateLimiter, RetryHandler, get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ondine.core.models import LLMResponse, PromptBatch, ResponseBatch
    from ondine.core.specifications import LLMSpec, PipelineSpecifications
    from ondine.orchestration.execution_context import ExecutionContext

logger = get_logger(__name__)

# Process-local result cache for the degenerate lifecycle. Keyed by the
# synthetic ``live-*`` id returned from submit(); drained by collect().
# A module-level dict (not instance state) because a live "job" is fully
# contained within submit()+collect() of one backend instance — but
# keeping it at module scope matches how the CLI rebuilds a fresh
# LiveBackend per process and still (correctly) finds nothing for a
# live-* id in a separate process.
_live_results: dict[str, list[LLMResponse]] = {}

# Monotonic counter for live job ids. One space is fine: live jobs
# complete synchronously inside submit(), so there is no cross-process
# coordination to worry about.
_live_job_counter = itertools.count(1)


def _build_rate_limiter(processing_spec: Any) -> RateLimiter | None:
    """Assemble the rate limiter for a pipeline run.

    Moved unchanged from ``ondine.api.pipeline`` so the live backend
    owns every dependency of the invocation middle.
    """
    rpm = processing_spec.rate_limit_rpm
    if not rpm:
        return None

    burst = min(20, processing_spec.concurrency)
    local = RateLimiter(rpm, burst_size=burst)

    redis_url = getattr(processing_spec, "rate_limit_redis_url", None)
    if not redis_url:
        return local

    from ondine.utils.redis_rate_limiter import RedisRateLimiter

    return RedisRateLimiter(
        requests_per_minute=rpm,
        redis_url=redis_url,
        scope=processing_spec.rate_limit_scope,
        burst_size=burst,
        fallback=local,
    )


class LiveBackend:
    """ExecutionBackend that runs the concurrent asyncio engine synchronously.

    Constructed with the LLM spec plus the full pipeline context the
    live engine needs (observers, progress tracker, cache) and an
    optional budget controller — these live on the
    :class:`~ondine.orchestration.execution_context.ExecutionContext`,
    which does not exist at the point batch backends are built, so the
    live backend takes them at construction instead. The pipeline builds
    a fresh LiveBackend per run once the context exists.

    The class deliberately does NOT subclass the Protocol — structural
    conformance (``@runtime_checkable``) is the contract, mirroring
    :class:`ProviderBatchBackend` and the rest of ondine's protocols.
    """

    def __init__(
        self,
        llm_spec: LLMSpec,
        *,
        specs: PipelineSpecifications | None = None,
        context: ExecutionContext | None = None,
        budget_controller: Any | None = None,
    ) -> None:
        """Configure the backend for a live run.

        Args:
            llm_spec: Provider + model + credentials (required by the
                protocol's ``llm_spec`` property).
            specs: Full pipeline specifications. The engine reads
                concurrency / retry / rate-limit / error-policy from
                ``specs.processing`` and the instructor-mode override
                from ``specs.metadata``. Required to actually run; may
                be ``None`` only for protocol-conformance checks.
            context: Execution context carrying observers, progress
                tracker, and response cache. Required to run; may be
                ``None`` only for protocol-conformance checks.
            budget_controller: Optional per-run budget guard injected by
                the pipeline when ``processing.max_budget`` is set.
        """
        self._llm_spec = llm_spec
        self._specs = specs
        self._context = context
        self._budget_controller = budget_controller

    @property
    def llm_spec(self) -> LLMSpec:
        """The LLM configuration this backend was built with."""
        return self._llm_spec

    def submit(self, prompts: list[PromptBatch]) -> str:
        """Run the engine to completion and return a ``live-*`` job id.

        Blocks synchronously (degenerate lifecycle — see the module
        docstring). The flattened results are cached in-process keyed by
        the returned id, so a subsequent :meth:`collect` in the same
        process yields them without re-running anything.
        """
        if not prompts:
            job_id = self._next_job_id()
            _live_results[job_id] = []
            return job_id

        response_batches = self._run_engine(prompts)
        responses = _flatten_to_llm_responses(response_batches)
        job_id = self._next_job_id()
        _live_results[job_id] = responses
        return job_id

    def poll(self, provider_job_id: str) -> BatchProgress:
        """Always terminal: a live job is complete by the time submit() returns."""
        n = len(_live_results.get(provider_job_id, []))
        return BatchProgress(
            provider_job_id=provider_job_id,
            status="completed",
            total=n,
            completed=n,
            failed=0,
            cost_so_far=Decimal("0"),
            is_terminal=True,
        )

    def collect(self, provider_job_id: str) -> Iterator[LLMResponse]:
        """Yield the cached responses for a job id (no re-run).

        Yields nothing (rather than raising) for an unknown id — a caller
        may hold a stale ``live-*`` id from a previous process, in which
        case the results are simply gone.
        """
        for r in _live_results.get(provider_job_id, []):
            yield r

    # ── the engine, adapted from the inline pipeline path ────────────

    def _run_engine(
        self, batches: list[PromptBatch]
    ) -> list[ResponseBatch]:
        """Construct and run LLMInvocationStage exactly as the pipeline did.

        Mirrors ``Pipeline._execute_stages_with_tracking`` stage 3:
        instructor-mode override, client creation, observer-dispatcher
        wiring, rate limiter, retry handler, and
        concurrency/error-policy/adaptive settings are all applied
        identically, wrapped in the same observer-lifecycle
        notifications (on_stage_start / on_stage_complete / on_stage_error).
        """
        specs = self._specs
        context = self._context
        if specs is None or context is None:
            raise RuntimeError(
                "LiveBackend cannot run the engine without specs and context; "
                "they are required at construction for a real run."
            )

        # --- instructor_mode override (moved verbatim) ---
        llm_spec = specs.llm
        if specs.metadata and "instructor_mode" in specs.metadata:
            llm_spec = llm_spec.model_copy(
                update={"instructor_mode": specs.metadata["instructor_mode"]}
            )

        llm_client = create_llm_client(llm_spec)

        # Wire observer dispatcher to LLM client (for direct SDK integration).
        if context.observer_dispatcher and hasattr(
            llm_client, "set_observer_dispatcher"
        ):
            llm_client.set_observer_dispatcher(context.observer_dispatcher)

        rate_limiter = _build_rate_limiter(specs.processing)
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
            output_cls=(
                specs.metadata.get("structured_output_model")
                if specs.metadata
                else None
            ),
            budget_controller=self._budget_controller,
            adaptive_concurrency=specs.processing.adaptive_concurrency,
        )

        # Observer lifecycle around the stage — same gating the pipeline
        # used so the progress tracker, not LoggingObserver, emits the
        # stage summary when a tracker is active.
        tracker_ref = getattr(context, "_progress_tracker_ref", None)
        tracker_active = tracker_ref is not None and not isinstance(
            tracker_ref, NoOpProgressTracker
        )
        observers = getattr(context, "observers", []) or []
        for observer in observers:
            if tracker_active and isinstance(observer, LoggingObserver):
                continue
            observer.on_stage_start(llm_stage, context)

        try:
            response_batches = llm_stage.execute(batches, context)
        except Exception as e:
            for observer in observers:
                observer.on_stage_error(llm_stage, context, e)
            raise

        for observer in observers:
            if tracker_active and isinstance(observer, LoggingObserver):
                continue
            observer.on_stage_complete(llm_stage, context, response_batches)

        return response_batches

    @staticmethod
    def _next_job_id() -> str:
        return f"live-{next(_live_job_counter)}"


def _flatten_to_llm_responses(batches: list[ResponseBatch]) -> list[LLMResponse]:
    """Flatten ``list[ResponseBatch]`` into ``list[LLMResponse]``.

    ``ResponseBatch.responses`` is typed ``list[str] | list[LLMResponse]``.
    When the engine emits full :class:`LLMResponse` objects (the normal
    path) they pass through unchanged. When it emits raw strings (e.g.
    the structured-output-disabled path), a minimal :class:`LLMResponse`
    is synthesised so the collect() contract — one ``LLMResponse`` per
    row — always holds. Token/cost details are unknown at the string
    layer and default to zero; the back-half parser consumes
    ``LLMResponse.text`` regardless.
    """
    from ondine.core.models import LLMResponse

    out: list[LLMResponse] = []
    for batch in batches:
        for idx, response in enumerate(batch.responses):
            if isinstance(response, LLMResponse):
                out.append(response)
            else:
                meta = (
                    batch.metadata[idx].row_index
                    if idx < len(batch.metadata)
                    else idx
                )
                out.append(
                    LLMResponse(
                        text=response,
                        tokens_in=0,
                        tokens_out=0,
                        model="",
                        cost=Decimal("0"),
                        latency_ms=(
                            batch.latencies_ms[idx]
                            if idx < len(batch.latencies_ms)
                            else 0.0
                        ),
                        metadata={"row_index": meta},
                    )
                )
    return out


__all__ = ["LiveBackend"]
