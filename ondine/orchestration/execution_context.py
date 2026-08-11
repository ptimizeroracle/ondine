"""
Execution context for carrying runtime state between stages.

Implements Memento pattern for checkpoint serialization.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import UUID, uuid4

from ondine.core.components import PipelineComponents
from ondine.core.models import ErrorInfo, ProcessingStats

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ondine.adapters.response_cache import ResponseCache
    from ondine.observability.dispatcher import ObserverDispatcher
    from ondine.orchestration.observers import ExecutionObserver


# ---------------------------------------------------------------------------
# Shared live-progress state  (single source of truth for cost & progress)
# ---------------------------------------------------------------------------


@dataclass
class StageProgressSnapshot:
    """Per-stage progress counters, readable by any tracker or observer."""

    stage_name: str
    total_rows: int = 0
    rows_completed: int = 0
    cost: Decimal = field(default_factory=lambda: Decimal("0"))
    deployment_rows: dict[str, int] = field(default_factory=dict)
    deployment_costs: dict[str, Decimal] = field(default_factory=dict)


@dataclass
class RunProgressState:
    """Thread-safe, shared live-progress state owned by the execution context.

    Every tracker implementation (Rich, Textual, Logging) and every observer
    reads from this object instead of maintaining its own cost accumulator.
    """

    total_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    _stages: dict[str, StageProgressSnapshot] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def init_stage(self, stage_name: str, total_rows: int) -> None:
        with self._lock:
            self._stages[stage_name] = StageProgressSnapshot(
                stage_name=stage_name, total_rows=total_rows
            )

    def apply_delta(
        self,
        stage_name: str,
        rows_completed: int = 0,
        cost: Decimal | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        deployment_id: str | None = None,
    ) -> None:
        """Atomically apply a progress delta from a single LLM response."""
        with self._lock:
            snap = self._stages.get(stage_name)
            if snap is None:
                return
            snap.rows_completed += rows_completed
            if cost is not None:
                snap.cost += cost
                self.total_cost += cost
            self.total_tokens += tokens_in + tokens_out
            self.input_tokens += tokens_in
            self.output_tokens += tokens_out
            if deployment_id is not None:
                snap.deployment_rows[deployment_id] = (
                    snap.deployment_rows.get(deployment_id, 0) + rows_completed
                )
                if cost is not None:
                    snap.deployment_costs[deployment_id] = (
                        snap.deployment_costs.get(deployment_id, Decimal("0")) + cost
                    )

    def get_stage(self, stage_name: str) -> StageProgressSnapshot | None:
        with self._lock:
            snap = self._stages.get(stage_name)
            if snap is None:
                return None
            return StageProgressSnapshot(
                stage_name=snap.stage_name,
                total_rows=snap.total_rows,
                rows_completed=snap.rows_completed,
                cost=snap.cost,
                deployment_rows=dict(snap.deployment_rows),
                deployment_costs=dict(snap.deployment_costs),
            )

    @property
    def snapshot_cost(self) -> Decimal:
        with self._lock:
            return self.total_cost


@dataclass
class ExecutionContext:
    """
    Lightweight orchestration state (passed between pipeline stages).

    Scope: Runtime execution state and progress tracking
    Pattern: Memento (serializable for checkpointing)

    Cost Tracking in ExecutionContext:
    - Simple accumulation for orchestration purposes
    - Used by: Executors to track overall progress
    - NOT for: Detailed accounting (use CostTracker for that)

    Why separate from CostTracker?
    - ExecutionContext = orchestration state (stage progress, session ID, timing)
    - CostTracker = detailed accounting (per-stage breakdowns, thread-safe entries, metrics)
    - Different concerns, different use cases

    ExecutionContext is:
    - Passed between stages in the pipeline
    - Serialized for checkpointing
    - Focused on execution orchestration

    CostTracker is:
    - Used within LLMInvocationStage for detailed tracking
    - Thread-safe for concurrent operations
    - Focused on cost reporting and analytics

    See Also:
    - CostTracker: For detailed cost accounting with breakdowns
    - docs/TECHNICAL_REFERENCE.md: Cost tracking architecture

    Carries shared state between stages and tracks progress.
    Immutable for most fields to prevent accidental modification.
    """

    session_id: UUID = field(default_factory=uuid4)
    pipeline_id: UUID = field(default_factory=uuid4)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    # Progress tracking
    current_stage_index: int = 0
    current_stage: str = ""

    # Highest row index such that *every* row up to and including it is done.
    #
    # A contiguous watermark, not "the last row that happened to finish".
    # Resume treats every index at or below this as complete and drops it from
    # the work, so a watermark that ran ahead of a gap deleted rows that had
    # never been processed: with concurrency 4, a run that died on row 1 while
    # rows 2-7 completed resumed with a watermark of 7 and quietly processed
    # nothing. See complete_rows().
    #
    # -1 means "no row is done yet", which 0 could not express — 0 also means
    # "row 0 is done". Checkpoints written before this field existed default to
    # 0 on load, keeping their old (slightly conservative) meaning.
    last_processed_row: int = -1

    # Rows finished ahead of the watermark, waiting for the gap below them to
    # close. Bounded by how far out of order completions can run — the
    # concurrency limit — not by the size of the dataset.
    pending_completions: set[int] = field(
        default_factory=set, repr=False, compare=False
    )

    total_rows: int = 0

    # Cost tracking
    accumulated_cost: Decimal = field(default_factory=lambda: Decimal("0.0"))
    accumulated_tokens: int = 0

    # Intermediate data storage
    intermediate_data: dict[str, Any] = field(default_factory=dict)

    # Statistics
    failed_rows: int = 0
    skipped_rows: int = 0

    # Which rows did not survive, and why.
    #
    # The counters above say *how many* rows were lost; without this list the
    # only way to learn *which* ones is to scan the output frame for the
    # ``[SKIPPED]`` marker — string-matching a sentinel, from user code. That
    # left ``ExecutionResult.errors`` documented but permanently empty.
    #
    # Bounded on purpose: a 5M-row run where the provider is down would
    # otherwise hold 5M ErrorInfo objects to say one thing. The count in
    # ``skipped_rows`` stays exact regardless of how many entries were kept.
    row_errors: list[ErrorInfo] = field(default_factory=list, repr=False)

    #: Entries kept before ``row_errors`` stops growing. Enough to identify a
    #: pattern in the failures; small enough to never be the reason a run runs
    #: out of memory.
    MAX_RECORDED_ROW_ERRORS: ClassVar[int] = 1000

    # Observers for progress notifications (backward compatibility)
    observers: list[ExecutionObserver] = field(default_factory=list)

    # New observability system
    observer_dispatcher: ObserverDispatcher | None = None

    # Shared live-progress state (single source of truth for cost/progress)
    run_progress: RunProgressState = field(default_factory=RunProgressState)

    # Distributed tracing
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid4()))

    # Progress tracker (set by pipeline, not serialized)
    _progress_tracker_ref: Any = field(default=None, repr=False, compare=False)
    progress_tracker: Any = field(default=None, repr=False, compare=False)

    # Durable LLM-response cache for crash-safe resume.
    # Set by the pipeline; NOT serialized into checkpoint JSON — the
    # cache is its own on-disk artifact (a SQLite file next to the
    # checkpoint) and the pipeline reattaches it by path on resume.
    response_cache: ResponseCache | None = field(
        default=None, repr=False, compare=False
    )

    # Live collaborators the caller injected (client, stores, parser, schema).
    # Runtime objects like the cache above: set by the pipeline, never
    # serialized into checkpoint JSON. They live here rather than in
    # PipelineSpecifications.metadata because specifications are configuration
    # and must stay dumpable — see ondine/core/components.py (#230, #232).
    # Defaults to an empty set rather than None: a context built outside the
    # pipeline (tests, backends constructing their own) must still be safe to
    # read from, without every caller guarding for None.
    components: PipelineComponents = field(
        default_factory=PipelineComponents, repr=False, compare=False
    )

    def update_stage(self, stage_index: int) -> None:
        """Update current stage."""
        self.current_stage_index = stage_index

    def update_row(self, row_index: int) -> None:
        """Force the watermark to `row_index`.

        For callers that process rows strictly in order and already know the
        contiguous position — the streaming executor counting whole chunks.
        Anything completing rows out of order must use complete_rows(), which
        will not advance the watermark past a gap.
        """
        self.last_processed_row = row_index

    def complete_rows(self, first_row_index: int, count: int = 1) -> None:
        """Mark `count` rows from `first_row_index` finished.

        Advances ``last_processed_row`` only across rows with no gap beneath
        them, so a crash leaves a watermark that resume can trust: everything
        at or below it really was processed.

        Out-of-order completions are held in ``pending_completions`` until the
        rows below them arrive, then absorbed in one pass. Rows completing
        twice (a retry re-reporting) are idempotent.
        """
        for offset in range(count):
            self.pending_completions.add(first_row_index + offset)

        watermark = self.last_processed_row
        while watermark + 1 in self.pending_completions:
            watermark += 1
            self.pending_completions.discard(watermark)
        self.last_processed_row = watermark

    def record_skipped_rows(
        self,
        first_row_index: int,
        count: int,
        stage: str,
        message: str,
    ) -> None:
        """Record that `count` rows starting at `first_row_index` were skipped.

        Args:
            first_row_index: Index of the first lost row, in the input frame.
            count: How many consecutive rows were lost (1 for a single row).
            stage: Stage that gave up on them.
            message: Provider or policy explanation, kept verbatim.
        """
        self.skipped_rows += count
        self._record_row_errors(
            range(first_row_index, first_row_index + count),
            stage=stage,
            message=message,
            error_type="skipped",
        )

    def record_failed_rows(
        self,
        row_indices: Iterable[int],
        stage: str,
        message: str,
    ) -> None:
        """Record rows whose output could not be produced.

        Distinct from a skip: a skip is the error policy deciding to move on,
        while this is work that came back unusable — a batch response the
        parser could not read, a provider that returned nothing. Both lose
        rows, and both used to be invisible in the metrics.

        Takes explicit indices rather than a range: a batch's rows are whatever
        the aggregator put in it, not necessarily consecutive.
        """
        indices = list(row_indices)
        self.failed_rows += len(indices)
        self._record_row_errors(
            indices, stage=stage, message=message, error_type="parse_failure"
        )

    def _record_row_errors(
        self,
        row_indices: Iterable[int],
        *,
        stage: str,
        message: str,
        error_type: str,
    ) -> None:
        """Append one ErrorInfo per lost row, up to the recording limit.

        Keeps the counters and ``row_errors`` in step so the count and the
        detail can never disagree — a caller re-processing the losers needs
        indices, not a batch id.
        """
        for row_index in row_indices:
            if len(self.row_errors) >= self.MAX_RECORDED_ROW_ERRORS:
                return
            self.row_errors.append(
                ErrorInfo(
                    row_index=row_index,
                    stage=stage,
                    error_type=error_type,
                    message=message,
                    retryable=False,
                )
            )

    def add_cost(self, cost: Decimal, tokens: int) -> None:
        """Add cost and token usage.

        Updates both the legacy ``accumulated_cost`` field and the shared
        ``run_progress`` state so that all consumers (trackers, observers,
        final report) read consistent numbers.
        """
        self.accumulated_cost += cost
        self.accumulated_tokens += tokens

    def notify_progress(self) -> None:
        """Notify all observers of progress update."""
        for observer in self.observers:
            try:
                observer.on_progress_update(self)
            except Exception:  # nosec B110
                # Silently ignore observer errors to not break pipeline
                # Observers are non-critical, pipeline should continue even if they fail
                pass

    def get_progress(self) -> float:
        """Get completion percentage."""
        if self.total_rows == 0:
            return 0.0
        return (self.last_processed_row / self.total_rows) * 100

    def get_stats(self) -> ProcessingStats:
        """Get processing statistics."""
        duration = (
            (datetime.now() - self.start_time).total_seconds()
            if self.end_time is None
            else (self.end_time - self.start_time).total_seconds()
        )

        # last_processed_row is 0-based index, so add 1 for count
        actual_processed = (
            self.last_processed_row + 1 if self.last_processed_row >= 0 else 0
        )

        rows_per_second = actual_processed / duration if duration > 0 else 0.0

        return ProcessingStats(
            total_rows=self.total_rows,
            processed_rows=actual_processed,
            failed_rows=self.failed_rows,
            skipped_rows=self.skipped_rows,
            rows_per_second=rows_per_second,
            total_duration_seconds=duration,
        )

    def to_checkpoint(self) -> dict[str, Any]:
        """
        Serialize to checkpoint dictionary (Memento pattern).

        Returns:
            Dictionary representation for persistence
        """
        return {
            "session_id": str(self.session_id),
            "pipeline_id": str(self.pipeline_id),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "current_stage_index": self.current_stage_index,
            "last_processed_row": self.last_processed_row,
            "total_rows": self.total_rows,
            "accumulated_cost": str(self.accumulated_cost),
            "accumulated_tokens": self.accumulated_tokens,
            "intermediate_data": self.intermediate_data,
            "failed_rows": self.failed_rows,
            "skipped_rows": self.skipped_rows,
        }

    @classmethod
    def from_checkpoint(cls, data: dict[str, Any]) -> ExecutionContext:
        """
        Deserialize from checkpoint dictionary.

        Args:
            data: Checkpoint data

        Returns:
            Restored ExecutionContext
        """
        return cls(
            session_id=UUID(data["session_id"]),
            pipeline_id=UUID(data["pipeline_id"]),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=(
                datetime.fromisoformat(data["end_time"])
                if data.get("end_time")
                else None
            ),
            current_stage_index=data["current_stage_index"],
            last_processed_row=data["last_processed_row"],
            total_rows=data["total_rows"],
            accumulated_cost=Decimal(data["accumulated_cost"]),
            accumulated_tokens=data["accumulated_tokens"],
            intermediate_data=data.get("intermediate_data", {}),
            failed_rows=data.get("failed_rows", 0),
            skipped_rows=data.get("skipped_rows", 0),
        )

    # Aliases for backward compatibility
    def to_dict(self) -> dict[str, Any]:
        """Alias for to_checkpoint()."""
        return self.to_checkpoint()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionContext:
        """Alias for from_checkpoint()."""
        return cls.from_checkpoint(data)
