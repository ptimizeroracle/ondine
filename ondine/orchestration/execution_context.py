"""
Execution context for carrying runtime state between stages.

Implements Memento pattern for checkpoint serialization.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import UUID, uuid4

from ondine.core.components import PipelineComponents
from ondine.core.models import ErrorInfo, ProcessingStats
from ondine.harness.contract import build_contract
from ondine.harness.ledger import LedgerPolicy, ReportedLosses, RowLedger

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ondine.adapters.response_cache import ResponseCache
    from ondine.harness.contract import OutputContract
    from ondine.observability.dispatcher import ObserverDispatcher
    from ondine.orchestration.observers import ExecutionObserver


# ---------------------------------------------------------------------------
# Row fate: the context's view of its row ledger
# ---------------------------------------------------------------------------

#: The policy of a ledger nobody opened: a context driven outside a pipeline
#: has no contract, so it runs under the unchecked one, as every run does
#: until contracts are wired.
_UNCHECKED_POLICY = LedgerPolicy.from_contract(build_contract(["output"]))


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


def _warn_counter_assignment(name: str) -> None:
    warnings.warn(
        f"Assigning ExecutionContext.{name} is deprecated: the row ledger counts "
        f"losses. Report them with record_failed_rows or record_skipped_rows.",
        DeprecationWarning,
        stacklevel=3,
    )


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

    # Rows a previous attempt lost, carried over by from_checkpoint: this
    # run's ledger never saw those rows, so the failed_rows and skipped_rows
    # counters add these to the losses reported to it.
    _carried_failed_rows: int = field(default=0, init=False, repr=False)
    _carried_skipped_rows: int = field(default=0, init=False, repr=False)

    # The single record of what happened to every row (ondine.harness.ledger).
    # The stages report losses and answers through the record_* methods below;
    # failed_rows, skipped_rows and row_errors are views of it. The pipeline
    # opens it once the input's size is known (open_ledger); a context driven
    # outside a pipeline gets one on its first report. Not serialized: a
    # checkpoint keeps the counters, and resume rebuilds answered rows from
    # the response cache.
    _ledger: RowLedger | None = field(
        default=None, init=False, repr=False, compare=False
    )

    #: Entries kept in ``row_errors``. Enough to identify a pattern in the
    #: failures; small enough to never be the reason a run runs out of memory.
    #: The counters stay exact regardless.
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

    # -- row fate: events go to the ledger, the old surface reads it ---------

    def open_ledger(self, contract: OutputContract) -> RowLedger:
        """Start this run's ledger over ``total_rows`` rows, under ``contract``.

        Called by the pipeline once the input is loaded, so every row exists,
        pending, before any stage reports on it. If something reported before
        that (a context created earlier and handed in), its ledger is kept and
        widened: its events must not be lost. It must then already run under
        the contract's policy, since statuses already derived cannot be
        re-decided under another.

        Raises:
            RuntimeError: a ledger with a different policy already holds
                reports.
        """
        policy = LedgerPolicy.from_contract(contract)
        if self._ledger is None:
            self._ledger = RowLedger(self.total_rows, policy)
        elif self._ledger.policy != policy:
            raise RuntimeError(
                "rows were reported to this context before its ledger was "
                "opened, under a different policy than the contract's"
            )
        else:
            self._ledger.widen(self.total_rows)
        return self._ledger

    @property
    def ledger(self) -> RowLedger:
        """The ledger every row event of this run goes to.

        Opened by the pipeline (:meth:`open_ledger`). A context driven outside
        a pipeline gets an unchecked one sized ``total_rows`` on first use.
        """
        if self._ledger is None:
            self._ledger = RowLedger(self.total_rows, _UNCHECKED_POLICY)
        return self._ledger

    @property
    def failed_rows(self) -> int:
        """Rows lost to a stage failure or an unreadable answer, this run and before.

        Read only: a loss is reported (:meth:`record_failed_rows`), never
        assigned. Includes what a checkpoint carried over.
        """
        return self._carried_failed_rows + self._reported_losses(0).failed_rows

    @failed_rows.setter
    def failed_rows(self, value: int) -> None:
        # Deprecated write path, kept so custom stages that assigned the
        # counter keep working. Report losses with record_failed_rows instead.
        _warn_counter_assignment("failed_rows")
        self._carried_failed_rows = value - self._reported_losses(0).failed_rows

    @property
    def skipped_rows(self) -> int:
        """Rows lost to a provider failure the error policy gave up on.

        Read only, like :attr:`failed_rows`, and likewise includes what a
        checkpoint carried over.
        """
        return self._carried_skipped_rows + self._reported_losses(0).skipped_rows

    @skipped_rows.setter
    def skipped_rows(self, value: int) -> None:
        # Deprecated write path; see failed_rows.
        _warn_counter_assignment("skipped_rows")
        self._carried_skipped_rows = value - self._reported_losses(0).skipped_rows

    @property
    def row_errors(self) -> list[ErrorInfo]:
        """Which rows did not survive, and why: one entry per lost row.

        The counters say *how many* rows were lost; without this list the only
        way to learn *which* ones is to scan the output frame for the
        ``[SKIPPED]`` marker. Bounded by ``MAX_RECORDED_ROW_ERRORS``: a 5M-row
        run where the provider is down would otherwise hold 5M entries to say
        one thing.

        A view of the ledger (:meth:`RowLedger.reported_losses`), in report
        order, each lost row once under its first cause — exactly the list
        this context used to keep itself. A fresh list on every read.
        """
        return list(self._reported_losses(self.MAX_RECORDED_ROW_ERRORS).errors)

    def record_answers(
        self,
        row_indices: Iterable[int],
        *,
        model: str,
        tokens: int,
        cost: Decimal,
    ) -> None:
        """Record one model answer covering ``row_indices`` (one row, or a batch).

        Without this a row that answered would look, to the ledger, like a row
        the run never reached. See :meth:`RowLedger.record_batch_answer` for
        how a batch's cost is shared.
        """
        rows = list(row_indices)
        self._ledger_covering(rows).record_batch_answer(
            rows, 1, model=model, tokens=tokens, cost=cost
        )

    def record_skipped_rows(
        self,
        first_row_index: int,
        count: int,
        stage: str,
        message: str,
    ) -> None:
        """Record that `count` rows starting at `first_row_index` were skipped.

        A provider failure in the ledger: the error policy gave up on a call
        the provider did not answer. A row is lost once; a later report about
        the same row does not count it again.

        Args:
            first_row_index: Index of the first lost row, in the input frame.
            count: How many consecutive rows were lost (1 for a single row).
            stage: Stage that gave up on them.
            message: Provider or policy explanation, kept verbatim.
        """
        rows = list(range(first_row_index, first_row_index + count))
        self._ledger_covering(rows).record_provider_failure(
            rows, 1, stage=stage, message=message
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
        parser could not read, a prompt that could not be built. A stage
        failure in the ledger. Both lose rows, and both used to be invisible
        in the metrics.

        Takes explicit indices rather than a range: a batch's rows are whatever
        the aggregator put in it, not necessarily consecutive.
        """
        rows = list(row_indices)
        self._ledger_covering(rows).record_stage_failure(
            rows, stage=stage, message=message
        )

    def _ledger_covering(self, rows: Sequence[int]) -> RowLedger:
        """The ledger, widened if a report names a row past its end.

        Only a context driven outside a pipeline needs this: it may report
        rows without ever being told the input's size.
        """
        ledger = self.ledger
        highest = max((row for row in rows if isinstance(row, int)), default=-1)
        if highest >= ledger.row_count:
            ledger.widen(highest + 1)
        return ledger

    def _reported_losses(self, limit: int) -> ReportedLosses:
        if self._ledger is None:
            return ReportedLosses(errors=(), skipped_rows=0, failed_rows=0)
        return self._ledger.reported_losses(limit)

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
        context = cls(
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
        )
        context._carried_failed_rows = data.get("failed_rows", 0)
        context._carried_skipped_rows = data.get("skipped_rows", 0)
        return context

    # Aliases for backward compatibility
    def to_dict(self) -> dict[str, Any]:
        """Alias for to_checkpoint()."""
        return self.to_checkpoint()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionContext:
        """Alias for from_checkpoint()."""
        return cls.from_checkpoint(data)
