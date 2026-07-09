"""Bridge from pipeline execution progress to the RunRegistry.

The MCP ``ondine_status`` tool needs live rows-done and cost-so-far for a
RUNNING job. That data flows through :class:`ExecutionContext` during
execution; this module is the :class:`ExecutionObserver` that copies the
relevant counters onto the durable :class:`RunRegistry` row so a status poller
(even in another process) sees them.

Design (information hiding):

* The observer knows two things: which ``run_id`` it serves, and which
  registry to write to. It does not know what a "stage" is, how cost is
  accumulated, or what transport reads it back. It is a pure forwarder.
* Writes are best-effort and throttled: progress churns on every batch, and a
  registry write per batch would be needlessly expensive and would spam the
  WAL. A minimum interval between writes collapses the burst while keeping
  status fresh enough for human-scale polling.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ondine.orchestration.observers import ExecutionObserver

if TYPE_CHECKING:
    from ondine.orchestration.execution_context import ExecutionContext
    from ondine.orchestration.run_registry import RunRegistry
    from ondine.stages.pipeline_stage import PipelineStage


class RegistryProgressObserver(ExecutionObserver):
    """Forward live execution progress to a RunRegistry row.

    Attached to a :class:`~ondine.api.pipeline.Pipeline` via
    ``pipeline.add_observer(...)`` before ``execute(run_id=..., registry=...)``.
    On every progress update it writes ``processed_rows`` and ``cost`` to the
    registry under ``run_id`` using :meth:`RunRegistry.update_metrics`, which
    persists without changing status (status transitions are the pipeline's
    job).

    Throttling: writes are collapsed to at most one per ``min_interval``
    seconds (default 0.25s) so a fast run does not write the WAL on every row.
    """

    def __init__(
        self,
        registry: RunRegistry,
        run_id: Any,
        min_interval: float = 0.25,
    ) -> None:
        self._registry = registry
        self._run_id = run_id
        self._min_interval = min_interval
        self._last_write: float = 0.0
        self._last_rows: int = -1

    # ── ExecutionObserver interface ─────────────────────────────────

    def on_pipeline_start(self, pipeline: Any, context: ExecutionContext) -> None:
        self._flush(context)

    def on_stage_start(self, stage: PipelineStage, context: ExecutionContext) -> None:
        pass  # progress is reported via on_progress_update / on_stage_complete

    def on_stage_complete(
        self, stage: PipelineStage, context: ExecutionContext, result: Any
    ) -> None:
        self._flush(context)

    def on_stage_error(
        self,
        stage: PipelineStage,
        context: ExecutionContext,
        error: Exception,
    ) -> None:
        self._flush(context)

    def on_pipeline_complete(self, context: ExecutionContext, result: Any) -> None:
        # Final flush with the authoritative end state — bypass the throttle.
        self._write(
            processed_rows=int(getattr(context, "last_processed_row", 0)),
            total_rows=int(getattr(context, "total_rows", 0)),
            cost=context.run_progress.snapshot_cost,
            force=True,
        )

    def on_pipeline_error(self, context: ExecutionContext, error: Exception) -> None:
        self._flush(context, force=True)

    def on_progress_update(self, context: ExecutionContext) -> None:
        self._flush(context)

    # ── internals ───────────────────────────────────────────────────

    def _flush(self, context: ExecutionContext, force: bool = False) -> None:
        rows = int(getattr(context, "last_processed_row", 0))
        # Skip if rows haven't advanced since the last write (no new info).
        if not force and rows == self._last_rows:
            return
        self._write(
            processed_rows=rows,
            total_rows=int(getattr(context, "total_rows", 0)),
            cost=context.run_progress.snapshot_cost,
            force=force,
        )

    def _write(
        self,
        processed_rows: int,
        total_rows: int,
        cost: Any,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if not force and (now - self._last_write) < self._min_interval:
            return
        self._last_write = now
        self._last_rows = processed_rows
        metrics: dict[str, Any] = {"processed_rows": processed_rows, "cost": str(cost)}
        if total_rows:
            metrics["total_rows"] = total_rows
        try:
            self._registry.update_metrics(self._run_id, metrics)
        except Exception:  # noqa: BLE001
            # A registry write failure must never kill the run. The terminal
            # transition (written by Pipeline.execute) is the source of truth
            # for final state; live-progress writes are best-effort.
            pass
