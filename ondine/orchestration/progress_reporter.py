"""
Progress reporter for pipeline stage execution.

Provides a clean interface for reporting progress to the UI tracker,
handling deployment-aware updates and lifecycle management.

All cost / progress deltas flow through :class:`RunProgressState` first
(the single source of truth) and are then forwarded to the visual tracker
for rendering.
"""

from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ondine.orchestration.deployment_tracker import DeploymentTracker
    from ondine.orchestration.execution_context import RunProgressState
    from ondine.orchestration.progress_tracker import ProgressTracker


class ProgressReporter:
    """
    Reports progress to the UI tracker with deployment awareness.

    Encapsulates progress tracking logic that was previously scattered
    throughout LLMInvocationStage, providing a clean lifecycle:

    1. start() - Initialize progress bar with total rows
    2. update() - Apply delta to shared state, then refresh tracker
    3. finish() - Mark stage complete

    Example:
        reporter = ProgressReporter(context.progress_tracker,
                                    run_progress=context.run_progress)
        reporter.start("LLMInvocation", total_rows=1000, deployments=[...])

        for response in process_items():
            reporter.update(
                rows_completed=1,
                cost=response.cost,
                deployment_id=response.deployment_id
            )

        reporter.finish()
    """

    def __init__(
        self,
        tracker: "ProgressTracker | None",
        deployment_tracker: "DeploymentTracker | None" = None,
        run_progress: "RunProgressState | None" = None,
    ):
        self._tracker = tracker
        self._deployment_tracker = deployment_tracker
        self._run_progress = run_progress
        self._task_id: Any = None
        self._stage_name: str = ""
        self._total_rows: int = 0
        self._started = False

    def start(
        self,
        stage_name: str,
        total_rows: int,
        deployments: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self._tracker:
            return

        self._stage_name = stage_name
        self._total_rows = total_rows

        if self._run_progress is not None:
            self._run_progress.init_stage(stage_name, total_rows)

        self._task_id = self._tracker.start_stage(
            f"{stage_name}: {total_rows:,} rows",
            total_rows=total_rows,
            deployments=deployments or [],
        )
        self._started = True

    def update(
        self,
        rows_completed: int = 1,
        cost: Decimal | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        deployment_id: str | None = None,
    ) -> None:
        """Apply a progress delta.

        The delta is written to ``RunProgressState`` first, then
        the tracker is asked to render the updated snapshot.
        """
        if not self._tracker or not self._task_id:
            return

        # --- resolve deployment display id ---------------------------------
        display_id = None
        label_info = ""

        if deployment_id and self._deployment_tracker:
            display_id = self._deployment_tracker.register_deployment(deployment_id)
            label_info = self._deployment_tracker.get_label(deployment_id)
            self._deployment_tracker.record_request(deployment_id)

            self._tracker.ensure_deployment_task(
                self._task_id,
                display_id,
                total_rows=self._total_rows,
                label_info=label_info,
            )

        # --- write to shared state FIRST -----------------------------------
        if self._run_progress is not None:
            self._run_progress.apply_delta(
                stage_name=self._stage_name,
                rows_completed=rows_completed,
                cost=cost,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                deployment_id=display_id,
            )

        # --- forward to tracker for rendering ------------------------------
        self._tracker.update(
            self._task_id,
            advance=rows_completed,
            cost=cost,
            deployment_id=display_id,
        )

    def finish(self) -> None:
        """Mark stage as complete."""
        if not self._tracker or not self._task_id:
            return

        self._tracker.finish(self._task_id)
        self._started = False

    @property
    def is_active(self) -> bool:
        """Check if progress tracking is active."""
        return self._started and self._tracker is not None

    @property
    def total_rows(self) -> int:
        """Get total rows being tracked."""
        return self._total_rows

    def __repr__(self) -> str:
        return (
            f"ProgressReporter(active={self.is_active}, total_rows={self._total_rows})"
        )
