"""
Progress reporter for pipeline stage execution.

Provides a clean interface for reporting progress to the UI tracker,
handling deployment-aware updates and lifecycle management.
"""

from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ondine.orchestration.deployment_tracker import DeploymentTracker
    from ondine.orchestration.progress_tracker import ProgressTracker


class ProgressReporter:
    """
    Reports progress to the UI tracker with deployment awareness.

    Encapsulates progress tracking logic that was previously scattered
    throughout LLMInvocationStage, providing a clean lifecycle:

    1. start() - Initialize progress bar with total rows
    2. update() - Increment progress, optionally with deployment info
    3. finish() - Mark stage complete

    Example:
        reporter = ProgressReporter(context.progress_tracker)
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
    ):
        """
        Initialize progress reporter.

        Args:
            tracker: Progress tracker instance (RichProgressTracker or LoggingProgressTracker)
            deployment_tracker: Optional deployment tracker for Router distribution
        """
        self._tracker = tracker
        self._deployment_tracker = deployment_tracker
        self._task_id: Any = None
        self._total_rows: int = 0
        self._started = False

    def start(
        self,
        stage_name: str,
        total_rows: int,
        deployments: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Start progress tracking for a stage.

        Args:
            stage_name: Name of the stage (e.g., "LLMInvocation")
            total_rows: Total number of rows to process
            deployments: Optional list of deployment info for Router visualization
        """
        if not self._tracker:
            return

        self._total_rows = total_rows
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
        deployment_id: str | None = None,
    ) -> None:
        """
        Update progress with completed rows.

        Args:
            rows_completed: Number of rows just completed (default: 1)
            cost: Optional cost for this batch
            deployment_id: Optional deployment ID for Router tracking
        """
        if not self._tracker or not self._task_id:
            return

        # Handle deployment registration if we have a tracker
        display_id = None
        label_info = ""

        if deployment_id and self._deployment_tracker:
            # Register deployment (First-Seen-First-Assigned)
            display_id = self._deployment_tracker.register_deployment(deployment_id)
            label_info = self._deployment_tracker.get_label(deployment_id)

            # Record the request for distribution tracking
            self._deployment_tracker.record_request(deployment_id)

            # Ensure deployment task exists in progress tracker
            # Use full total - actual count will be shown at finish()
            self._tracker.ensure_deployment_task(
                self._task_id,
                display_id,
                total_rows=self._total_rows,
                label_info=label_info,
            )

        # Update progress (updates both main bar and deployment bar)
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
