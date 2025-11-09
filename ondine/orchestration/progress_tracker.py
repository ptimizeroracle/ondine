"""
Progress tracking abstraction with pluggable implementations.

Provides a generic interface for progress tracking that can be implemented
using different libraries (rich, tqdm, logging) without coupling pipeline
code to specific implementations.
"""

import sys
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any


class ProgressTracker(ABC):
    """
    Abstract interface for progress tracking.

    Enables pluggable progress tracking implementations (rich, tqdm, logging)
    without coupling pipeline code to specific libraries.

    Design Pattern: Strategy Pattern
    - Pipeline depends on ProgressTracker interface (abstraction)
    - Concrete implementations (RichProgressTracker, TqdmProgressTracker) are interchangeable
    - Follows Dependency Inversion Principle (SOLID)

    Example:
        ```python
        tracker = create_progress_tracker(mode="auto")

        with tracker:
            task_id = tracker.start_stage("Classification", total_rows=1000)

            for row in rows:
                process(row)
                tracker.update(task_id, advance=1, cost=0.001)

            tracker.finish(task_id)
        ```
    """

    @abstractmethod
    def start_stage(self, stage_name: str, total_rows: int, **metadata: Any) -> str:
        """
        Start tracking a new stage.

        Args:
            stage_name: Human-readable stage name (e.g., "Primary Category Classification")
            total_rows: Total number of rows to process
            **metadata: Additional metadata (cost_so_far, stage_number, etc.)

        Returns:
            Task ID for updating progress
        """
        pass

    @abstractmethod
    def update(self, task_id: str, advance: int = 1, **metadata: Any) -> None:
        """
        Update progress for a task.

        Args:
            task_id: Task identifier from start_stage()
            advance: Number of rows processed
            **metadata: Additional metadata (cost, tokens, etc.)
        """
        pass

    @abstractmethod
    def finish(self, task_id: str) -> None:
        """
        Mark task as complete.

        Args:
            task_id: Task identifier
        """
        pass

    @abstractmethod
    def __enter__(self) -> "ProgressTracker":
        """Context manager entry."""
        pass

    @abstractmethod
    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        pass


class RichProgressTracker(ProgressTracker):
    """
    Progress tracker using rich.progress for beautiful terminal UI.

    Features:
    - Multiple progress bars (one per stage)
    - Automatic ETA and throughput calculation
    - Color-coded output
    - Cost tracking per stage
    - Thread-safe for concurrent execution

    Requires:
        rich library (already a dependency)

    Example:
        ```python
        tracker = RichProgressTracker()

        with tracker:
            task = tracker.start_stage("Stage 1: Classification", total_rows=1000)

            for i in range(1000):
                process_row(i)
                tracker.update(task, advance=1, cost=0.001)

            tracker.finish(task)
        ```
    """

    def __init__(self):
        """Initialize rich progress tracker."""
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            TextColumn("[bold green]${task.fields[cost]:.4f}"),
            expand=True,
        )
        self.tasks: dict[str, Any] = {}

    def start_stage(self, stage_name: str, total_rows: int, **metadata: Any) -> str:
        """Start tracking a stage with rich progress bar."""
        task_id = self.progress.add_task(
            f"🚀 {stage_name}",
            total=total_rows,
            cost=metadata.get("cost", 0.0),
        )
        self.tasks[stage_name] = task_id
        return stage_name  # Use stage_name as task_id for simplicity

    def update(self, task_id: str, advance: int = 1, **metadata: Any) -> None:
        """Update progress bar."""
        if task_id not in self.tasks:
            return

        rich_task_id = self.tasks[task_id]

        # Update cost if provided
        update_kwargs = {"advance": advance}
        if "cost" in metadata:
            # Get current cost and add new cost
            task = self.progress.tasks[rich_task_id]
            current_cost = task.fields.get("cost", 0.0)
            new_cost = float(current_cost) + float(metadata["cost"])
            update_kwargs["cost"] = new_cost

        self.progress.update(rich_task_id, **update_kwargs)

    def finish(self, task_id: str) -> None:
        """Mark task as complete."""
        if task_id in self.tasks:
            rich_task_id = self.tasks[task_id]
            self.progress.update(rich_task_id, completed=True)

    def __enter__(self) -> "RichProgressTracker":
        """Start progress display."""
        self.progress.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop progress display."""
        self.progress.stop()


class LoggingProgressTracker(ProgressTracker):
    """
    Fallback progress tracker using standard logging.

    Used when:
    - Not running in a TTY (CI, logs to file)
    - rich library not available
    - User explicitly requests logging mode

    Provides basic progress updates via log messages without fancy UI.

    Example:
        ```python
        tracker = LoggingProgressTracker()

        with tracker:
            task = tracker.start_stage("Classification", total_rows=1000)
            # Logs: "Starting Classification (1000 rows)"

            for i in range(1000):
                tracker.update(task, advance=1)
                # Logs periodically: "Classification: 250/1000 (25%)"

            tracker.finish(task)
            # Logs: "Completed Classification"
        ```
    """

    def __init__(self):
        """Initialize logging tracker."""
        from ondine.utils import get_logger

        self.logger = get_logger(__name__)
        self.tasks: dict[str, dict[str, Any]] = {}

    def start_stage(self, stage_name: str, total_rows: int, **metadata: Any) -> str:
        """Start tracking via logging."""
        self.tasks[stage_name] = {
            "total": total_rows,
            "current": 0,
            "cost": Decimal("0.0"),
            "last_log_percent": 0,
        }
        self.logger.info(f"Starting {stage_name} ({total_rows} rows)")
        return stage_name

    def update(self, task_id: str, advance: int = 1, **metadata: Any) -> None:
        """Update progress via periodic logging."""
        if task_id not in self.tasks:
            return

        task = self.tasks[task_id]
        task["current"] += advance

        if "cost" in metadata:
            task["cost"] += Decimal(str(metadata["cost"]))

        # Log at 25%, 50%, 75%, 100%
        percent = (task["current"] / task["total"]) * 100
        milestones = [25, 50, 75, 100]

        for milestone in milestones:
            if percent >= milestone and task["last_log_percent"] < milestone:
                self.logger.info(
                    f"{task_id}: {task['current']}/{task['total']} "
                    f"({percent:.1f}%) | Cost: ${task['cost']:.4f}"
                )
                task["last_log_percent"] = milestone
                break

    def finish(self, task_id: str) -> None:
        """Log completion."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            self.logger.info(
                f"Completed {task_id}: {task['current']}/{task['total']} rows, "
                f"${task['cost']:.4f}"
            )

    def __enter__(self) -> "LoggingProgressTracker":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        pass


def create_progress_tracker(mode: str = "auto") -> ProgressTracker:
    """
    Factory function to create appropriate progress tracker.

    Automatically detects the best progress tracker based on environment
    and available libraries, or uses explicit mode if specified.

    Args:
        mode: Progress tracker mode
            - "auto": Auto-detect (rich if TTY, else logging)
            - "rich": Use rich.progress (beautiful UI)
            - "tqdm": Use tqdm (simple, compatible)
            - "logging": Use standard logging (fallback)
            - "none": Disable progress tracking

    Returns:
        ProgressTracker implementation

    Example:
        ```python
        # Auto-detect best option
        tracker = create_progress_tracker(mode="auto")

        # Force rich (will fail if not available)
        tracker = create_progress_tracker(mode="rich")

        # Force logging (always works)
        tracker = create_progress_tracker(mode="logging")
        ```
    """
    if mode == "none":
        return NoOpProgressTracker()

    if mode == "auto":
        # Auto-detect best option
        if sys.stdout.isatty():
            # Running in terminal - try rich first
            try:
                from rich.progress import Progress  # noqa: F401

                return RichProgressTracker()
            except ImportError:
                # rich not available, fall back to logging
                return LoggingProgressTracker()
        else:
            # Non-TTY environment (CI, logs to file) - use logging
            return LoggingProgressTracker()

    elif mode == "rich":
        return RichProgressTracker()

    elif mode == "tqdm":
        # Future: implement TqdmProgressTracker
        raise NotImplementedError(
            "tqdm tracker not yet implemented, use 'rich' or 'logging'"
        )

    elif mode == "logging":
        return LoggingProgressTracker()

    else:
        raise ValueError(
            f"Invalid progress mode: {mode}. "
            f"Use 'auto', 'rich', 'tqdm', 'logging', or 'none'"
        )


class NoOpProgressTracker(ProgressTracker):
    """No-op tracker that does nothing (for disabling progress)."""

    def start_stage(self, stage_name: str, total_rows: int, **metadata: Any) -> str:
        return stage_name

    def update(self, task_id: str, advance: int = 1, **metadata: Any) -> None:
        pass

    def finish(self, task_id: str) -> None:
        pass

    def __enter__(self) -> "NoOpProgressTracker":
        return self

    def __exit__(self, *args: Any) -> None:
        pass
