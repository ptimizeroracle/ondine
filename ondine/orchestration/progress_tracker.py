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
    - Router deployment tracking (sub-progress per deployment)
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
                tracker.update(task, advance=1, cost=0.001, deployment_id="groq-key-1")

            tracker.finish(task)
        ```
    """

    def __init__(self):
        """Initialize rich progress tracker."""
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
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
            MofNCompleteColumn(),  # Shows "completed/total" (e.g., "120/1200")
            TaskProgressColumn(),  # Shows percentage
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            TextColumn("[bold green]${task.fields[cost]:.4f}"),
            expand=True,
            auto_refresh=True,  # Enable auto-refresh for smooth animation (spinners, timers)
            refresh_per_second=10,
        )
        self.tasks: dict[str, Any] = {}

        # Router deployment tracking
        self.deployment_tasks: dict[
            str, dict[str, Any]
        ] = {}  # stage -> {deployment_id: task_id}
        self.deployment_stats: dict[
            str, dict[str, int]
        ] = {}  # stage -> {deployment_id: count}

    def start_stage(self, stage_name: str, total_rows: int, **metadata: Any) -> str:
        """Start tracking a stage with rich progress bar."""
        # Add main task
        task_id = self.progress.add_task(
            f"🚀 {stage_name}",
            total=total_rows,
            cost=metadata.get("cost", 0.0),
        )
        self.tasks[stage_name] = task_id

        # Initialize deployment tracking for this stage
        deployments = metadata.get("deployments", [])
        if deployments:
            self.deployment_tasks[stage_name] = {}
            self.deployment_stats[stage_name] = {}

            # Calculate rows per deployment (weighted distribution)
            weights = [d.get("weight", 1.0) for d in deployments]
            total_weight = sum(weights)

            # Add ALL deployment bars
            for deployment in deployments:
                dep_id = deployment.get("model_id", deployment.get("name", "unknown"))
                weight = deployment.get("weight", 1.0)
                dep_rows = int(total_rows * (weight / total_weight))

                # Use provided label if available
                if "label" in deployment:
                    label = f"   ├─ {deployment['label']}"
                else:
                    # Build label: "key-id (provider/model)"
                    model = deployment.get("model", "")
                    if model:
                        # Extract provider from model string (e.g., "groq/llama-3.3" -> "groq")
                        provider = model.split("/")[0] if "/" in model else ""
                        model_short = model.split("/")[1] if "/" in model else model
                        # Truncate long model names
                        if len(model_short) > 25:
                            model_short = model_short[:22] + "..."
                        label = f"   ├─ {dep_id} ({provider}/{model_short})"
                    else:
                        label = f"   ├─ {dep_id}"

                # Create sub-task for this deployment
                dep_task_id = self.progress.add_task(
                    label,
                    total=dep_rows,
                    cost=0.0,
                )
                self.deployment_tasks[stage_name][dep_id] = dep_task_id
                self.deployment_stats[stage_name][dep_id] = 0

        # Manual refresh to show all bars at once
        self.progress.refresh()

        return stage_name

    def ensure_deployment_task(
        self, stage_name: str, deployment_id: str, total_rows: int, label_info: str = ""
    ) -> None:
        """Dynamically add a deployment task if it doesn't exist."""
        # Initialize dictionary if needed
        if stage_name not in self.deployment_tasks:
            self.deployment_tasks[stage_name] = {}
            self.deployment_stats[stage_name] = {}

        # If task doesn't exist, create it
        if deployment_id not in self.deployment_tasks[stage_name]:
            # Use provided label info (e.g. "groq/llama-3...") or format the ID
            if label_info:
                # Use full label without truncation
                label = f"   ├─ {label_info}"
            else:
                # Create clean label from ID
                if len(deployment_id) > 30 and " " not in deployment_id:
                    # Likely a hash - show shortened version
                    short_id = deployment_id[:8]
                    label = f"   ├─ Deployment ({short_id}...)"
                else:
                    # Friendly name or short ID
                    label = (
                        f"   ├─ {deployment_id[:30]}..."
                        if len(deployment_id) > 33
                        else f"   ├─ {deployment_id}"
                    )

            dep_task_id = self.progress.add_task(
                label,
                total=total_rows,
                cost=0.0,
            )
            self.deployment_tasks[stage_name][deployment_id] = dep_task_id
            self.deployment_stats[stage_name][deployment_id] = 0

            # Refresh to show new bar
            self.progress.refresh()

    def update(self, task_id: str, advance: int = 1, **metadata: Any) -> None:
        """Update progress bar, including deployment-specific tracking."""
        if task_id not in self.tasks:
            return

        rich_task_id = self.tasks[task_id]

        # Update cost if provided
        update_kwargs = {"advance": advance}
        if "cost" in metadata:
            task = self.progress.tasks[rich_task_id]
            current_cost = task.fields.get("cost", 0.0)
            new_cost = float(current_cost) + float(metadata["cost"])
            update_kwargs["cost"] = new_cost

        self.progress.update(rich_task_id, **update_kwargs)

        # Update deployment-specific progress if provided
        deployment_id = metadata.get("deployment_id")
        if deployment_id and task_id in self.deployment_tasks:
            if deployment_id in self.deployment_tasks[task_id]:
                dep_task_id = self.deployment_tasks[task_id][deployment_id]

                # Update deployment cost individually
                dep_kwargs = {"advance": advance}
                if "cost" in metadata:
                    dep_task = self.progress.tasks[dep_task_id]
                    current_dep_cost = dep_task.fields.get("cost", 0.0)
                    dep_kwargs["cost"] = float(current_dep_cost) + float(
                        metadata["cost"]
                    )

                self.progress.update(dep_task_id, **dep_kwargs)
                self.deployment_stats[task_id][deployment_id] += advance

    def finish(self, task_id: str) -> None:
        """Mark task as complete."""
        if task_id in self.tasks:
            rich_task_id = self.tasks[task_id]
            
            # Get total to ensure bar fills up completely
            task = self.progress.tasks[rich_task_id]
            total = task.total or 0
            
            self.progress.update(rich_task_id, completed=total)

            # Also complete deployment sub-tasks
            if task_id in self.deployment_tasks:
                for dep_task_id in self.deployment_tasks[task_id].values():
                    # Get total for sub-task
                    dep_task = self.progress.tasks[dep_task_id]
                    dep_total = dep_task.total or 0
                    self.progress.update(dep_task_id, completed=dep_total)

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
