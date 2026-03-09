"""
Progress tracking abstraction with pluggable implementations.

Provides a generic interface for progress tracking that can be implemented
using different libraries (rich, tqdm, logging) without coupling pipeline
code to specific implementations.
"""

import collections
import re
import sys
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


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


class _StdoutCapture:
    """Intercepts stdout writes (from structlog's PrintLoggerFactory) and
    feeds them into the log panel buffer while the Rich Live display is active."""

    def __init__(self, buffer: collections.deque, original: Any):
        self._buffer = buffer
        self._original = original

    def write(self, text: str) -> int:
        if text.strip():
            for line in text.strip().splitlines():
                clean = _ANSI_RE.sub("", line).strip()
                if clean:
                    self._buffer.append(clean)
        return len(text)

    def flush(self) -> None:
        pass

    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", "utf-8")

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return self._original.fileno()


class RichProgressTracker(ProgressTracker):
    """
    Progress tracker using Rich for a split-panel terminal UI.

    Layout:
    - Top panel: progress bars (fixed, always visible)
    - Bottom panel: scrolling log messages (fills remaining height)

    Features:
    - Contribution-based deployment sub-bars (visually add up to main bar)
    - Automatic ETA and throughput calculation
    - Cost tracking per stage and per deployment
    - Router deployment tracking (sub-progress per deployment)
    - Thread-safe for concurrent execution
    """

    def __init__(self):
        """Initialize rich progress tracker with split layout."""
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
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            TextColumn("[bold green]${task.fields[cost]:.4f}"),
            expand=True,
            auto_refresh=False,
        )
        self.tasks: dict[str, Any] = {}

        self.deployment_tasks: dict[str, dict[str, Any]] = {}
        self.deployment_stats: dict[str, dict[str, int]] = {}

        self._log_buffer: collections.deque[str] = collections.deque(maxlen=500)
        self._live: Any = None

    def _build_layout(self) -> Any:
        """Build a Rich Layout with progress on top and logs below."""
        import shutil

        from rich.layout import Layout
        from rich.panel import Panel

        term_height = shutil.get_terminal_size().lines

        n_tasks = len(self.progress.tasks)
        progress_height = max(n_tasks + 4, 6)

        log_height = max(term_height - progress_height - 2, 5)

        layout = Layout()
        layout.split_column(
            Layout(
                Panel(self.progress, title="[bold]Progress", border_style="blue"),
                name="progress",
                size=progress_height,
            ),
            Layout(
                Panel(
                    self._render_logs(log_height - 2),
                    title="[bold]Logs",
                    border_style="dim",
                ),
                name="logs",
            ),
        )
        return layout

    def _render_logs(self, max_lines: int) -> Any:
        """Render the last N log lines as a Rich Text object."""
        from rich.text import Text

        lines = list(self._log_buffer)
        visible = lines[-max_lines:] if len(lines) > max_lines else lines

        if not visible:
            return Text("  Waiting for log output...", style="dim italic")

        text = Text()
        for i, line in enumerate(visible):
            style = "dim" if "info" in line.lower() else "yellow"
            if "error" in line.lower():
                style = "red"
            elif "warning" in line.lower() or "skipping" in line.lower():
                style = "yellow"
            text.append(line + "\n", style=style)
        return text

    def start_stage(self, stage_name: str, total_rows: int, **metadata: Any) -> str:
        """Start tracking a stage with rich progress bar."""
        task_id = self.progress.add_task(
            f"{stage_name}",
            total=total_rows,
            cost=metadata.get("cost", 0.0),
        )
        self.tasks[stage_name] = task_id

        deployments = metadata.get("deployments", [])
        if deployments:
            self.deployment_tasks[stage_name] = {}
            self.deployment_stats[stage_name] = {}

            for deployment in deployments:
                dep_id = deployment.get("model_id", deployment.get("name", "unknown"))

                if "label" in deployment:
                    label = f"   ├─ {deployment['label']}"
                else:
                    model = deployment.get("model", "")
                    if model:
                        provider = model.split("/")[0] if "/" in model else ""
                        model_short = model.split("/")[1] if "/" in model else model
                        if len(model_short) > 25:
                            model_short = model_short[:22] + "..."
                        label = f"   ├─ {dep_id} ({provider}/{model_short})"
                    else:
                        label = f"   ├─ {dep_id}"

                dep_task_id = self.progress.add_task(
                    label,
                    total=total_rows,
                    cost=0.0,
                )
                self.deployment_tasks[stage_name][dep_id] = dep_task_id
                self.deployment_stats[stage_name][dep_id] = 0

        self._refresh()
        return stage_name

    def ensure_deployment_task(
        self, stage_name: str, deployment_id: str, total_rows: int, label_info: str = ""
    ) -> None:
        """Dynamically add a deployment task if it doesn't exist."""
        if stage_name not in self.deployment_tasks:
            self.deployment_tasks[stage_name] = {}
            self.deployment_stats[stage_name] = {}

        if deployment_id not in self.deployment_tasks[stage_name]:
            if label_info:
                label = f"   ├─ {label_info}"
            else:
                if len(deployment_id) > 30 and " " not in deployment_id:
                    short_id = deployment_id[:8]
                    label = f"   ├─ Deployment ({short_id}...)"
                else:
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

            self._refresh()

    def update(self, task_id: str, advance: int = 1, **metadata: Any) -> None:
        """Update progress bar, including deployment-specific tracking."""
        if task_id not in self.tasks:
            return

        rich_task_id = self.tasks[task_id]

        update_kwargs: dict[str, Any] = {"advance": advance}
        if "cost" in metadata:
            task = self.progress.tasks[rich_task_id]
            current_cost = task.fields.get("cost", 0.0)
            new_cost = float(current_cost) + float(metadata["cost"])
            update_kwargs["cost"] = new_cost

        self.progress.update(rich_task_id, **update_kwargs)

        deployment_id = metadata.get("deployment_id")
        if deployment_id and task_id in self.deployment_tasks:
            if deployment_id in self.deployment_tasks[task_id]:
                dep_task_id = self.deployment_tasks[task_id][deployment_id]

                self.deployment_stats[task_id][deployment_id] += advance
                new_count = self.deployment_stats[task_id][deployment_id]

                dep_kwargs: dict[str, Any] = {"completed": new_count}
                if "cost" in metadata:
                    dep_task = self.progress.tasks[dep_task_id]
                    current_dep_cost = dep_task.fields.get("cost", 0.0)
                    dep_kwargs["cost"] = float(current_dep_cost) + float(
                        metadata["cost"]
                    )

                self.progress.update(dep_task_id, **dep_kwargs)

        self._refresh()

    def finish(self, task_id: str) -> None:
        """Mark task as complete."""
        if task_id in self.tasks:
            rich_task_id = self.tasks[task_id]

            task = self.progress.tasks[rich_task_id]
            total = task.total or 0
            self.progress.update(rich_task_id, completed=total)

            if task_id in self.deployment_tasks:
                for dep_id, dep_task_id in self.deployment_tasks[task_id].items():
                    actual_count = self.deployment_stats.get(task_id, {}).get(dep_id, 0)
                    self.progress.update(dep_task_id, completed=actual_count)

        self._refresh()

    def _refresh(self) -> None:
        """Refresh the live layout display."""
        if self._live is not None:
            self._live.update(self._build_layout())

    def __enter__(self) -> "RichProgressTracker":
        """Start split-panel display and capture stdout for the log panel."""
        from rich.console import Console
        from rich.live import Live

        self._original_stdout = sys.stdout

        console = Console(file=self._original_stdout)
        self._live = Live(
            self._build_layout(),
            refresh_per_second=8,
            screen=False,
            console=console,
        )
        self._live.start()

        sys.stdout = _StdoutCapture(self._log_buffer, self._original_stdout)

        return self

    def __exit__(self, *args: Any) -> None:
        """Restore stdout and stop display."""
        sys.stdout = getattr(self, "_original_stdout", sys.__stdout__)

        if self._live is not None:
            self._live.stop()
            self._live = None


class LoggingProgressTracker(ProgressTracker):
    """
    Fallback progress tracker using standard logging.

    Used when:
    - Not running in a TTY (CI, logs to file)
    - rich library not available
    - User explicitly requests logging mode
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

    def ensure_deployment_task(
        self, stage_name: str, deployment_id: str, total_rows: int, label_info: str = ""
    ) -> None:
        """LoggingTracker has no deployment sub-tasks."""
        pass

    def update(self, task_id: str, advance: int = 1, **metadata: Any) -> None:
        """Update progress via periodic logging."""
        if task_id not in self.tasks:
            return

        task = self.tasks[task_id]
        task["current"] += advance

        if "cost" in metadata:
            task["cost"] += Decimal(str(metadata["cost"]))

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

    Args:
        mode: Progress tracker mode
            - "auto": Auto-detect (rich if TTY, else logging)
            - "rich": Use rich.progress (beautiful UI)
            - "tqdm": Use tqdm (simple, compatible)
            - "logging": Use standard logging (fallback)
            - "none": Disable progress tracking
    """
    if mode == "none":
        return NoOpProgressTracker()

    if mode == "auto":
        if sys.stdout.isatty():
            try:
                from rich.progress import Progress  # noqa: F401

                return RichProgressTracker()
            except ImportError:
                return LoggingProgressTracker()
        else:
            return LoggingProgressTracker()

    elif mode == "rich":
        return RichProgressTracker()

    elif mode == "tqdm":
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
