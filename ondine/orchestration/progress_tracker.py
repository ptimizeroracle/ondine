"""
Progress tracking abstraction with pluggable implementations.

Provides a generic interface for progress tracking that can be implemented
using different libraries (rich, tqdm, logging) without coupling pipeline
code to specific implementations.
"""

import collections
import contextlib
import re
import sys
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _build_summary_panel(result: Any) -> Any:
    """Build a Rich Panel containing a pipeline execution summary table."""
    from rich.panel import Panel
    from rich.table import Table

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column("Key", style="bold cyan", no_wrap=True)
    t.add_column("Value", style="white")

    duration = getattr(result, "duration", 0.0)
    minutes, secs = divmod(duration, 60)
    time_str = f"{int(minutes)}m {secs:.1f}s" if minutes else f"{secs:.1f}s"

    metrics = result.metrics
    costs = result.costs

    output_rows = len(result.data) if result.data is not None else 0
    llm_calls = metrics.processed_rows
    cached = max(
        output_rows - llm_calls - metrics.failed_rows - metrics.skipped_rows, 0
    )
    cpr = float(costs.total_cost) / max(output_rows, 1)

    status = "[bold green]COMPLETE" if result.success else "[bold red]FAILED"
    t.add_row("Status", status)
    t.add_row("Output rows", f"{output_rows:,}")
    if cached > 0:
        t.add_row("Cached rows", f"[dim]{cached:,} (no LLM cost)[/dim]")
    t.add_row("LLM calls", f"{llm_calls:,}")
    if metrics.failed_rows:
        t.add_row("Failed rows", f"[red]{metrics.failed_rows:,}[/red]")
    if metrics.skipped_rows:
        t.add_row("Skipped rows", f"[yellow]{metrics.skipped_rows:,}[/yellow]")
    t.add_row("Duration", time_str)
    if metrics.rows_per_second:
        t.add_row("Throughput", f"{metrics.rows_per_second:.1f} rows/sec")
    t.add_row("", "")
    t.add_row("Total cost", f"${costs.total_cost:.4f} (${cpr:.6f}/row)")
    t.add_row("Input tokens", f"{costs.input_tokens:,}")
    t.add_row("Output tokens", f"{costs.output_tokens:,}")
    t.add_row("Total tokens", f"{costs.total_tokens:,}")

    return Panel(t, title="[bold]Pipeline Report", border_style="green", padding=(1, 2))


class ProgressTracker(ABC):
    """
    Abstract interface for progress tracking.

    Enables pluggable progress tracking implementations (rich, tqdm, logging)
    without coupling pipeline code to specific libraries.

    Design Pattern: Strategy Pattern
    - Pipeline depends on ProgressTracker interface (abstraction)
    - Concrete implementations (RichProgressTracker, TqdmProgressTracker) are interchangeable
    - Follows Dependency Inversion Principle (SOLID)

    All tracker implementations receive the same cost deltas from
    :class:`ProgressReporter`, which first writes to the shared
    :class:`RunProgressState`.  Trackers are *renderers* — they may keep
    rendering-local state (e.g. Rich ``task.fields``) but the authoritative
    numbers live in ``RunProgressState``.

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

    _run_progress: Any = None

    def set_run_progress(self, run_progress: Any) -> None:
        """Attach the shared RunProgressState for live cost reads."""
        self._run_progress = run_progress

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

    def show_summary(self, result: Any) -> None:
        """Display a final pipeline summary report.

        Called automatically after pipeline execution completes.
        Default implementation does nothing; rich/textual trackers
        render a formatted table.

        Args:
            result: ExecutionResult with metrics, costs, duration, etc.
        """

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

        self.deployment_stats: dict[str, dict[str, int]] = {}
        self._deployment_labels: dict[str, dict[str, str]] = {}
        self._deployment_costs: dict[str, dict[str, float]] = {}

        self._log_buffer: collections.deque[str] = collections.deque(maxlen=500)
        self._live: Any = None

    def _build_layout(self) -> Any:
        """Build a Rich Layout with progress on top and logs below."""
        import shutil

        from rich.console import Group
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.text import Text

        term_height = shutil.get_terminal_size().lines

        dep_lines: list[str] = []
        for stage_name in self.deployment_stats:
            text = self._build_deployment_text(stage_name)
            if text:
                dep_lines.extend(text.split("\n"))

        n_tasks = len(self.progress.tasks)
        n_dep_lines = len(dep_lines)
        progress_height = max(n_tasks + n_dep_lines + 4, 6)

        log_height = max(term_height - progress_height - 2, 5)

        progress_content: Any
        if dep_lines:
            dep_text = Text()
            for line in dep_lines:
                dep_text.append(line + "\n", style="dim")
            progress_content = Group(self.progress, dep_text)
        else:
            progress_content = self.progress

        layout = Layout()
        layout.split_column(
            Layout(
                Panel(progress_content, title="[bold]Progress", border_style="blue"),
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
            self.deployment_stats[stage_name] = {}
            self._deployment_labels[stage_name] = {}
            self._deployment_costs[stage_name] = {}

            for deployment in deployments:
                dep_id = deployment.get("model_id", deployment.get("name", "unknown"))

                if "label" in deployment:
                    base_label = deployment["label"]
                else:
                    model = deployment.get("model", "")
                    if model:
                        provider = model.split("/")[0] if "/" in model else ""
                        model_short = model.split("/")[1] if "/" in model else model
                        if len(model_short) > 25:
                            model_short = model_short[:22] + "..."
                        base_label = f"{dep_id} ({provider}/{model_short})"
                    else:
                        base_label = dep_id

                self.deployment_stats[stage_name][dep_id] = 0
                self._deployment_labels[stage_name][dep_id] = base_label
                self._deployment_costs[stage_name][dep_id] = 0.0

        self._refresh()
        return stage_name

    def ensure_deployment_task(
        self, stage_name: str, deployment_id: str, total_rows: int, label_info: str = ""
    ) -> None:
        """Register a deployment for text-only sub-row tracking."""
        if stage_name not in self.deployment_stats:
            self.deployment_stats[stage_name] = {}
            self._deployment_labels[stage_name] = {}
            self._deployment_costs[stage_name] = {}

        if deployment_id not in self.deployment_stats[stage_name]:
            if label_info:
                base_label = label_info
            elif len(deployment_id) > 30 and " " not in deployment_id:
                base_label = f"Deployment ({deployment_id[:8]}...)"
            elif len(deployment_id) > 33:
                base_label = f"{deployment_id[:30]}..."
            else:
                base_label = deployment_id

            self.deployment_stats[stage_name][deployment_id] = 0
            self._deployment_labels[stage_name][deployment_id] = base_label
            self._deployment_costs[stage_name][deployment_id] = 0.0

            self._refresh()

    def _build_deployment_text(self, stage_name: str) -> str:
        """Build text lines for deployment sub-rows of a given stage."""
        if stage_name not in self.deployment_stats:
            return ""
        lines: list[str] = []
        for dep_id in self.deployment_stats[stage_name]:
            label = self._deployment_labels.get(stage_name, {}).get(dep_id, dep_id)
            count = self.deployment_stats[stage_name][dep_id]
            cost = self._deployment_costs.get(stage_name, {}).get(dep_id, 0.0)
            cost_str = f"  ${cost:.4f}" if cost > 0 else ""
            lines.append(f"   ├─ {label}  {count:,} rows{cost_str}")
        return "\n".join(lines)

    def update(self, task_id: str, advance: int = 1, **metadata: Any) -> None:
        """Update progress bar, including deployment-specific tracking."""
        if task_id not in self.tasks:
            return

        rich_task_id = self.tasks[task_id]

        update_kwargs: dict[str, Any] = {"advance": advance}
        if "cost" in metadata and metadata["cost"] is not None:
            task = self.progress.tasks[rich_task_id]
            current_cost = task.fields.get("cost", 0.0)
            new_cost = float(current_cost) + float(metadata["cost"])
            update_kwargs["cost"] = new_cost

        self.progress.update(rich_task_id, **update_kwargs)

        deployment_id = metadata.get("deployment_id")
        if deployment_id and task_id in self.deployment_stats:
            if deployment_id in self.deployment_stats[task_id]:
                self.deployment_stats[task_id][deployment_id] += advance

                if "cost" in metadata and metadata["cost"] is not None:
                    self._deployment_costs.setdefault(task_id, {})[deployment_id] = (
                        self._deployment_costs.get(task_id, {}).get(deployment_id, 0.0)
                        + float(metadata["cost"])
                    )

        self._refresh()

    def finish(self, task_id: str) -> None:
        """Mark task as complete."""
        if task_id in self.tasks:
            rich_task_id = self.tasks[task_id]

            task = self.progress.tasks[rich_task_id]
            total = task.total or 0
            self.progress.update(rich_task_id, completed=total)

        self._refresh()

    def _refresh(self) -> None:
        """Refresh the live layout display."""
        if self._live is not None:
            self._live.update(self._build_layout())

    def show_summary(self, result: Any) -> None:
        """Print a Rich summary panel to the terminal."""
        from rich.console import Console

        Console().print()
        Console().print(_build_summary_panel(result))

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


class TextualProgressTracker(ProgressTracker):
    """
    Progress tracker using Textual for a full TUI with interactive scrolling.

    Layout:
    - Top: fixed progress bars (same Rich Progress columns as RichProgressTracker)
    - Bottom: RichLog widget with native scrollbar for log inspection

    Requires ``textual`` (install with ``pip install ondine[tui]``).

    Threading model: Textual owns its own event loop in a daemon thread.
    All UI mutations go through ``App.call_from_thread()`` for thread safety.
    """

    def __init__(self) -> None:
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
        self.deployment_stats: dict[str, dict[str, int]] = {}
        self._deployment_labels: dict[str, dict[str, str]] = {}
        self._deployment_costs: dict[str, dict[str, float]] = {}

        self._app: Any = None
        self._app_thread: Any = None
        self._ready_event: Any = None

    # -- lifecycle ------------------------------------------------------------

    def __enter__(self) -> "TextualProgressTracker":
        import threading

        from ondine.orchestration._pipeline_tui import (
            PipelineApp,
            _install_thread_safe_signal_patch,
        )

        self._ready_event = threading.Event()
        self._original_stdout = sys.stdout

        is_tty = (
            hasattr(self._original_stdout, "isatty") and self._original_stdout.isatty()
        )

        if is_tty:
            _install_thread_safe_signal_patch()

        self._app = PipelineApp(self.progress, self._ready_event)

        run_kwargs: dict[str, Any] = {}
        if not is_tty:
            run_kwargs["headless"] = True
            run_kwargs["size"] = (120, 40)

        def _run() -> None:
            self._app.run(**run_kwargs)

        self._app_thread = threading.Thread(target=_run, daemon=True)
        self._app_thread.start()

        self._ready_event.wait(timeout=10)

        sys.stdout = _TextualStdoutCapture(self._app, self._original_stdout)

        return self

    def show_summary(self, result: Any) -> None:
        """Print a Rich summary panel to the terminal (after TUI has closed)."""
        from rich.console import Console

        Console().print()
        Console().print(_build_summary_panel(result))

    def __exit__(self, *args: Any) -> None:
        sys.stdout = getattr(self, "_original_stdout", sys.__stdout__)

        if self._app is not None:
            with contextlib.suppress(Exception):
                if self._app.is_running:
                    self._app.call_from_thread(self._app.exit)
        if self._app_thread is not None:
            self._app_thread.join(timeout=5)
        self._app = None
        self._app_thread = None

    # -- interface ------------------------------------------------------------

    def start_stage(self, stage_name: str, total_rows: int, **metadata: Any) -> str:
        task_id = self.progress.add_task(
            f"{stage_name}",
            total=total_rows,
            cost=metadata.get("cost", 0.0),
        )
        self.tasks[stage_name] = task_id

        deployments = metadata.get("deployments", [])
        if deployments:
            self.deployment_stats[stage_name] = {}
            self._deployment_labels[stage_name] = {}
            self._deployment_costs[stage_name] = {}
            for deployment in deployments:
                dep_id = deployment.get("model_id", deployment.get("name", "unknown"))
                base_label = self._base_label(deployment, dep_id)
                self.deployment_stats[stage_name][dep_id] = 0
                self._deployment_labels[stage_name][dep_id] = base_label
                self._deployment_costs[stage_name][dep_id] = 0.0

        self._refresh_progress()
        return stage_name

    def ensure_deployment_task(
        self, stage_name: str, deployment_id: str, total_rows: int, label_info: str = ""
    ) -> None:
        if stage_name not in self.deployment_stats:
            self.deployment_stats[stage_name] = {}
            self._deployment_labels[stage_name] = {}
            self._deployment_costs[stage_name] = {}

        if deployment_id not in self.deployment_stats[stage_name]:
            if label_info:
                base_label = label_info
            elif len(deployment_id) > 30 and " " not in deployment_id:
                base_label = f"Deployment ({deployment_id[:8]}...)"
            elif len(deployment_id) > 33:
                base_label = f"{deployment_id[:30]}..."
            else:
                base_label = deployment_id

            self.deployment_stats[stage_name][deployment_id] = 0
            self._deployment_labels[stage_name][deployment_id] = base_label
            self._deployment_costs[stage_name][deployment_id] = 0.0
            self._refresh_progress()

    def _build_deployment_text(self, stage_name: str) -> str:
        """Build text lines for deployment sub-rows of a given stage."""
        if stage_name not in self.deployment_stats:
            return ""
        lines: list[str] = []
        for dep_id in self.deployment_stats[stage_name]:
            label = self._deployment_labels.get(stage_name, {}).get(dep_id, dep_id)
            count = self.deployment_stats[stage_name][dep_id]
            cost = self._deployment_costs.get(stage_name, {}).get(dep_id, 0.0)
            cost_str = f"  ${cost:.4f}" if cost > 0 else ""
            lines.append(f"   ├─ {label}  {count:,} rows{cost_str}")
        return "\n".join(lines)

    def update(self, task_id: str, advance: int = 1, **metadata: Any) -> None:
        if task_id not in self.tasks:
            return

        rich_task_id = self.tasks[task_id]
        update_kwargs: dict[str, Any] = {"advance": advance}
        if "cost" in metadata and metadata["cost"] is not None:
            task = self.progress.tasks[rich_task_id]
            current_cost = task.fields.get("cost", 0.0)
            update_kwargs["cost"] = float(current_cost) + float(metadata["cost"])
        self.progress.update(rich_task_id, **update_kwargs)

        deployment_id = metadata.get("deployment_id")
        if deployment_id and task_id in self.deployment_stats:
            if deployment_id in self.deployment_stats[task_id]:
                self.deployment_stats[task_id][deployment_id] += advance

                if "cost" in metadata and metadata["cost"] is not None:
                    self._deployment_costs.setdefault(task_id, {})[deployment_id] = (
                        self._deployment_costs.get(task_id, {}).get(deployment_id, 0.0)
                        + float(metadata["cost"])
                    )

        self._refresh_progress()

    def finish(self, task_id: str) -> None:
        if task_id in self.tasks:
            rich_task_id = self.tasks[task_id]
            task = self.progress.tasks[rich_task_id]
            self.progress.update(rich_task_id, completed=task.total or 0)

        self._refresh_progress()

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _base_label(deployment: dict, dep_id: str) -> str:
        """Extract a clean base label from deployment config."""
        if "label" in deployment:
            return deployment["label"]
        model = deployment.get("model", "")
        if model:
            provider = model.split("/")[0] if "/" in model else ""
            model_short = model.split("/")[1] if "/" in model else model
            if len(model_short) > 25:
                model_short = model_short[:22] + "..."
            return f"{dep_id} ({provider}/{model_short})"
        return dep_id

    def _refresh_progress(self) -> None:
        if self._app is not None and self._app.is_running:
            dep_lines: list[str] = []
            for stage_name in self.deployment_stats:
                text = self._build_deployment_text(stage_name)
                if text:
                    dep_lines.append(text)
            dep_text = "\n".join(dep_lines)
            with contextlib.suppress(Exception):
                self._app.call_from_thread(self._app.refresh_progress_widget, dep_text)


class _TextualStdoutCapture:
    """Redirects stdout writes into the Textual RichLog widget."""

    def __init__(self, app: Any, original: Any) -> None:
        self._app = app
        self._original = original

    def write(self, text: str) -> int:
        if text.strip():
            for line in text.strip().splitlines():
                clean = _ANSI_RE.sub("", line).strip()
                if clean and self._app.is_running:
                    with contextlib.suppress(Exception):
                        self._app.call_from_thread(self._app.add_log_line, clean)
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


class LoggingProgressTracker(ProgressTracker):
    """
    Fallback progress tracker using standard logging.

    Used when:
    - Not running in a TTY (CI, logs to file)
    - rich library not available
    - User explicitly requests logging mode
    """

    _LOG_INTERVAL_PCT = 5
    _LOG_INTERVAL_SECONDS = 10.0

    def __init__(self):
        """Initialize logging tracker."""
        import time

        from ondine.utils import get_logger

        self.logger = get_logger(__name__)
        self.tasks: dict[str, dict[str, Any]] = {}
        self._deployments: dict[str, dict[str, int]] = {}
        self._deployment_labels: dict[str, dict[str, str]] = {}
        self._time = time

    def start_stage(self, stage_name: str, total_rows: int, **metadata: Any) -> str:
        """Start tracking via logging."""
        self.tasks[stage_name] = {
            "total": total_rows,
            "current": 0,
            "cost": Decimal("0.0"),
            "last_log_percent": 0,
            "start_time": self._time.monotonic(),
            "last_log_time": 0.0,
        }

        deployments = metadata.get("deployments", [])
        if deployments:
            self._deployments[stage_name] = {}
            self._deployment_labels[stage_name] = {}
            for deployment in deployments:
                dep_id = deployment.get("model_id", deployment.get("name", "unknown"))
                self._deployments[stage_name][dep_id] = 0
                self._deployment_labels[stage_name][dep_id] = self._base_label(
                    deployment, dep_id
                )

        self.logger.info(f"Starting stage: {stage_name}")
        return stage_name

    def ensure_deployment_task(
        self, stage_name: str, deployment_id: str, total_rows: int, label_info: str = ""
    ) -> None:
        """Register a deployment for router-aware log snapshots."""
        if stage_name not in self._deployments:
            self._deployments[stage_name] = {}
            self._deployment_labels[stage_name] = {}

        if deployment_id not in self._deployments[stage_name]:
            self._deployments[stage_name][deployment_id] = 0
            self._deployment_labels[stage_name][deployment_id] = (
                label_info or deployment_id
            )

    def update(self, task_id: str, advance: int = 1, **metadata: Any) -> None:
        """Update progress with milestone or heartbeat logging."""
        if task_id not in self.tasks:
            return

        task = self.tasks[task_id]
        task["current"] += advance

        if "cost" in metadata and metadata["cost"] is not None:
            task["cost"] += Decimal(str(metadata["cost"]))

        deployment_id = metadata.get("deployment_id")
        if deployment_id and task_id in self._deployments:
            self._deployments[task_id].setdefault(deployment_id, 0)
            self._deployments[task_id][deployment_id] += advance

        total = task["total"]
        current = task["current"]
        if total == 0:
            return

        percent = (current / total) * 100
        now = self._time.monotonic()
        next_milestone = task["last_log_percent"] + self._LOG_INTERVAL_PCT
        hit_milestone = percent >= next_milestone
        hit_heartbeat = (
            current > 0 and (now - task["last_log_time"]) >= self._LOG_INTERVAL_SECONDS
        )
        if not (hit_milestone or hit_heartbeat):
            return

        task["last_log_percent"] = (
            int(percent // self._LOG_INTERVAL_PCT) * self._LOG_INTERVAL_PCT
        )
        task["last_log_time"] = now

        # Use shared run-level cost when available (single source of truth)
        live_cost = (
            self._run_progress.snapshot_cost
            if self._run_progress is not None
            else task["cost"]
        )

        elapsed = now - task["start_time"]
        rows_per_second = current / max(elapsed, 0.01)
        remaining = (total - current) / max(rows_per_second, 0.01)
        eta_str = self._format_duration(remaining) if current < total else "done"
        elapsed_str = self._format_duration(elapsed)

        self.logger.info(
            f"[progress] {task_id} | {percent:.1f}% | {current:,}/{total:,} rows | "
            f"{rows_per_second:.0f} rows/s | {elapsed_str} elapsed | ETA {eta_str} | "
            f"${live_cost:.4f}"
        )

        for line in self._format_deployment_lines(task_id, elapsed):
            self.logger.info(line)

    def finish(self, task_id: str) -> None:
        """Log completion."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            elapsed = self._time.monotonic() - task["start_time"]
            rows_per_second = task["current"] / max(elapsed, 0.01)
            live_cost = (
                self._run_progress.snapshot_cost
                if self._run_progress is not None
                else task["cost"]
            )
            self.logger.info(
                f"Completed {task_id}: {task['current']:,}/{task['total']:,} rows | "
                f"{self._format_duration(elapsed)} | {rows_per_second:.0f} rows/s | "
                f"${live_cost:.4f}"
            )
            for line in self._format_deployment_lines(task_id, elapsed):
                self.logger.info(line)

    def show_summary(self, result: Any) -> None:
        """Log a framed ASCII summary report."""
        metrics = result.metrics
        costs = result.costs
        duration = getattr(result, "duration", 0.0)

        output_rows = len(result.data) if result.data is not None else 0
        llm_calls = metrics.processed_rows
        cached = max(
            output_rows - llm_calls - metrics.failed_rows - metrics.skipped_rows, 0
        )

        status = "COMPLETE" if result.success else "FAILED"
        dur_str = self._format_duration(duration)
        cpr = float(costs.total_cost) / max(output_rows, 1)

        rows: list[tuple[str, str]] = [
            ("Status", status),
            ("Output rows", f"{output_rows:,}"),
        ]
        if cached > 0:
            rows.append(("Cached rows", f"{cached:,} (no LLM cost)"))
        rows.append(("LLM calls", f"{llm_calls:,}"))
        if metrics.failed_rows:
            rows.append(("Failed rows", f"{metrics.failed_rows:,}"))
        if metrics.skipped_rows:
            rows.append(("Skipped rows", f"{metrics.skipped_rows:,}"))
        rows.append(("Duration", f"{dur_str} ({metrics.rows_per_second:.1f} rows/sec)"))
        rows.append(("", ""))
        rows.append(("Total cost", f"${costs.total_cost:.4f} (${cpr:.6f}/row)"))
        rows.append(("Input tokens", f"{costs.input_tokens:,}"))
        rows.append(("Output tokens", f"{costs.output_tokens:,}"))
        rows.append(("Total tokens", f"{costs.total_tokens:,}"))

        key_w = max(len(k) for k, _ in rows if k)
        val_w = max(len(v) for _, v in rows if v)
        inner = key_w + 3 + val_w
        border = "─" * (inner + 2)

        lines = [f"┌{border}┐", f"│{'Pipeline Report':^{inner + 2}}│", f"├{border}┤"]
        for key, val in rows:
            if not key and not val:
                lines.append(f"├{border}┤")
            else:
                lines.append(f"│ {key:<{key_w}} : {val:<{val_w}} │")
        lines.append(f"└{border}┘")

        for line in lines:
            self.logger.info(line)

    @staticmethod
    def _base_label(deployment: dict[str, Any], dep_id: str) -> str:
        """Extract a concise display label from deployment metadata."""
        if "label" in deployment:
            return deployment["label"]
        model = deployment.get("model", "")
        if model:
            provider = model.split("/")[0] if "/" in model else ""
            model_short = model.split("/")[1] if "/" in model else model
            if len(model_short) > 25:
                model_short = model_short[:22] + "..."
            return f"{dep_id} ({provider}/{model_short})"
        return dep_id

    def _format_deployment_lines(self, task_id: str, elapsed: float) -> list[str]:
        """Build per-endpoint scoreboard lines (one per logger.info call)."""
        dep_stats = self._deployments.get(task_id, {})
        if not dep_stats:
            return []

        total = sum(dep_stats.values())
        if total <= 0:
            return []

        labels = self._deployment_labels.get(task_id, {})
        ranked = sorted(dep_stats.items(), key=lambda item: (-item[1], item[0]))

        if len(ranked) == 1:
            dep_id, count = ranked[0]
            label = labels.get(dep_id, dep_id)
            rps = count / max(elapsed, 0.01)
            return [f"► {label}  {count:,} rows  {rps:.0f}/s"]

        w = max(len(labels.get(d, d)) for d, _ in ranked)
        w = max(w, 8)

        lines: list[str] = []
        for dep_id, count in ranked:
            label = labels.get(dep_id, dep_id)
            share = (count / total) * 100
            rps = count / max(elapsed, 0.01)
            lines.append(
                f"► {label:<{w}}  {count:>8,} rows  {rps:>5.0f}/s  {share:>4.0f}%"
            )

        return lines

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Render durations in short human-readable form."""
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        m, s = divmod(s, 60)
        if m < 60:
            return f"{m}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m{s:02d}s"

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
            - "rich": Use rich.progress (split-panel UI)
            - "textual": Use Textual TUI (interactive scrollable logs)
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

    elif mode == "textual":
        try:
            from textual.app import App  # noqa: F401

            return TextualProgressTracker()
        except ImportError:
            raise ImportError(
                "textual is required for mode='textual'. "
                "Install with: pip install ondine[tui]"
            )

    elif mode == "tqdm":
        raise NotImplementedError(
            "tqdm tracker not yet implemented, use 'rich' or 'logging'"
        )

    elif mode == "logging":
        return LoggingProgressTracker()

    else:
        raise ValueError(
            f"Invalid progress mode: {mode}. "
            f"Use 'auto', 'rich', 'textual', 'tqdm', 'logging', or 'none'"
        )


class NoOpProgressTracker(ProgressTracker):
    """No-op tracker that does nothing (for disabling progress)."""

    def start_stage(self, stage_name: str, total_rows: int, **metadata: Any) -> str:
        return stage_name

    def update(self, task_id: str, advance: int = 1, **metadata: Any) -> None:
        pass

    def finish(self, task_id: str) -> None:
        pass

    def ensure_deployment_task(
        self, stage_name: str, deployment_id: str, total_rows: int, label_info: str = ""
    ) -> None:
        pass

    def __enter__(self) -> "NoOpProgressTracker":
        return self

    def __exit__(self, *args: Any) -> None:
        pass
