"""Textual TUI application for pipeline progress display.

Provides a full-screen terminal UI with:
- Fixed progress panel (top) using Rich Progress bars
- Scrollable log panel (bottom) with native scrollbar via RichLog
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import RichLog, Static

if TYPE_CHECKING:
    import threading

_APP_CSS = """
#progress-panel {
    height: auto;
    max-height: 50%;
    padding: 0 1;
}

#log-panel {
    height: 1fr;
    border-top: solid $accent;
}
"""


class PipelineApp(App[None]):
    """Textual app with a fixed progress panel and a scrollable log panel."""

    CSS = _APP_CSS
    TITLE = "Ondine Pipeline"

    def __init__(self, progress: Any, ready_event: threading.Event) -> None:
        super().__init__()
        self._progress = progress
        self._ready_event = ready_event

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self._progress, id="progress-panel")
            yield RichLog(
                highlight=True,
                markup=True,
                auto_scroll=True,
                id="log-panel",
            )

    def on_mount(self) -> None:
        self._ready_event.set()

    def refresh_progress_widget(self) -> None:
        with contextlib.suppress(Exception):
            widget = self.query_one("#progress-panel", Static)
            widget.update(self._progress)

    def add_log_line(self, line: str) -> None:
        with contextlib.suppress(Exception):
            log_widget = self.query_one("#log-panel", RichLog)
            style = "dim"
            lower = line.lower()
            if "error" in lower:
                style = "red"
            elif "warning" in lower or "skipping" in lower:
                style = "yellow"
            from rich.text import Text

            log_widget.write(Text(line, style=style))
