"""Textual TUI application for pipeline progress display.

Provides a full-screen terminal UI with:
- Header with app title
- Fixed progress panel (top) with bordered container
- Scrollable log panel (bottom) with native scrollbar via RichLog
- Footer with keybindings

Threading: the app runs in a daemon thread. A signal.signal patch is
installed so Textual's LinuxDriver doesn't crash from a non-main thread.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Footer, Header, RichLog, Static

if TYPE_CHECKING:
    import threading

_APP_CSS = """
Screen {
    layout: vertical;
}

#progress-container {
    height: auto;
    max-height: 40%;
    border: round $primary;
    padding: 1 2;
    margin: 1 1 0 1;
}

#progress-bars {
    height: auto;
}

#log-container {
    height: 1fr;
    border: round $accent;
    margin: 1 1 0 1;
}

#log-panel {
    height: 1fr;
    scrollbar-size: 1 1;
}
"""


def _install_thread_safe_signal_patch() -> None:
    """Patch signal.signal to no-op when called from a non-main thread.

    Textual's LinuxDriver calls signal.signal() in __init__,
    start_application_mode, and stop_application_mode.  CPython forbids
    signal registration outside the main thread, so we wrap the function
    to silently skip those calls instead of raising ValueError.

    The patch is idempotent and only applied once.
    """
    import signal as _signal
    import threading

    if getattr(_signal.signal, "_ondine_patched", False):
        return

    _original = _signal.signal

    def _safe_signal(signalnum: Any, handler: Any) -> Any:
        if threading.current_thread() is not threading.main_thread():
            return _signal.getsignal(signalnum)
        return _original(signalnum, handler)

    _safe_signal._ondine_patched = True  # type: ignore[attr-defined]
    _signal.signal = _safe_signal  # type: ignore[assignment]


class PipelineApp(App[None]):
    """Textual app with a fixed progress panel and a scrollable log panel."""

    CSS = _APP_CSS
    TITLE = "Ondine Pipeline"
    SUB_TITLE = "LLM Dataset Engine"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("s", "toggle_scroll", "Auto-scroll", show=True),
    ]

    def __init__(
        self,
        progress: Any,
        ready_event: threading.Event,
    ) -> None:
        super().__init__()
        self._progress = progress
        self._ready_event = ready_event
        self._auto_scroll = True

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Vertical(id="progress-container"):
            yield Static(self._progress, id="progress-bars")

        with VerticalScroll(id="log-container"):
            yield RichLog(
                highlight=True,
                markup=True,
                auto_scroll=True,
                max_lines=2000,
                id="log-panel",
            )

        yield Footer()

    def on_mount(self) -> None:
        self._ready_event.set()

    def action_toggle_scroll(self) -> None:
        log = self.query_one("#log-panel", RichLog)
        self._auto_scroll = not self._auto_scroll
        log.auto_scroll = self._auto_scroll
        state = "ON" if self._auto_scroll else "OFF"
        self.notify(f"Auto-scroll: {state}", timeout=2)

    def refresh_progress_widget(self) -> None:
        with contextlib.suppress(Exception):
            widget = self.query_one("#progress-bars", Static)
            widget.update(self._progress)

    def add_log_line(self, line: str) -> None:
        with contextlib.suppress(Exception):
            log_widget = self.query_one("#log-panel", RichLog)
            from rich.text import Text

            style = "dim"
            lower = line.lower()
            if "error" in lower:
                style = "bold red"
            elif "warning" in lower or "skipping" in lower:
                style = "yellow"
            elif "completed" in lower or "✓" in line:
                style = "green"

            log_widget.write(Text(line, style=style))
