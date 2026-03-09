"""Tests for TextualProgressTracker and factory registration."""

import sys
from unittest.mock import patch

import pytest

from ondine.orchestration.progress_tracker import (
    TextualProgressTracker,
    create_progress_tracker,
)

try:
    import textual  # noqa: F401

    _has_textual = True
except ModuleNotFoundError:
    _has_textual = False

_skip_no_textual = pytest.mark.skipif(
    not _has_textual, reason="textual not installed (optional dependency)"
)


class TestTextualProgressTrackerFactory:
    """Factory returns TextualProgressTracker for mode='textual'."""

    @_skip_no_textual
    def test_factory_returns_textual_tracker(self):
        tracker = create_progress_tracker("textual")
        assert isinstance(tracker, TextualProgressTracker)

    def test_factory_raises_on_missing_textual(self):
        with (
            patch.dict("sys.modules", {"textual": None, "textual.app": None}),
            pytest.raises(ImportError, match="textual is required"),
        ):
            create_progress_tracker("textual")


class TestTextualProgressTrackerLifecycle:
    """Test the tracker interface without launching the full Textual app."""

    def _make_tracker(self) -> TextualProgressTracker:
        return TextualProgressTracker()

    def test_start_stage_registers_task(self):
        tracker = self._make_tracker()
        task_id = tracker.start_stage("LLMInvocation: 100 rows", total_rows=100)
        assert task_id == "LLMInvocation: 100 rows"
        assert task_id in tracker.tasks

    def test_start_stage_with_deployments(self):
        tracker = self._make_tracker()
        deployments = [
            {"model_id": "sweden", "model": "azure/gpt-5-nano"},
            {"model_id": "france", "model": "azure/gpt-5-nano"},
        ]
        task_id = tracker.start_stage("Stage", total_rows=200, deployments=deployments)
        assert "sweden" in tracker.deployment_tasks[task_id]
        assert "france" in tracker.deployment_tasks[task_id]
        assert tracker.deployment_stats[task_id]["sweden"] == 0

    def test_update_advances_progress(self):
        tracker = self._make_tracker()
        task_id = tracker.start_stage("Stage", total_rows=50)
        tracker.update(task_id, advance=10, cost=0.05)

        rich_task_id = tracker.tasks[task_id]
        task = tracker.progress.tasks[rich_task_id]
        assert task.completed == 10
        assert task.fields["cost"] == pytest.approx(0.05)

    def test_update_with_deployment(self):
        tracker = self._make_tracker()
        task_id = tracker.start_stage("Stage", total_rows=100)
        tracker.ensure_deployment_task(task_id, "dep-1", total_rows=100)

        tracker.update(task_id, advance=5, deployment_id="dep-1", cost=0.02)
        assert tracker.deployment_stats[task_id]["dep-1"] == 5

    def test_ensure_deployment_task_idempotent(self):
        tracker = self._make_tracker()
        task_id = tracker.start_stage("Stage", total_rows=100)
        tracker.ensure_deployment_task(task_id, "dep-1", total_rows=100)
        tracker.ensure_deployment_task(task_id, "dep-1", total_rows=100)
        assert len(tracker.deployment_tasks[task_id]) == 1

    def test_ensure_deployment_task_with_label(self):
        tracker = self._make_tracker()
        task_id = tracker.start_stage("Stage", total_rows=100)
        tracker.ensure_deployment_task(
            task_id, "dep-1", total_rows=100, label_info="Sweden (gpt-5-nano)"
        )
        assert "dep-1" in tracker.deployment_tasks[task_id]

    def test_finish_completes_task(self):
        tracker = self._make_tracker()
        task_id = tracker.start_stage("Stage", total_rows=50)
        tracker.update(task_id, advance=30)
        tracker.finish(task_id)

        rich_task_id = tracker.tasks[task_id]
        task = tracker.progress.tasks[rich_task_id]
        assert task.completed == 50

    def test_finish_preserves_deployment_actual_counts(self):
        tracker = self._make_tracker()
        task_id = tracker.start_stage("Stage", total_rows=100)
        tracker.ensure_deployment_task(task_id, "d1", total_rows=100)
        tracker.ensure_deployment_task(task_id, "d2", total_rows=100)

        tracker.update(task_id, advance=30, deployment_id="d1")
        tracker.update(task_id, advance=70, deployment_id="d2")
        tracker.finish(task_id)

        d1_task = tracker.progress.tasks[tracker.deployment_tasks[task_id]["d1"]]
        d2_task = tracker.progress.tasks[tracker.deployment_tasks[task_id]["d2"]]
        assert d1_task.completed == 30
        assert d2_task.completed == 70

    def test_update_noop_for_unknown_task(self):
        tracker = self._make_tracker()
        tracker.update("nonexistent", advance=1)

    def test_finish_noop_for_unknown_task(self):
        tracker = self._make_tracker()
        tracker.finish("nonexistent")


@_skip_no_textual
class TestTextualProgressTrackerContextManager:
    """Test __enter__ / __exit__ with the Textual app (headless in CI)."""

    def test_enter_exit_lifecycle(self):
        """App starts in a thread and shuts down cleanly."""
        tracker = TextualProgressTracker()
        tracker.__enter__()
        try:
            assert tracker._app is not None
            assert tracker._app_thread is not None
        finally:
            tracker.__exit__(None, None, None)
        assert tracker._app is None

    def test_context_manager_protocol(self):
        """Full lifecycle through with-statement works end-to-end."""
        tracker = TextualProgressTracker()
        with tracker:
            assert tracker._app is not None
            task_id = tracker.start_stage("Test", total_rows=10)
            tracker.update(task_id, advance=5, cost=0.01)
            tracker.finish(task_id)
        assert tracker._app is None

    def test_stdout_captured_during_context(self):
        """Stdout writes inside the context go through _TextualStdoutCapture."""
        tracker = TextualProgressTracker()
        with tracker:
            assert type(sys.stdout).__name__ == "_TextualStdoutCapture"
        assert type(sys.stdout).__name__ != "_TextualStdoutCapture"

    def test_double_exit_safe(self):
        """Calling __exit__ twice doesn't raise."""
        tracker = TextualProgressTracker()
        with tracker:
            pass
        tracker.__exit__(None, None, None)


@_skip_no_textual
class TestPipelineApp:
    """Test the Textual App widget interface."""

    def test_app_has_required_methods(self):
        """PipelineApp exposes refresh_progress_widget and add_log_line."""
        import threading

        from rich.progress import Progress

        from ondine.orchestration._pipeline_tui import PipelineApp

        app = PipelineApp(Progress(), threading.Event())
        assert hasattr(app, "refresh_progress_widget")
        assert hasattr(app, "add_log_line")
        assert callable(app.refresh_progress_widget)
        assert callable(app.add_log_line)
