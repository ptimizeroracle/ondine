"""Tests for ProgressReporter."""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from ondine.orchestration.deployment_tracker import DeploymentTracker
from ondine.orchestration.progress_reporter import ProgressReporter


class TestProgressReporter:
    """Test ProgressReporter functionality."""

    def test_init_without_tracker(self):
        """Test initialization without progress tracker."""
        reporter = ProgressReporter(tracker=None)
        assert not reporter.is_active
        assert reporter.total_rows == 0

    def test_init_with_tracker(self):
        """Test initialization with progress tracker."""
        mock_tracker = MagicMock()
        reporter = ProgressReporter(tracker=mock_tracker)
        assert reporter._tracker is mock_tracker

    def test_start_without_tracker(self):
        """Test start does nothing without tracker."""
        reporter = ProgressReporter(tracker=None)
        reporter.start("TestStage", total_rows=100)
        # Should not raise, just no-op
        assert not reporter.is_active

    def test_start_with_tracker(self):
        """Test start initializes progress tracking."""
        mock_tracker = MagicMock()
        mock_tracker.start_stage.return_value = "task-123"

        reporter = ProgressReporter(tracker=mock_tracker)
        reporter.start("LLMInvocation", total_rows=1000, deployments=[{"model_id": "test"}])

        mock_tracker.start_stage.assert_called_once()
        assert reporter.is_active
        assert reporter.total_rows == 1000

    def test_update_without_tracker(self):
        """Test update does nothing without tracker."""
        reporter = ProgressReporter(tracker=None)
        reporter.update(rows_completed=10, cost=Decimal("0.01"))
        # Should not raise

    def test_update_with_tracker(self):
        """Test update calls tracker methods."""
        mock_tracker = MagicMock()
        mock_tracker.start_stage.return_value = "task-123"

        reporter = ProgressReporter(tracker=mock_tracker)
        reporter.start("Test", total_rows=100)
        reporter.update(rows_completed=5, cost=Decimal("0.05"))

        mock_tracker.update.assert_called_once()
        call_kwargs = mock_tracker.update.call_args[1]
        assert call_kwargs["advance"] == 5
        assert call_kwargs["cost"] == Decimal("0.05")

    def test_update_with_deployment_tracking(self):
        """Test update with deployment ID tracking."""
        mock_tracker = MagicMock()
        mock_tracker.start_stage.return_value = "task-123"

        model_list = [
            {"model_id": "fast", "litellm_params": {"model": "groq/model"}},
        ]
        deployment_tracker = DeploymentTracker(model_list)

        reporter = ProgressReporter(
            tracker=mock_tracker, deployment_tracker=deployment_tracker
        )
        reporter.start("Test", total_rows=100)
        reporter.update(
            rows_completed=1,
            cost=Decimal("0.01"),
            deployment_id="hash-xyz",
        )

        # Should register deployment and call ensure_deployment_task
        mock_tracker.ensure_deployment_task.assert_called_once()
        mock_tracker.update.assert_called_once()

        # Deployment should be registered
        assert deployment_tracker.get_friendly_id("hash-xyz") == "fast_0"

    def test_finish_without_tracker(self):
        """Test finish does nothing without tracker."""
        reporter = ProgressReporter(tracker=None)
        reporter.finish()
        # Should not raise

    def test_finish_with_tracker(self):
        """Test finish calls tracker.finish()."""
        mock_tracker = MagicMock()
        mock_tracker.start_stage.return_value = "task-123"

        reporter = ProgressReporter(tracker=mock_tracker)
        reporter.start("Test", total_rows=100)
        reporter.finish()

        mock_tracker.finish.assert_called_once_with("task-123")
        assert not reporter.is_active

    def test_repr(self):
        """Test string representation."""
        reporter = ProgressReporter(tracker=None)
        repr_str = repr(reporter)
        assert "ProgressReporter" in repr_str
        assert "active=False" in repr_str

        mock_tracker = MagicMock()
        mock_tracker.start_stage.return_value = "task"
        reporter_active = ProgressReporter(tracker=mock_tracker)
        reporter_active.start("Test", total_rows=50)

        repr_str = repr(reporter_active)
        assert "active=True" in repr_str
        assert "total_rows=50" in repr_str

