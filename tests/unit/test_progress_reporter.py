"""Tests for ProgressReporter and shared RunProgressState."""

from decimal import Decimal
from unittest.mock import MagicMock

from ondine.orchestration.deployment_tracker import DeploymentTracker
from ondine.orchestration.execution_context import RunProgressState
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
        reporter.start(
            "LLMInvocation", total_rows=1000, deployments=[{"model_id": "test"}]
        )

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


class TestRunProgressState:
    """Verify RunProgressState as single source of truth for live cost."""

    def test_init_defaults(self):
        state = RunProgressState()
        assert state.total_cost == Decimal("0")
        assert state.total_tokens == 0
        assert state.input_tokens == 0
        assert state.output_tokens == 0

    def test_init_stage_and_apply_delta(self):
        state = RunProgressState()
        state.init_stage("LLMInvocation", total_rows=100)

        state.apply_delta(
            "LLMInvocation",
            rows_completed=10,
            cost=Decimal("0.05"),
            tokens_in=500,
            tokens_out=200,
        )

        assert state.total_cost == Decimal("0.05")
        assert state.total_tokens == 700
        assert state.input_tokens == 500
        assert state.output_tokens == 200

        snap = state.get_stage("LLMInvocation")
        assert snap is not None
        assert snap.rows_completed == 10
        assert snap.cost == Decimal("0.05")

    def test_multiple_deltas_accumulate(self):
        state = RunProgressState()
        state.init_stage("LLMInvocation", total_rows=100)

        for _ in range(5):
            state.apply_delta(
                "LLMInvocation",
                rows_completed=2,
                cost=Decimal("0.01"),
                tokens_in=100,
                tokens_out=50,
            )

        assert state.total_cost == Decimal("0.05")
        assert state.total_tokens == 750
        snap = state.get_stage("LLMInvocation")
        assert snap.rows_completed == 10

    def test_deployment_tracking(self):
        state = RunProgressState()
        state.init_stage("LLM", total_rows=50)

        state.apply_delta(
            "LLM", rows_completed=5, cost=Decimal("0.02"), deployment_id="dep-a"
        )
        state.apply_delta(
            "LLM", rows_completed=3, cost=Decimal("0.01"), deployment_id="dep-b"
        )
        state.apply_delta(
            "LLM", rows_completed=2, cost=Decimal("0.01"), deployment_id="dep-a"
        )

        snap = state.get_stage("LLM")
        assert snap.deployment_rows == {"dep-a": 7, "dep-b": 3}
        assert snap.deployment_costs["dep-a"] == Decimal("0.03")
        assert snap.deployment_costs["dep-b"] == Decimal("0.01")
        assert state.total_cost == Decimal("0.04")

    def test_snapshot_cost_property(self):
        state = RunProgressState()
        state.init_stage("S", total_rows=10)
        state.apply_delta("S", cost=Decimal("0.1"))
        assert state.snapshot_cost == Decimal("0.1")

    def test_get_stage_returns_copy(self):
        state = RunProgressState()
        state.init_stage("S", total_rows=10)
        state.apply_delta("S", rows_completed=5)

        snap = state.get_stage("S")
        snap.rows_completed = 999
        real = state.get_stage("S")
        assert real.rows_completed == 5

    def test_get_stage_nonexistent_returns_none(self):
        state = RunProgressState()
        assert state.get_stage("missing") is None

    def test_apply_delta_to_nonexistent_stage_is_noop(self):
        state = RunProgressState()
        state.apply_delta("missing", rows_completed=5, cost=Decimal("1"))
        assert state.total_cost == Decimal("0")

    def test_none_cost_does_not_add(self):
        state = RunProgressState()
        state.init_stage("S", total_rows=10)
        state.apply_delta("S", rows_completed=1, cost=None)
        assert state.total_cost == Decimal("0")


class TestReporterWithSharedState:
    """Integration: ProgressReporter writes to RunProgressState before tracker."""

    def test_update_writes_to_shared_state_and_tracker(self):
        mock_tracker = MagicMock()
        mock_tracker.start_stage.return_value = "task-1"

        run_progress = RunProgressState()

        reporter = ProgressReporter(tracker=mock_tracker, run_progress=run_progress)
        reporter.start("LLMInvocation", total_rows=100)
        reporter.update(
            rows_completed=10,
            cost=Decimal("0.05"),
            tokens_in=500,
            tokens_out=200,
        )

        assert run_progress.total_cost == Decimal("0.05")
        assert run_progress.total_tokens == 700
        snap = run_progress.get_stage("LLMInvocation")
        assert snap.rows_completed == 10

        mock_tracker.update.assert_called_once()

    def test_start_inits_stage_in_shared_state(self):
        mock_tracker = MagicMock()
        mock_tracker.start_stage.return_value = "task-1"
        run_progress = RunProgressState()

        reporter = ProgressReporter(tracker=mock_tracker, run_progress=run_progress)
        reporter.start("LLMInvocation", total_rows=200)

        snap = run_progress.get_stage("LLMInvocation")
        assert snap is not None
        assert snap.total_rows == 200

    def test_reporter_without_shared_state_still_works(self):
        mock_tracker = MagicMock()
        mock_tracker.start_stage.return_value = "task-1"

        reporter = ProgressReporter(tracker=mock_tracker)
        reporter.start("Test", total_rows=50)
        reporter.update(rows_completed=5, cost=Decimal("0.01"))

        mock_tracker.update.assert_called_once()

    def test_logging_tracker_reads_shared_cost(self):
        from ondine.orchestration.progress_tracker import LoggingProgressTracker

        run_progress = RunProgressState()
        tracker = LoggingProgressTracker()
        tracker.set_run_progress(run_progress)

        task_id = tracker.start_stage("LLM: 100 rows", total_rows=100)

        run_progress.init_stage("LLM", total_rows=100)
        run_progress.apply_delta("LLM", rows_completed=100, cost=Decimal("0.08"))

        tracker.update(task_id, advance=100, cost=Decimal("0.08"))

        assert run_progress.snapshot_cost == Decimal("0.08")


class _FakeTime:
    """Deterministic monotonic clock for logging tracker tests."""

    def __init__(self, values):
        self._values = iter(values)

    def monotonic(self):
        return next(self._values)


class TestLoggingProgressTracker:
    """Focused tests for pure logging progress formatting."""

    def test_logs_router_breakdown_for_multiple_endpoints(self):
        from ondine.orchestration.progress_tracker import LoggingProgressTracker

        tracker = LoggingProgressTracker()
        tracker.logger = MagicMock()
        tracker._time = _FakeTime([0.0, 1.0])

        task_id = tracker.start_stage(
            "LLMInvocation: 100 rows",
            total_rows=100,
            deployments=[
                {"model_id": "swap-reviewer-east-us", "label": "East US"},
                {"model_id": "swap-reviewer-france", "label": "France"},
            ],
        )

        tracker.update(task_id, advance=50, deployment_id="swap-reviewer-east-us")

        messages = [call.args[0] for call in tracker.logger.info.call_args_list]
        assert any(
            "[progress] LLMInvocation: 100 rows | 50.0%" in msg for msg in messages
        )
        scoreboard = [m for m in messages if "East US" in m and "►" in m]
        assert len(scoreboard) == 1
        assert "France" in scoreboard[0]
        assert "0%" in scoreboard[0]

    def test_logs_single_api_breakdown_for_one_endpoint(self):
        from ondine.orchestration.progress_tracker import LoggingProgressTracker

        tracker = LoggingProgressTracker()
        tracker.logger = MagicMock()
        tracker._time = _FakeTime([0.0, 1.0])

        task_id = tracker.start_stage("LLMInvocation: 100 rows", total_rows=100)
        tracker.ensure_deployment_task(
            task_id,
            "swap-reviewer-france",
            total_rows=100,
            label_info="France",
        )

        tracker.update(task_id, advance=10, deployment_id="swap-reviewer-france")

        messages = [call.args[0] for call in tracker.logger.info.call_args_list]
        assert any(
            "[progress] LLMInvocation: 100 rows | 10.0%" in msg for msg in messages
        )
        assert any("► France" in msg for msg in messages)
