"""
Execution observers for monitoring and logging.

Implements Observer pattern for decoupled event notification.
"""

from abc import ABC, abstractmethod
from typing import Any

from tqdm import tqdm

from ondine.core.models import ExecutionResult
from ondine.orchestration.execution_context import ExecutionContext
from ondine.stages.pipeline_stage import PipelineStage
from ondine.utils import get_logger


class ExecutionObserver(ABC):
    """
    Abstract base for execution observers.

    Observers can monitor pipeline execution without coupling
    to the executor implementation.
    """

    @abstractmethod
    def on_pipeline_start(self, pipeline: Any, context: ExecutionContext) -> None:
        """Called before first stage execution."""
        pass

    @abstractmethod
    def on_stage_start(self, stage: PipelineStage, context: ExecutionContext) -> None:
        """Called before each stage."""
        pass

    @abstractmethod
    def on_stage_complete(
        self, stage: PipelineStage, context: ExecutionContext, result: Any
    ) -> None:
        """Called after successful stage completion."""
        pass

    @abstractmethod
    def on_stage_error(
        self, stage: PipelineStage, context: ExecutionContext, error: Exception
    ) -> None:
        """Called on stage failure."""
        pass

    @abstractmethod
    def on_pipeline_complete(
        self, context: ExecutionContext, result: ExecutionResult
    ) -> None:
        """Called after all stages complete."""
        pass

    @abstractmethod
    def on_pipeline_error(self, context: ExecutionContext, error: Exception) -> None:
        """Called on fatal pipeline failure."""
        pass

    def on_progress_update(self, context: ExecutionContext) -> None:
        """Called periodically during execution for progress updates."""
        pass


class ProgressBarObserver(ExecutionObserver):
    """Observer that displays progress bar with tqdm."""

    def __init__(self):
        """Initialize progress bar observer."""
        self.progress_bar: tqdm | None = None

    def on_pipeline_start(self, pipeline: Any, context: ExecutionContext) -> None:
        """Initialize progress bar."""
        if context.total_rows > 0:
            self.progress_bar = tqdm(
                total=context.total_rows,
                desc="Processing",
                unit="rows",
            )

    def on_stage_start(self, stage: PipelineStage, context: ExecutionContext) -> None:
        """Update progress bar description."""
        if self.progress_bar:
            self.progress_bar.set_description(f"Stage: {stage.name}")

    def on_stage_complete(
        self, stage: PipelineStage, context: ExecutionContext, result: Any
    ) -> None:
        """Update progress bar."""
        if self.progress_bar:
            self.progress_bar.n = context.last_processed_row
            self.progress_bar.refresh()

    def on_stage_error(
        self, stage: PipelineStage, context: ExecutionContext, error: Exception
    ) -> None:
        """Handle error in progress bar."""
        if self.progress_bar:
            self.progress_bar.set_description(f"Error in {stage.name}")

    def on_pipeline_complete(
        self, context: ExecutionContext, result: ExecutionResult
    ) -> None:
        """Close progress bar."""
        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None

    def on_pipeline_error(self, context: ExecutionContext, error: Exception) -> None:
        """Close progress bar on error."""
        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None

    def on_progress_update(self, context: ExecutionContext) -> None:
        """Update progress bar with current row count."""
        if self.progress_bar:
            self.progress_bar.n = context.last_processed_row
            self.progress_bar.set_postfix(
                {
                    "cost": f"${context.accumulated_cost:.4f}",
                    "progress": f"{context.get_progress():.1f}%",
                }
            )
            self.progress_bar.refresh()


class LoggingObserver(ExecutionObserver):
    """Observer that logs execution events."""

    def __init__(self):
        """Initialize logging observer."""
        self.logger = get_logger(__name__)

    def on_pipeline_start(self, pipeline: Any, context: ExecutionContext) -> None:
        """Log pipeline start."""
        self.logger.info(f"Pipeline execution started (session: {context.session_id})")

    def on_stage_start(self, stage: PipelineStage, context: ExecutionContext) -> None:
        """Log stage start."""
        self.logger.info(
            f"Starting stage: {stage.name} (progress: {context.get_progress():.1f}%)"
        )

    def on_stage_complete(
        self, stage: PipelineStage, context: ExecutionContext, result: Any
    ) -> None:
        """Log stage completion."""
        self.logger.info(
            f"Completed stage: {stage.name} (cost: ${context.accumulated_cost:.4f})"
        )

    def on_stage_error(
        self, stage: PipelineStage, context: ExecutionContext, error: Exception
    ) -> None:
        """Log stage error."""
        self.logger.error(f"Stage {stage.name} failed: {error}")

    def on_pipeline_complete(
        self, context: ExecutionContext, result: ExecutionResult
    ) -> None:
        """Log pipeline completion."""
        self.logger.info(
            f"Pipeline execution completed successfully\n"
            f"  Processed: {result.metrics.processed_rows} rows\n"
            f"  Duration: {result.metrics.total_duration_seconds:.2f}s\n"
            f"  Total cost: ${result.costs.total_cost:.4f}\n"
            f"  Errors: {result.metrics.failed_rows}"
        )

    def on_pipeline_error(self, context: ExecutionContext, error: Exception) -> None:
        """Log pipeline error."""
        self.logger.error(f"Pipeline execution failed: {error}")

    def on_progress_update(self, context: ExecutionContext) -> None:
        """Log progress update."""
        # Make progress very visible with separators
        self.logger.info(
            f"━━━━━━ PROGRESS: {context.last_processed_row}/{context.total_rows} rows "
            f"({context.get_progress():.1f}%) | Cost: ${context.accumulated_cost:.4f} ━━━━━━"
        )


class CostTrackingObserver(ExecutionObserver):
    """Observer that tracks and warns about costs."""

    def __init__(self, warning_threshold: float = 0.75):
        """
        Initialize cost tracking observer.

        Args:
            warning_threshold: Warn when this fraction of budget used
        """
        self.logger = get_logger(__name__)
        self.warning_threshold = warning_threshold
        self.max_budget: float | None = None

    def on_pipeline_start(self, pipeline: Any, context: ExecutionContext) -> None:
        """Set max budget if available."""
        # Could extract from pipeline specs
        pass

    def on_stage_start(self, stage: PipelineStage, context: ExecutionContext) -> None:
        """No action on stage start."""
        pass

    def on_stage_complete(
        self, stage: PipelineStage, context: ExecutionContext, result: Any
    ) -> None:
        """Check cost after stage completion."""
        if self.max_budget:
            usage_ratio = float(context.accumulated_cost) / self.max_budget

            if usage_ratio >= self.warning_threshold:
                self.logger.warning(
                    f"Cost warning: {usage_ratio * 100:.1f}% of budget used "
                    f"(${context.accumulated_cost:.4f} / ${self.max_budget:.2f})"
                )

    def on_stage_error(
        self, stage: PipelineStage, context: ExecutionContext, error: Exception
    ) -> None:
        """No action on error."""
        pass

    def on_pipeline_complete(
        self, context: ExecutionContext, result: ExecutionResult
    ) -> None:
        """Log final cost summary."""
        self.logger.info(
            f"Cost summary:\n"
            f"  Total: ${result.costs.total_cost:.4f}\n"
            f"  Input tokens: {result.costs.input_tokens:,}\n"
            f"  Output tokens: {result.costs.output_tokens:,}\n"
            f"  Cost per row: ${float(result.costs.total_cost) / result.metrics.total_rows:.6f}"
        )

    def on_pipeline_error(self, context: ExecutionContext, error: Exception) -> None:
        """Log cost at failure."""
        self.logger.info(f"Cost at failure: ${context.accumulated_cost:.4f}")

    def on_progress_update(self, context: ExecutionContext) -> None:
        """No action on progress update for cost tracking."""
        pass
