"""
Tracing observer for pipeline execution.

Implements ExecutionObserver interface to create OpenTelemetry spans
for pipeline and stage execution.
"""

from typing import Any

from opentelemetry import trace  # noqa: TC002
from opentelemetry.trace import Status, StatusCode

from ondine.core.models import ExecutionResult
from ondine.orchestration.execution_context import ExecutionContext
from ondine.orchestration.observers import ExecutionObserver
from ondine.stages.pipeline_stage import PipelineStage

from .tracer import get_tracer, is_tracing_enabled


class TracingObserver(ExecutionObserver):
    """
    Observer that creates OpenTelemetry spans for pipeline execution.

    Single Responsibility: Create and manage trace spans for observability.

    The observer creates a hierarchical span structure:
    - Root span for pipeline execution
    - Nested spans for each stage
    - Attributes include metrics, errors, and metadata

    Args:
        include_prompts: If True, include prompts in spans (PII risk)

    Examples:
        >>> from ondine.observability import TracingObserver, enable_tracing
        >>> enable_tracing(exporter="console")
        >>> observer = TracingObserver(include_prompts=False)
        >>> # Attach to pipeline execution
    """

    def __init__(self, include_prompts: bool = False):
        """Initialize tracing observer."""
        self._include_prompts = include_prompts
        self._spans: dict[str, trace.Span] = {}  # Track active spans

    def on_pipeline_start(self, pipeline: Any, context: ExecutionContext) -> None:
        """
        Create root span for pipeline execution.

        Args:
            pipeline: Pipeline instance
            context: Execution context with total rows, etc.
        """
        if not is_tracing_enabled():
            return

        tracer = get_tracer()

        # Create root span
        span = tracer.start_span("pipeline.execute")

        # Add attributes
        span.set_attribute("ondine.total_rows", context.total_rows)
        span.set_attribute("ondine.session_id", context.session_id)

        # Store span for later
        self._spans["pipeline"] = span

    def on_stage_start(self, stage: PipelineStage, context: ExecutionContext) -> None:
        """
        Create span for stage execution.

        Args:
            stage: Pipeline stage being executed
            context: Current execution context
        """
        if not is_tracing_enabled():
            return

        tracer = get_tracer()
        stage_name = stage.__class__.__name__

        # Create nested span under pipeline span
        # Note: OpenTelemetry automatically handles span context
        span = tracer.start_span(f"stage.{stage_name}")

        # Add stage-specific attributes
        span.set_attribute("ondine.stage", stage_name)
        span.set_attribute("ondine.processed_rows", context.last_processed_row + 1)

        # Add prompt if stage has it (and sanitize based on flag)
        if hasattr(stage, "prompt_template") and stage.prompt_template:
            from .sanitizer import sanitize_prompt

            prompt_value = sanitize_prompt(
                stage.prompt_template, include_prompts=self._include_prompts
            )
            span.set_attribute("ondine.prompt", prompt_value)

        # Store span for completion/error handling
        self._spans[stage_name] = span

    def on_stage_complete(
        self, stage: PipelineStage, context: ExecutionContext, result: Any
    ) -> None:
        """
        Close stage span with success attributes.

        Args:
            stage: Pipeline stage that completed
            context: Execution context after stage
            result: Stage execution result
        """
        if not is_tracing_enabled():
            return

        stage_name = stage.__class__.__name__
        span = self._spans.get(stage_name)

        if span is not None:
            # Mark as successful
            span.set_status(Status(StatusCode.OK))

            # Add completion metrics
            span.set_attribute("ondine.rows_processed", context.last_processed_row + 1)

            # End span
            span.end()

            # Remove from active spans
            self._spans.pop(stage_name, None)

    def on_stage_error(
        self, stage: PipelineStage, context: ExecutionContext, error: Exception
    ) -> None:
        """
        Close stage span with error details.

        Args:
            stage: Pipeline stage that failed
            context: Execution context at failure
            error: Exception that occurred
        """
        if not is_tracing_enabled():
            return

        stage_name = stage.__class__.__name__
        span = self._spans.get(stage_name)

        if span is not None:
            # Record exception
            span.record_exception(error)

            # Mark as error
            span.set_status(Status(StatusCode.ERROR, str(error)))

            # End span
            span.end()

            # Remove from active spans
            self._spans.pop(stage_name, None)

    def on_pipeline_complete(
        self, context: ExecutionContext, result: ExecutionResult
    ) -> None:
        """
        Close root span with final metrics.

        Args:
            context: Final execution context
            result: Pipeline execution result
        """
        if not is_tracing_enabled():
            return

        span = self._spans.get("pipeline")

        if span is not None:
            # Add final metrics
            span.set_attribute("ondine.processed_rows", result.metrics.processed_rows)
            span.set_attribute("ondine.failed_rows", result.metrics.failed_rows)
            span.set_attribute(
                "ondine.duration_seconds", result.metrics.total_duration_seconds
            )
            span.set_attribute("ondine.total_cost", float(result.costs.total_cost))

            # Mark as successful
            span.set_status(Status(StatusCode.OK))

            # End span
            span.end()

            # Remove from active spans
            self._spans.pop("pipeline", None)

    def on_pipeline_error(self, context: ExecutionContext, error: Exception) -> None:
        """
        Close root span with error.

        Args:
            context: Execution context at failure
            error: Exception that occurred
        """
        if not is_tracing_enabled():
            return

        span = self._spans.get("pipeline")

        if span is not None:
            # Record exception
            span.record_exception(error)

            # Mark as error
            span.set_status(Status(StatusCode.ERROR, str(error)))

            # End span
            span.end()

            # Remove from active spans
            self._spans.pop("pipeline", None)
