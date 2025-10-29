"""
OpenTelemetry observer for infrastructure monitoring.

Integrates with OpenTelemetry to create distributed traces for pipeline execution.
Works with Jaeger, Datadog, Grafana, and other OTEL-compatible backends.
"""

from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Tracer

from ondine.observability.base import PipelineObserver
from ondine.observability.events import (
    ErrorEvent,
    LLMCallEvent,
    PipelineEndEvent,
    PipelineStartEvent,
    StageEndEvent,
    StageStartEvent,
)
from ondine.observability.registry import observer


@observer("opentelemetry")
class OpenTelemetryObserver(PipelineObserver):
    """
    Observer that creates OpenTelemetry spans for pipeline execution.

    Creates a hierarchical span structure:
    - Root span for pipeline execution
    - Nested spans for each stage
    - Nested spans for LLM calls
    - Attributes include metrics, errors, and metadata

    Configuration:
        - tracer_name: Name for the tracer (default: "ondine.pipeline")
        - include_prompts: Include full prompts in spans (default: False for PII safety)
        - max_prompt_length: Max prompt length to include (default: 1000 chars)

    Example:
        observer = OpenTelemetryObserver(config={
            "tracer_name": "my_pipeline",
            "include_prompts": False,
            "max_prompt_length": 500
        })
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize OpenTelemetry observer."""
        super().__init__(config)

        # Get tracer
        tracer_name = self.config.get("tracer_name", "ondine.pipeline")
        self.tracer: Tracer = trace.get_tracer(tracer_name)

        # Configuration
        self.include_prompts = self.config.get("include_prompts", False)
        self.max_prompt_length = self.config.get("max_prompt_length", 1000)

        # Track active spans
        self._spans: dict[str, trace.Span] = {}

    def on_pipeline_start(self, event: PipelineStartEvent) -> None:
        """Create root span for pipeline execution."""
        span = self.tracer.start_span(
            "pipeline.execute",
            attributes={
                "ondine.pipeline_id": str(event.pipeline_id),
                "ondine.run_id": str(event.run_id),
                "ondine.total_rows": event.total_rows,
            },
        )

        # Store span for later
        self._spans["pipeline"] = span

    def on_stage_start(self, event: StageStartEvent) -> None:
        """Create span for stage execution."""
        span = self.tracer.start_span(
            f"stage.{event.stage_name}",
            attributes={
                "ondine.stage_name": event.stage_name,
                "ondine.stage_type": event.stage_type,
            },
        )

        # Store span for completion/error handling
        self._spans[event.stage_name] = span

    def on_llm_call(self, event: LLMCallEvent) -> None:
        """
        Create span for LLM invocation.

        Note: Prompts can exceed OpenTelemetry's 64KB attribute limit,
        so we hash them by default or truncate if include_prompts=True.
        """
        # Prepare prompt value
        if self.include_prompts:
            prompt_value = event.prompt[: self.max_prompt_length]
            if len(event.prompt) > self.max_prompt_length:
                prompt_value += "...(truncated)"
        else:
            # Hash prompt to avoid PII and size limits
            prompt_hash = hash(event.prompt) % 100000
            prompt_value = f"<prompt-hash-{prompt_hash}>"

        # Create span
        span = self.tracer.start_span(
            "llm.call",
            attributes={
                "ondine.row_index": event.row_index,
                "llm.model": event.model,
                "llm.provider": event.provider,
                "llm.temperature": event.temperature,
                "llm.prompt": prompt_value,
                "llm.input_tokens": event.input_tokens,
                "llm.output_tokens": event.output_tokens,
                "llm.total_tokens": event.total_tokens,
                "llm.cost": float(event.cost),
                "llm.latency_ms": event.latency_ms,
            },
        )

        # Add RAG metadata if present
        if event.rag_technique:
            span.set_attribute("rag.technique", event.rag_technique)
        if event.retrieval_latency_ms:
            span.set_attribute("rag.retrieval_latency_ms", event.retrieval_latency_ms)

        # End span immediately (LLM call is atomic)
        span.end()

    def on_stage_end(self, event: StageEndEvent) -> None:
        """Close stage span with success/error status."""
        span = self._spans.get(event.stage_name)

        if span is not None:
            # Add completion metrics
            span.set_attribute("ondine.rows_processed", event.rows_processed)
            span.set_attribute("ondine.duration_ms", event.duration_ms)

            # Set status
            if event.success:
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status(Status(StatusCode.ERROR))
                if event.error:
                    span.record_exception(event.error)

            # End span
            span.end()

            # Remove from active spans
            self._spans.pop(event.stage_name, None)

    def on_error(self, event: ErrorEvent) -> None:
        """Record error in current span."""
        # Find the most relevant span (stage or pipeline)
        span = None
        if event.stage_name and event.stage_name in self._spans:
            span = self._spans[event.stage_name]
        elif "pipeline" in self._spans:
            span = self._spans["pipeline"]

        if span and event.error:
            span.record_exception(event.error)
            span.set_status(Status(StatusCode.ERROR, event.error_message))

    def on_pipeline_end(self, event: PipelineEndEvent) -> None:
        """Close root span with final metrics."""
        span = self._spans.get("pipeline")

        if span is not None:
            # Add final metrics
            span.set_attribute("ondine.rows_processed", event.rows_processed)
            span.set_attribute("ondine.rows_succeeded", event.rows_succeeded)
            span.set_attribute("ondine.rows_failed", event.rows_failed)
            span.set_attribute("ondine.duration_ms", event.total_duration_ms)
            span.set_attribute("ondine.total_cost", float(event.total_cost))
            span.set_attribute("ondine.total_tokens", event.total_tokens)

            # Set status
            if event.success:
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status(Status(StatusCode.ERROR))

            # End span
            span.end()

            # Remove from active spans
            self._spans.pop("pipeline", None)

    def flush(self) -> None:
        """Flush OpenTelemetry spans."""
        # OpenTelemetry SDK handles flushing automatically
        # But we can force it if needed
        pass

    def close(self) -> None:
        """Clean up OpenTelemetry resources."""
        # Close any remaining spans
        for span in list(self._spans.values()):
            span.end()
        self._spans.clear()
