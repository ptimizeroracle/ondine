"""
OpenTelemetry observer for infrastructure monitoring.

Uses OpenTelemetry SDK directly for tracing LLM calls.
No dependency on LiteLLM's internal callback mechanism.
"""

import logging
import os
from typing import Any

from ondine.observability.base import PipelineObserver
from ondine.observability.events import (
    LLMCallEvent,
    PipelineEndEvent,
    PipelineStartEvent,
    StageEndEvent,
    StageStartEvent,
)
from ondine.observability.registry import observer

logger = logging.getLogger(__name__)


@observer("opentelemetry")
class OpenTelemetryObserver(PipelineObserver):
    """
    Observer that uses OpenTelemetry SDK directly for tracing.

    This implementation:
    - Works with any OTel-compatible backend (Jaeger, Honeycomb, Datadog, etc.)
    - Does NOT depend on LiteLLM's internal callbacks
    - Receives events from Ondine's ObserverDispatcher

    Configuration:
    - Uses standard OTel environment variables (OTEL_EXPORTER_OTLP_ENDPOINT, etc.)
    - Or pass specific config in 'config' dict

    Example:
        observer = OpenTelemetryObserver(config={
            "service_name": "my-pipeline",
            "endpoint": "http://localhost:4318"
        })
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize OpenTelemetry observer with tracer.
        """
        super().__init__(config)

        self._tracer = None
        self._provider = None
        self._pipeline_span = None
        self._stage_spans: dict[str, Any] = {}

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider

            # Get or create tracer provider
            provider = trace.get_tracer_provider()
            if not isinstance(provider, TracerProvider):
                provider = TracerProvider()
                trace.set_tracer_provider(provider)
            self._provider = provider

            # Set service name if configured
            if self.config and self.config.get("service_name"):
                os.environ["OTEL_SERVICE_NAME"] = self.config["service_name"]

            # Set endpoint if configured
            if self.config and self.config.get("endpoint"):
                os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = self.config["endpoint"]

            # Create tracer
            self._tracer = trace.get_tracer("ondine.llm")
            logger.info("OpenTelemetry observer initialized (direct SDK)")

        except ImportError:
            logger.warning(
                "OpenTelemetry SDK not installed. Install with: pip install opentelemetry-sdk"
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")

    def on_pipeline_start(self, event: PipelineStartEvent) -> None:
        """
        Start a root span for the pipeline.
        """
        if not self._tracer:
            return

        try:
            self._pipeline_span = self._tracer.start_span(
                name="ondine.pipeline",
                attributes={
                    "pipeline.id": str(event.pipeline_id),
                    "pipeline.run_id": str(event.run_id),
                    "pipeline.total_rows": event.total_rows,
                },
            )
            logger.debug(f"Started OTel pipeline span: {event.run_id}")
        except Exception as e:
            logger.debug(f"Failed to start OTel pipeline span: {e}")

    def on_stage_start(self, event: StageStartEvent) -> None:
        """
        Start a span for a pipeline stage.
        """
        if not self._tracer:
            return

        try:
            span = self._tracer.start_span(
                name=f"ondine.stage.{event.stage_name}",
                attributes={
                    "stage.name": event.stage_name,
                    "stage.type": event.stage_type,
                },
            )
            self._stage_spans[event.stage_name] = span
        except Exception as e:
            logger.debug(f"Failed to start OTel stage span: {e}")

    def on_llm_call(self, event: LLMCallEvent) -> None:
        """
        Create a span for the LLM call with relevant attributes.
        """
        if not self._tracer:
            return

        try:
            with self._tracer.start_as_current_span("llm.completion") as span:
                # Standard LLM attributes (following emerging semantic conventions)
                span.set_attribute("llm.model", event.model)
                span.set_attribute("llm.provider", event.provider)
                span.set_attribute("llm.temperature", event.temperature)

                # Token usage
                span.set_attribute("llm.usage.input_tokens", event.input_tokens)
                span.set_attribute("llm.usage.output_tokens", event.output_tokens)
                span.set_attribute("llm.usage.total_tokens", event.total_tokens)

                # Cost and latency
                span.set_attribute("llm.cost", float(event.cost))
                span.set_attribute("llm.latency_ms", event.latency_ms)

                # Context
                span.set_attribute("llm.stage_name", event.stage_name)
                span.set_attribute("llm.row_index", event.row_index)

                # Optionally log prompts (configurable for PII concerns)
                if self.config and self.config.get("log_prompts", False):
                    # Truncate to avoid huge spans
                    span.set_attribute("llm.prompt", event.prompt[:1000])
                    span.set_attribute("llm.completion", event.completion[:1000])

        except Exception as e:
            logger.debug(f"Failed to create OTel LLM span: {e}")

    def on_stage_end(self, event: StageEndEvent) -> None:
        """
        End the stage span.
        """
        if event.stage_name in self._stage_spans:
            try:
                span = self._stage_spans.pop(event.stage_name)
                span.set_attribute("stage.success", event.success)
                span.set_attribute("stage.duration_ms", event.duration_ms)
                span.set_attribute("stage.rows_processed", event.rows_processed)
                span.end()
            except Exception as e:
                logger.debug(f"Failed to end OTel stage span: {e}")

    def on_pipeline_end(self, event: PipelineEndEvent) -> None:
        """
        End the pipeline span with final metrics.
        """
        if not self._pipeline_span:
            return

        try:
            self._pipeline_span.set_attribute("pipeline.success", event.success)
            self._pipeline_span.set_attribute(
                "pipeline.rows_processed", event.rows_processed
            )
            self._pipeline_span.set_attribute(
                "pipeline.rows_succeeded", event.rows_succeeded
            )
            self._pipeline_span.set_attribute("pipeline.rows_failed", event.rows_failed)
            self._pipeline_span.set_attribute(
                "pipeline.total_cost", float(event.total_cost)
            )
            self._pipeline_span.set_attribute(
                "pipeline.total_tokens", event.total_tokens
            )
            self._pipeline_span.set_attribute(
                "pipeline.duration_ms", event.total_duration_ms
            )
            self._pipeline_span.end()
            self._pipeline_span = None
            logger.debug("Ended OTel pipeline span")
        except Exception as e:
            logger.debug(f"Failed to end OTel pipeline span: {e}")

    def flush(self) -> None:
        """Flush spans to exporter."""
        if self._provider and hasattr(self._provider, "force_flush"):
            try:
                self._provider.force_flush()
            except Exception as e:
                logger.debug(f"Failed to flush OTel spans: {e}")

    def close(self) -> None:
        """Cleanup OpenTelemetry resources."""
        # End any open spans
        for span in self._stage_spans.values():
            try:
                span.end()
            except Exception:  # nosec B110
                # Cleanup errors are non-critical
                pass
        self._stage_spans.clear()

        if self._pipeline_span:
            try:
                self._pipeline_span.end()
            except Exception:  # nosec B110
                # Cleanup errors are non-critical
                pass
            self._pipeline_span = None

        self.flush()
