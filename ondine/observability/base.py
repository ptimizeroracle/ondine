"""
Base class for pipeline observers.

Observers receive events during pipeline execution and can log, trace,
or send data to external observability systems.
"""

from abc import ABC, abstractmethod
from typing import Any

from ondine.observability.events import (
    ErrorEvent,
    LLMCallEvent,
    PipelineEndEvent,
    PipelineStartEvent,
    StageEndEvent,
    StageStartEvent,
)


class PipelineObserver(ABC):
    """
    Base class for all pipeline observers.

    Observers receive events and can:
    - Log to files or stdout
    - Send to external services (Langfuse, OpenTelemetry, etc.)
    - Update metrics dashboards
    - Trigger alerts

    All methods are optional (default: no-op) except on_llm_call(),
    which is the most critical event for LLM observability.

    Observer implementations should be fault-tolerant - errors in observers
    should never crash the pipeline.

    Args:
        config: Observer-specific configuration dictionary

    Example:
        class MyObserver(PipelineObserver):
            def on_llm_call(self, event: LLMCallEvent) -> None:
                print(f"LLM called: {event.model} - ${event.cost}")
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize observer with configuration.

        Args:
            config: Observer-specific configuration (e.g., API keys, endpoints)
        """
        self.config = config or {}

    def on_pipeline_start(self, event: PipelineStartEvent) -> None:
        """
        Called when pipeline execution starts.

        Use this to initialize traces, create root spans, or start timers.

        Args:
            event: Pipeline start event with configuration
        """
        pass

    def on_stage_start(self, event: StageStartEvent) -> None:
        """
        Called when a pipeline stage begins execution.

        Use this to create nested spans or log stage transitions.

        Args:
            event: Stage start event with stage details
        """
        pass

    @abstractmethod
    def on_llm_call(self, event: LLMCallEvent) -> None:
        """
        Called on every LLM invocation.

        This is the MOST CRITICAL method for LLM observability.
        ALL observers MUST implement this method.

        Contains full prompt, completion, tokens, cost, and metadata.
        Observers should handle PII sanitization if needed.

        Args:
            event: LLM call event with request/response details
        """
        pass

    def on_stage_end(self, event: StageEndEvent) -> None:
        """
        Called when a pipeline stage completes.

        Use this to close spans, log duration, or record metrics.

        Args:
            event: Stage end event with success status and metrics
        """
        pass

    def on_error(self, event: ErrorEvent) -> None:
        """
        Called when errors occur during execution.

        Use this for error tracking, alerting, or debugging.

        Args:
            event: Error event with exception details and context
        """
        pass

    def on_pipeline_end(self, event: PipelineEndEvent) -> None:
        """
        Called when pipeline execution completes.

        Use this to close root spans, log final metrics, or cleanup.

        Args:
            event: Pipeline end event with final statistics
        """
        pass

    def flush(self) -> None:
        """
        Flush any buffered events.

        Called at the end of pipeline execution to ensure
        all events are sent before the pipeline exits.

        Observers that batch events should send them here.
        """
        pass

    def close(self) -> None:
        """
        Clean up resources.

        Called when observer is no longer needed.
        Close connections, files, or other resources.
        """
        pass
