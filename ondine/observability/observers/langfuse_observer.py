"""
Langfuse observer for LLM-specific observability.

Integrates with Langfuse to track prompts, completions, tokens, costs,
and other LLM-specific metrics.
"""

from datetime import timedelta
from decimal import Decimal
from typing import Any, Optional

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

try:
    from langfuse import Langfuse

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False


@observer("langfuse")
class LangfuseObserver(PipelineObserver):
    """
    Observer that sends LLM metrics to Langfuse.

    Langfuse is purpose-built for LLM observability, providing:
    - Full prompt and completion tracking
    - Token usage and cost analysis
    - Prompt versioning and A/B testing
    - User feedback integration
    - Quality metrics

    Configuration:
        - public_key: Langfuse public key (required)
        - secret_key: Langfuse secret key (required)
        - host: Langfuse host URL (optional, defaults to cloud)
        - max_context_length: Max RAG context length to include (default: 2000)
        - flush_interval: Number of events to buffer before flushing (default: 100)

    Example:
        observer = LangfuseObserver(config={
            "public_key": "pk-lf-...",
            "secret_key": "sk-lf-...",
            "host": "https://cloud.langfuse.com",  # optional
            "max_context_length": 2000
        })

    Raises:
        ImportError: If langfuse package not installed
        ValueError: If required config (public_key, secret_key) missing
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize Langfuse observer."""
        super().__init__(config)

        if not LANGFUSE_AVAILABLE:
            raise ImportError(
                "Langfuse SDK not installed. Install with: pip install langfuse"
            )

        # Validate required config
        public_key = self.config.get("public_key")
        secret_key = self.config.get("secret_key")

        if not public_key or not secret_key:
            raise ValueError(
                "Langfuse requires 'public_key' and 'secret_key' in config. "
                "Get your keys from: https://cloud.langfuse.com"
            )

        # Initialize Langfuse client
        host = self.config.get("host", "https://cloud.langfuse.com")
        self.langfuse = Langfuse(
            public_key=public_key, secret_key=secret_key, host=host
        )

        # Configuration
        self.max_context_length = self.config.get("max_context_length", 2000)
        self.flush_interval = self.config.get("flush_interval", 100)

        # State
        self._current_trace_id: Optional[str] = None
        self._event_count = 0

    def on_pipeline_start(self, event: PipelineStartEvent) -> None:
        """Create Langfuse trace for pipeline execution."""
        # Create trace
        trace_id = str(event.run_id)
        self._current_trace_id = trace_id

        self.langfuse.trace(
            id=trace_id,
            name="ondine_pipeline_execution",
            metadata={
                "pipeline_id": str(event.pipeline_id),
                "total_rows": event.total_rows,
                **event.metadata,
            },
            tags=["ondine", "pipeline"],
        )

    def on_stage_start(self, event: StageStartEvent) -> None:
        """Create Langfuse span for stage execution."""
        if not self._current_trace_id:
            return

        self.langfuse.span(
            id=event.span_id,
            trace_id=self._current_trace_id,
            name=f"stage_{event.stage_name}",
            start_time=event.timestamp,
            metadata={
                "stage_type": event.stage_type,
                **event.metadata,
            },
        )

    def on_llm_call(self, event: LLMCallEvent) -> None:
        """
        Send LLM call to Langfuse as a generation.

        This is the most important event - Langfuse excels at LLM tracking.
        """
        if not self._current_trace_id:
            return

        # Prepare metadata
        metadata = {
            "model": event.model,
            "provider": event.provider,
            "temperature": event.temperature,
            "row_index": event.row_index,
            **event.metadata,
        }

        # Add RAG metadata if present
        if event.rag_technique:
            metadata["rag_technique"] = event.rag_technique

            # Truncate RAG context if too long
            if event.rag_context:
                rag_context = event.rag_context
                if len(rag_context) > self.max_context_length:
                    rag_context = (
                        rag_context[: self.max_context_length] + "...(truncated)"
                    )
                metadata["rag_context_preview"] = rag_context

            if event.rag_sources:
                metadata["rag_sources"] = event.rag_sources
                metadata["num_rag_sources"] = len(event.rag_sources)

            if event.retrieval_latency_ms:
                metadata["retrieval_latency_ms"] = event.retrieval_latency_ms

        # Send generation to Langfuse
        self.langfuse.generation(
            id=event.span_id,
            trace_id=self._current_trace_id,
            name=f"llm_call_{event.row_index}",
            start_time=event.timestamp,
            end_time=event.timestamp + timedelta(milliseconds=event.latency_ms),
            model=event.model,
            prompt=event.prompt,
            completion=event.completion,
            usage={
                "input": event.input_tokens,
                "output": event.output_tokens,
                "total": event.total_tokens,
            },
            metadata=metadata,
            level="DEFAULT",
            status_message=event.finish_reason,
        )

        # Increment event count for flushing
        self._event_count += 1

        # Flush if we've buffered enough events
        if self._event_count >= self.flush_interval:
            self.flush()
            self._event_count = 0

    def on_stage_end(self, event: StageEndEvent) -> None:
        """Update Langfuse span with completion metrics."""
        if not self._current_trace_id:
            return

        # Langfuse spans are updated via the SDK's span update method
        # We'll just track metrics in metadata for now
        pass

    def on_error(self, event: ErrorEvent) -> None:
        """Log error to Langfuse trace."""
        if not self._current_trace_id:
            return

        # Create an event in the trace for the error
        self.langfuse.event(
            trace_id=self._current_trace_id,
            name="pipeline_error",
            metadata={
                "error_type": event.error_type,
                "error_message": event.error_message,
                "stage_name": event.stage_name,
                "row_index": event.row_index,
                **event.context,
            },
            level="ERROR",
        )

    def on_pipeline_end(self, event: PipelineEndEvent) -> None:
        """Update Langfuse trace with final metrics."""
        if not self._current_trace_id:
            return

        # Update trace with final metrics
        self.langfuse.trace(
            id=self._current_trace_id,
            metadata={
                "rows_processed": event.rows_processed,
                "rows_succeeded": event.rows_succeeded,
                "rows_failed": event.rows_failed,
                "total_duration_ms": event.total_duration_ms,
                "total_cost": float(event.total_cost),
                "total_tokens": event.total_tokens,
                "input_tokens": event.input_tokens,
                "output_tokens": event.output_tokens,
                **event.metrics,
            },
            tags=["ondine", "pipeline", "completed" if event.success else "failed"],
        )

        # Clear current trace
        self._current_trace_id = None

    def flush(self) -> None:
        """Flush buffered events to Langfuse."""
        try:
            self.langfuse.flush()
        except Exception:
            # Langfuse SDK handles errors internally
            pass

    def close(self) -> None:
        """Close Langfuse connection."""
        # Final flush
        self.flush()

        # Langfuse SDK cleans up automatically
        self._current_trace_id = None
