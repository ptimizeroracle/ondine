"""
Langfuse observer for LLM-specific observability.

Uses Langfuse SDK directly (compatible with v2 and v3).
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
)
from ondine.observability.registry import observer

logger = logging.getLogger(__name__)


@observer("langfuse")
class LangfuseObserver(PipelineObserver):
    """
    Observer that uses Langfuse SDK directly for LLM observability.

    This implementation:
    - Works with Langfuse v2.x and v3.x
    - Does NOT depend on LiteLLM's internal callbacks
    - Receives events from Ondine's ObserverDispatcher

    Tracks:
    - Full prompts and completions
    - Token usage and costs
    - Latency metrics
    - Model information
    - Pipeline traces with nested generations

    Configuration:
    - Requires environment variables: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
    - Optional: LANGFUSE_HOST (defaults to cloud)
    - Or pass in 'config' dict with keys

    Example:
        observer = LangfuseObserver(config={
            "public_key": "pk-lf-xxx",  # pragma: allowlist secret
            "secret_key": "sk-lf-xxx",  # pragma: allowlist secret
            "host": "https://cloud.langfuse.com"
        })
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Langfuse observer with direct SDK client.
        """
        super().__init__(config)

        # Initialize Langfuse client directly
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=self.config.get("public_key")
                or os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=self.config.get("secret_key")
                or os.getenv("LANGFUSE_SECRET_KEY"),
                host=self.config.get("host")
                or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
            self._current_trace = None
            logger.info("Langfuse observer initialized (direct SDK)")

        except ImportError:
            logger.warning(
                "Langfuse SDK not installed. Install with: pip install langfuse"
            )
            self._client = None
            self._current_trace = None
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {e}")
            self._client = None
            self._current_trace = None

    def on_pipeline_start(self, event: PipelineStartEvent) -> None:
        """
        Create a new trace for the pipeline run.
        """
        if not self._client:
            return

        try:
            self._current_trace = self._client.trace(
                id=str(event.run_id),
                name="ondine-pipeline",
                metadata={
                    "pipeline_id": str(event.pipeline_id),
                    "total_rows": event.total_rows,
                    **event.metadata,
                },
            )
            logger.debug(f"Created Langfuse trace: {event.run_id}")
        except Exception as e:
            logger.warning(f"Failed to create Langfuse trace: {e}")

    def on_llm_call(self, event: LLMCallEvent) -> None:
        """
        Log LLM call as a generation span in Langfuse.
        """
        if not self._client:
            return

        try:
            # Use current trace if available, otherwise create standalone generation
            if self._current_trace:
                self._current_trace.generation(
                    name=f"llm-{event.model}",
                    model=event.model,
                    input=event.prompt,
                    output=event.completion,
                    usage={
                        "input": event.input_tokens,
                        "output": event.output_tokens,
                        "total": event.total_tokens,
                    },
                    metadata={
                        "provider": event.provider,
                        "temperature": event.temperature,
                        "max_tokens": event.max_tokens,
                        "latency_ms": event.latency_ms,
                        "cost": float(event.cost),
                        "row_index": event.row_index,
                        "stage_name": event.stage_name,
                        **event.metadata,
                    },
                )
            else:
                # Standalone generation (no pipeline trace)
                trace = self._client.trace(
                    name=f"llm-call-{event.trace_id[:8]}",
                )
                trace.generation(
                    name=f"llm-{event.model}",
                    model=event.model,
                    input=event.prompt,
                    output=event.completion,
                    usage={
                        "input": event.input_tokens,
                        "output": event.output_tokens,
                        "total": event.total_tokens,
                    },
                    metadata={
                        "provider": event.provider,
                        "temperature": event.temperature,
                        "latency_ms": event.latency_ms,
                        "cost": float(event.cost),
                        **event.metadata,
                    },
                )

        except Exception as e:
            logger.debug(f"Failed to log LLM call to Langfuse: {e}")

    def on_pipeline_end(self, event: PipelineEndEvent) -> None:
        """
        Update trace with final pipeline metrics.
        """
        if not self._client or not self._current_trace:
            return

        try:
            self._current_trace.update(
                output={
                    "success": event.success,
                    "rows_processed": event.rows_processed,
                    "rows_succeeded": event.rows_succeeded,
                    "rows_failed": event.rows_failed,
                    "total_cost": float(event.total_cost),
                    "total_tokens": event.total_tokens,
                    "duration_ms": event.total_duration_ms,
                },
            )
            logger.debug(f"Updated Langfuse trace with final metrics")
        except Exception as e:
            logger.debug(f"Failed to update Langfuse trace: {e}")

    def flush(self) -> None:
        """Flush buffered events to Langfuse."""
        if not self._client:
            return

        try:
            self._client.flush()
        except Exception as e:
            logger.debug(f"Failed to flush Langfuse: {e}")

    def close(self) -> None:
        """Cleanup Langfuse client."""
        self.flush()
        if self._client:
            try:
                # Langfuse v2 uses shutdown(), v3 may differ
                if hasattr(self._client, "shutdown"):
                    self._client.shutdown()
            except Exception:  # nosec B110
                # Cleanup errors are non-critical
                pass
        self._current_trace = None
