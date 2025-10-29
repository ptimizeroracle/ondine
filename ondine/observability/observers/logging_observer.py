"""
Logging observer for simple console/file observability.

Provides structured logging of pipeline events for debugging and monitoring.
"""

import logging
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


@observer("logging")
class LoggingObserver(PipelineObserver):
    """
    Observer that logs pipeline events to console or file.

    Useful for:
    - Development and debugging
    - Simple monitoring without external dependencies
    - Audit trails

    Configuration:
        - logger_name: Logger name (default: "ondine.pipeline")
        - log_level: Logging level (default: "INFO")
        - include_prompts: Include prompt previews in logs (default: False)
        - prompt_preview_length: Max prompt preview length (default: 100)

    Example:
        observer = LoggingObserver(config={
            "logger_name": "my_pipeline",
            "log_level": "DEBUG",
            "include_prompts": True,
            "prompt_preview_length": 200
        })
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize logging observer."""
        super().__init__(config)

        # Get logger
        logger_name = self.config.get("logger_name", "ondine.pipeline")
        self.logger = logging.getLogger(logger_name)

        # Set log level
        log_level = self.config.get("log_level", "INFO")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Configuration
        self.include_prompts = self.config.get("include_prompts", False)
        self.prompt_preview_length = self.config.get("prompt_preview_length", 100)

    def on_pipeline_start(self, event: PipelineStartEvent) -> None:
        """Log pipeline start."""
        self.logger.info(
            f"Pipeline execution started | "
            f"run_id={event.run_id} | "
            f"total_rows={event.total_rows}"
        )

    def on_stage_start(self, event: StageStartEvent) -> None:
        """Log stage start."""
        self.logger.info(
            f"Stage started | stage={event.stage_name} | type={event.stage_type}"
        )

    def on_llm_call(self, event: LLMCallEvent) -> None:
        """
        Log LLM invocation with key metrics.

        This provides visibility into LLM usage, costs, and performance.
        """
        # Prepare prompt preview
        prompt_preview = ""
        if self.include_prompts:
            prompt_text = event.prompt[: self.prompt_preview_length]
            if len(event.prompt) > self.prompt_preview_length:
                prompt_text += "..."
            prompt_preview = f' | prompt="{prompt_text}"'

        # Log LLM call
        self.logger.info(
            f"LLM call | "
            f"row={event.row_index} | "
            f"model={event.model} | "
            f"provider={event.provider} | "
            f"tokens={event.total_tokens} "
            f"(in={event.input_tokens}, out={event.output_tokens}) | "
            f"cost=${event.cost:.6f} | "
            f"latency={event.latency_ms:.2f}ms"
            f"{prompt_preview}"
        )

        # Log RAG metadata if present
        if event.rag_technique:
            rag_info = (
                f"RAG retrieval | "
                f"technique={event.rag_technique} | "
                f"retrieval_latency={event.retrieval_latency_ms:.2f}ms"
            )
            if event.rag_sources:
                rag_info += f" | num_sources={len(event.rag_sources)}"
            self.logger.debug(rag_info)

    def on_stage_end(self, event: StageEndEvent) -> None:
        """Log stage completion."""
        status = "completed" if event.success else "failed"

        self.logger.info(
            f"Stage {status} | "
            f"stage={event.stage_name} | "
            f"duration={event.duration_ms:.2f}ms | "
            f"rows={event.rows_processed}"
        )

        if not event.success and event.error:
            self.logger.error(f"Stage error: {event.error}")

    def on_error(self, event: ErrorEvent) -> None:
        """Log error details."""
        context_str = ""
        if event.stage_name:
            context_str += f" | stage={event.stage_name}"
        if event.row_index is not None:
            context_str += f" | row={event.row_index}"

        self.logger.error(
            f"Pipeline error{context_str} | "
            f"type={event.error_type} | "
            f"message={event.error_message}"
        )

        if event.stack_trace:
            self.logger.debug(f"Stack trace:\n{event.stack_trace}")

    def on_pipeline_end(self, event: PipelineEndEvent) -> None:
        """Log pipeline completion with final metrics."""
        status = "completed successfully" if event.success else "failed"

        self.logger.info(
            f"Pipeline execution {status}\n"
            f"  Processed: {event.rows_processed} rows\n"
            f"  Succeeded: {event.rows_succeeded} rows\n"
            f"  Failed: {event.rows_failed} rows\n"
            f"  Duration: {event.total_duration_ms / 1000:.2f}s\n"
            f"  Total cost: ${event.total_cost:.4f}\n"
            f"  Total tokens: {event.total_tokens:,} "
            f"(in={event.input_tokens:,}, out={event.output_tokens:,})"
        )

    def flush(self) -> None:
        """Flush logging handlers."""
        for handler in self.logger.handlers:
            handler.flush()

    def close(self) -> None:
        """Close logging handlers."""
        # Logging handlers are managed by the logging module
        pass
