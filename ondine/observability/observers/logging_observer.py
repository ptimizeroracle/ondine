"""
Logging observer for simple console/file observability.

Logs LLM interactions using standard Python logging.
No dependency on LiteLLM's internal callback mechanism.
"""

import logging
from typing import Any

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

logger = logging.getLogger(__name__)


@observer("logging")
class LoggingObserver(PipelineObserver):
    """
    Observer that logs events using standard Python logging.

    This provides basic console/file logging without external dependencies.

    Configuration:
        - level: Logging level (INFO, DEBUG) - default INFO
        - log_prompts: Whether to log full prompts (default False for PII)
        - log_completions: Whether to log full completions (default False)

    Example:
        observer = LoggingObserver(config={
            "level": "DEBUG",
            "log_prompts": True
        })
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize logging observer.
        """
        super().__init__(config)

        # Configure log level
        level_name = self.config.get("level", "INFO") if self.config else "INFO"
        self._level = getattr(logging, level_name.upper(), logging.INFO)

        # Configure what to log
        self._log_prompts = self.config.get("log_prompts", False) if self.config else False
        self._log_completions = self.config.get("log_completions", False) if self.config else False

        # Ensure we have a handler
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.setLevel(self._level)
        logger.info("Logging observer initialized")

    def on_pipeline_start(self, event: PipelineStartEvent) -> None:
        """Log pipeline start."""
        logger.log(
            self._level,
            f"Pipeline started: {event.run_id} ({event.total_rows} rows)",
        )

    def on_stage_start(self, event: StageStartEvent) -> None:
        """Log stage start."""
        logger.log(self._level, f"Stage started: {event.stage_name}")

    def on_llm_call(self, event: LLMCallEvent) -> None:
        """Log LLM call with key metrics."""
        msg = (
            f"LLM call: model={event.model} "
            f"tokens={event.input_tokens}+{event.output_tokens} "
            f"cost=${float(event.cost):.6f} "
            f"latency={event.latency_ms:.0f}ms"
        )

        if self._log_prompts:
            prompt_preview = event.prompt[:100] + "..." if len(event.prompt) > 100 else event.prompt
            msg += f" prompt={prompt_preview!r}"

        if self._log_completions:
            completion_preview = event.completion[:100] + "..." if len(event.completion) > 100 else event.completion
            msg += f" completion={completion_preview!r}"

        logger.log(self._level, msg)

    def on_stage_end(self, event: StageEndEvent) -> None:
        """Log stage completion."""
        status = "completed" if event.success else "failed"
        logger.log(
            self._level,
            f"Stage {status}: {event.stage_name} "
            f"({event.rows_processed} rows, {event.duration_ms:.0f}ms)",
        )

    def on_error(self, event: ErrorEvent) -> None:
        """Log errors."""
        logger.error(
            f"Error in {event.stage_name or 'pipeline'}: "
            f"{event.error_type}: {event.error_message}"
        )

    def on_pipeline_end(self, event: PipelineEndEvent) -> None:
        """Log pipeline completion."""
        status = "completed" if event.success else "failed"
        logger.log(
            self._level,
            f"Pipeline {status}: {event.rows_processed} rows, "
            f"${float(event.total_cost):.4f}, {event.total_duration_ms:.0f}ms",
        )

    def flush(self) -> None:
        """Flush log handlers."""
        for handler in logger.handlers:
            handler.flush()

    def close(self) -> None:
        """Cleanup."""
        self.flush()
