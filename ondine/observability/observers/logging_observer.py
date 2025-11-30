"""
Logging observer for simple console/file observability.

Logs LLM interactions using standard Python logging.
"""

import logging
from typing import Any

import litellm

from ondine.observability.base import PipelineObserver
from ondine.observability.registry import observer

logger = logging.getLogger(__name__)


@observer("logging")
class LoggingObserver(PipelineObserver):
    """
    Observer that uses LiteLLM's verbose logging or standard Python logging.

    This provides basic console logging without external dependencies.

    Configuration:
        - level: Logging level (INFO, DEBUG)
        - verbose: Enable LiteLLM verbose mode (boolean)

    Example:
        observer = LoggingObserver(config={"verbose": True})
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize logging observer.
        """
        super().__init__(config)

        # Configure LiteLLM logging
        if self.config and self.config.get("verbose", False):
            litellm.set_verbose = True
            logger.info("LiteLLM verbose logging enabled")

        # Ensure we have a handler for this logger if none exists
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        logger.info("Logging observer initialized")

    def on_llm_call(self, event: Any) -> None:
        """
        LLM calls are logged via standard logging if enabled.
        """
        pass

    def flush(self) -> None:
        """Flush (handled by logging module)."""
        pass

    def close(self) -> None:
        """Cleanup (handled by logging module)."""
        pass
