"""
Logging observer for simple console/file observability.

Delegates to LlamaIndex's Simple handler for automatic LLM call logging.
"""

import logging
from typing import Any

from ondine.observability.base import PipelineObserver
from ondine.observability.llamaindex_handlers import LlamaIndexHandlerManager
from ondine.observability.registry import observer

logger = logging.getLogger(__name__)


@observer("logging")
class LoggingObserver(PipelineObserver):
    """
    Observer that delegates to LlamaIndex's Simple handler.

    LlamaIndex automatically logs:
    - ✅ LLM calls with prompts and completions
    - ✅ Token usage
    - ✅ Latency metrics

    This provides basic console logging without external dependencies.

    Configuration:
        - (No specific config needed - uses LlamaIndex defaults)

    Example:
        observer = LoggingObserver(config={})
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize logging observer.

        Configures LlamaIndex's Simple handler for console logging.
        """
        super().__init__(config)

        # Configure LlamaIndex's Simple handler (console logging)
        # This will automatically log all LLM calls!
        LlamaIndexHandlerManager.configure_handler("simple", self.config)

        logger.info("Logging observer initialized (using LlamaIndex SimpleHandler)")

    def on_llm_call(self, event: Any) -> None:
        """
        LLM calls are automatically logged by LlamaIndex.

        No action needed - LlamaIndex's Simple handler logs:
        - Prompt and completion
        - Token usage
        - Latency
        """
        # LlamaIndex handles this automatically!
        pass

    def flush(self) -> None:
        """Flush (handled by logging module)."""
        pass

    def close(self) -> None:
        """Cleanup (handled by logging module)."""
        pass
