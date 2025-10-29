"""
OpenTelemetry observer for infrastructure monitoring.

Delegates to LlamaIndex's built-in OpenTelemetry handler for LLM call tracking,
while adding pipeline-level observability on top.
"""

import logging
from typing import Any

from ondine.observability.base import PipelineObserver
from ondine.observability.llamaindex_handlers import LlamaIndexHandlerManager
from ondine.observability.registry import observer

logger = logging.getLogger(__name__)


@observer("opentelemetry")
class OpenTelemetryObserver(PipelineObserver):
    """
    Observer that delegates to LlamaIndex's OpenTelemetry handler.

    LlamaIndex automatically instruments:
    - ✅ All LLM calls (prompts, completions, tokens, latency)
    - ✅ Embeddings
    - ✅ Retrieval operations (when using QueryEngines)

    This observer configures the LlamaIndex handler and can add
    pipeline-level spans on top if needed.

    Configuration:
        - Any config accepted by LlamaIndex's OpenTelemetry handler
        - See: https://docs.llamaindex.ai/en/stable/module_guides/observability/

    Example:
        observer = OpenTelemetryObserver(config={})
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize OpenTelemetry observer.

        Configures LlamaIndex's global OpenTelemetry handler.
        """
        super().__init__(config)

        # Configure LlamaIndex's OpenTelemetry handler
        # This will automatically instrument all LLM calls!
        LlamaIndexHandlerManager.configure_handler("opentelemetry", self.config)

        logger.info("OpenTelemetry observer initialized (using LlamaIndex handler)")

    def on_llm_call(self, event: Any) -> None:
        """
        LLM calls are automatically traced by LlamaIndex.

        No action needed - LlamaIndex's OpenTelemetry handler captures:
        - Prompt and completion
        - Token usage
        - Latency
        - Model information
        """
        # LlamaIndex handles this automatically!
        pass

    def flush(self) -> None:
        """Flush spans (handled by OpenTelemetry SDK)."""
        pass

    def close(self) -> None:
        """Cleanup (handled by OpenTelemetry SDK)."""
        pass
