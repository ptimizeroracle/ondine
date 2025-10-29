"""
Langfuse observer for LLM-specific observability.

Delegates to LlamaIndex's built-in Langfuse handler for automatic tracking
of prompts, completions, tokens, and costs.
"""

import logging
from typing import Any

from ondine.observability.base import PipelineObserver
from ondine.observability.llamaindex_handlers import LlamaIndexHandlerManager
from ondine.observability.registry import observer

logger = logging.getLogger(__name__)


@observer("langfuse")
class LangfuseObserver(PipelineObserver):
    """
    Observer that delegates to LlamaIndex's Langfuse handler.

    LlamaIndex automatically tracks:
    - ✅ Full prompts and completions
    - ✅ Token usage and costs
    - ✅ Latency metrics
    - ✅ Model information
    - ✅ Prompt versioning

    Configuration:
        - public_key: Langfuse public key (required)
        - secret_key: Langfuse secret key (required)
        - host: Langfuse host URL (optional, defaults to cloud)

    Example:
        observer = LangfuseObserver(config={
            "public_key": "pk-lf-...",
            "secret_key": "sk-lf-...",
        })

    Raises:
        ValueError: If required config (public_key, secret_key) missing
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Langfuse observer.

        Configures LlamaIndex's global Langfuse handler.
        """
        super().__init__(config)

        # Validate required config
        if not self.config.get("public_key") or not self.config.get("secret_key"):
            raise ValueError(
                "Langfuse requires 'public_key' and 'secret_key' in config. "
                "Get your keys from: https://cloud.langfuse.com"
            )

        # Configure LlamaIndex's Langfuse handler
        # This will automatically track all LLM calls!
        LlamaIndexHandlerManager.configure_handler("langfuse", self.config)

        logger.info("Langfuse observer initialized (using LlamaIndex handler)")

    def on_llm_call(self, event: Any) -> None:
        """
        LLM calls are automatically tracked by LlamaIndex.

        No action needed - LlamaIndex's Langfuse handler captures:
        - Full prompt and completion text
        - Token usage and costs
        - Model information
        - Latency metrics
        """
        # LlamaIndex handles this automatically!
        pass

    def flush(self) -> None:
        """Flush events (handled by LlamaIndex/Langfuse SDK)."""
        pass

    def close(self) -> None:
        """Cleanup (handled by LlamaIndex/Langfuse SDK)."""
        pass
