"""
Langfuse observer for LLM-specific observability.

Uses LiteLLM's native Langfuse integration.
"""

import logging
import os
from typing import Any

import litellm

from ondine.observability.base import PipelineObserver
from ondine.observability.registry import observer

logger = logging.getLogger(__name__)


@observer("langfuse")
class LangfuseObserver(PipelineObserver):
    """
    Observer that configures LiteLLM's native Langfuse integration.

    LiteLLM automatically tracks:
    - ✅ Full prompts and completions
    - ✅ Token usage and costs
    - ✅ Latency metrics
    - ✅ Model information
    - ✅ Prompt versioning (via Langfuse)

    Configuration:
    - Requires environment variables: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
    - Optional: LANGFUSE_HOST (defaults to cloud)
    - Or pass in 'config' dict which will be mapped to env vars

    Example:
        observer = LangfuseObserver(config={
            "public_key": "pk-lf-...",
            "secret_key": "sk-lf-...",
            "host": "https://cloud.langfuse.com"
        })
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Langfuse observer.
        """
        super().__init__(config)

        # 1. Add 'langfuse' to LiteLLM callbacks if not present
        if "langfuse" not in litellm.success_callback:
            litellm.success_callback.append("langfuse")
        if "langfuse" not in litellm.failure_callback:
            litellm.failure_callback.append("langfuse")
            logger.info("Added 'langfuse' to LiteLLM callbacks")

        # 2. Map config to env vars (LiteLLM/Langfuse SDK relies on these)
        if self.config:
            if self.config.get("public_key"):
                os.environ["LANGFUSE_PUBLIC_KEY"] = self.config["public_key"]
            if self.config.get("secret_key"):
                os.environ["LANGFUSE_SECRET_KEY"] = self.config["secret_key"]
            if self.config.get("host"):
                os.environ["LANGFUSE_HOST"] = self.config["host"]

        logger.info("Langfuse observer initialized (LiteLLM native)")

    def on_llm_call(self, event: Any) -> None:
        """
        LLM calls are automatically tracked by LiteLLM's native callback.
        """
        pass

    def flush(self) -> None:
        """Flush events (handled by Langfuse SDK background worker)."""
        import langfuse

        # Attempt to flush if the SDK exposes it, otherwise rely on atexit
        try:
            if hasattr(langfuse, "flush"):
                langfuse.flush()
        except Exception:
            pass

    def close(self) -> None:
        """Cleanup."""
        self.flush()
