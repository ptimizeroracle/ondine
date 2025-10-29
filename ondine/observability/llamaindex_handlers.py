"""
LlamaIndex handler management for Ondine observability.

This module configures LlamaIndex's built-in observability handlers
via Ondine's observer API.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LlamaIndexHandlerManager:
    """
    Manages LlamaIndex global handlers.

    LlamaIndex provides built-in handlers for various observability platforms.
    This manager configures them based on Ondine's observer configuration.
    """

    _active_handler: str | None = None

    @classmethod
    def configure_handler(cls, handler_type: str, config: dict[str, Any]) -> None:
        """
        Configure a LlamaIndex global handler.

        Note: LlamaIndex only supports ONE global handler at a time.
        Calling this multiple times will replace the previous handler.
        For multiple observers, use configure_multi_handler() instead.

        Args:
            handler_type: Handler type ("opentelemetry", "langfuse", "simple", etc.)
            config: Handler-specific configuration

        Raises:
            ImportError: If LlamaIndex or handler dependencies not installed
        """
        from llama_index.core import set_global_handler

        try:
            if handler_type == "opentelemetry":
                # OpenTelemetry handler for infrastructure monitoring
                set_global_handler("opentelemetry", **config)
                logger.info("Configured LlamaIndex OpenTelemetry handler")

            elif handler_type == "langfuse":
                # Langfuse handler for LLM-specific observability
                public_key = config.get("public_key")
                secret_key = config.get("secret_key")
                host = config.get("host", "https://cloud.langfuse.com")

                if not public_key or not secret_key:
                    raise ValueError(
                        "Langfuse requires 'public_key' and 'secret_key' in config"
                    )

                set_global_handler(
                    "langfuse",
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                logger.info(f"Configured LlamaIndex Langfuse handler (host={host})")

            elif handler_type == "simple" or handler_type == "logging":
                # Simple handler for console logging
                set_global_handler("simple")
                logger.info("Configured LlamaIndex Simple handler (console logging)")

            elif handler_type == "arize_phoenix":
                # Arize Phoenix handler
                set_global_handler("arize_phoenix", **config)
                logger.info("Configured LlamaIndex Arize Phoenix handler")

            elif handler_type == "wandb":
                # Weights & Biases handler
                set_global_handler("wandb", **config)
                logger.info("Configured LlamaIndex W&B handler")

            else:
                logger.warning(
                    f"Unknown LlamaIndex handler type: {handler_type}. "
                    f"Supported: opentelemetry, langfuse, simple, arize_phoenix, wandb"
                )
                return

            cls._active_handler = handler_type

        except ImportError as e:
            logger.error(
                f"Failed to configure {handler_type} handler: {e}. "
                f"Install required dependencies or use a different observer."
            )
            raise

    @classmethod
    def configure_multi_handler(
        cls, handlers: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """
        Configure multiple LlamaIndex handlers simultaneously.

        Uses LlamaIndex's lower-level dispatcher API to register multiple
        event handlers at once, since set_global_handler() only supports one.

        Args:
            handlers: List of (handler_type, config) tuples

        Note: This is more advanced and may require custom event handler
        implementations for some platforms.
        """
        from llama_index.core.instrumentation import get_dispatcher

        get_dispatcher()

        logger.info(f"Configuring {len(handlers)} LlamaIndex handlers via dispatcher")

        # For now, just use the last handler as global
        # TODO: Implement proper multi-handler support using dispatcher.add_event_handler()
        if handlers:
            last_handler_type, last_config = handlers[-1]
            cls.configure_handler(last_handler_type, last_config)

            if len(handlers) > 1:
                logger.warning(
                    f"Multiple LlamaIndex handlers requested but only '{last_handler_type}' "
                    f"is active. Full multi-handler support coming soon."
                )

    @classmethod
    def get_active_handler(cls) -> str | None:
        """
        Get the currently active handler type.

        Returns:
            Handler type string or None if no handler active
        """
        return cls._active_handler

    @classmethod
    def reset_handler(cls) -> None:
        """Reset/disable the global handler."""
        # LlamaIndex doesn't have an official API to disable handlers
        # but setting it to None or creating a no-op handler works
        cls._active_handler = None
        logger.info("LlamaIndex handler reset")
