"""
OpenTelemetry observer for infrastructure monitoring.

Uses LiteLLM's native OpenTelemetry callbacks.
"""

import logging
import os
from typing import Any

import litellm

from ondine.observability.base import PipelineObserver
from ondine.observability.registry import observer

logger = logging.getLogger(__name__)


@observer("opentelemetry")
class OpenTelemetryObserver(PipelineObserver):
    """
    Observer that configures LiteLLM's native OpenTelemetry integration.

    LiteLLM automatically instruments:
    - ✅ All LLM calls (prompts, completions, tokens, latency)
    - ✅ Standard OTLP export (compatible with Jaeger, Honeycomb, Datadog, etc.)

    Configuration:
    - Uses standard environment variables (OTEL_EXPORTER_OTLP_ENDPOINT, etc.)
    - Or passes specific config in 'config' dict

    Example:
        observer = OpenTelemetryObserver(config={
            "service_name": "my-pipeline"
        })
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize OpenTelemetry observer.
        """
        super().__init__(config)

        # 1. Add 'otel' to LiteLLM callbacks if not present
        if "otel" not in litellm.callbacks:
            litellm.callbacks.append("otel")
            logger.info("Added 'otel' to LiteLLM callbacks")

        # 2. Map config to env vars (standard OTel SDK configuration)
        if self.config:
            # Allow overriding service name via config
            if "service_name" in self.config:
                os.environ["OTEL_SERVICE_NAME"] = self.config["service_name"]

            # Allow overriding endpoint via config
            if "endpoint" in self.config:
                os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = self.config["endpoint"]

        logger.info("OpenTelemetry observer initialized (LiteLLM native)")

    def on_llm_call(self, event: Any) -> None:
        """
        LLM calls are automatically traced by LiteLLM's native callback.
        """
        pass

    def flush(self) -> None:
        """Flush spans (handled by OpenTelemetry SDK)."""
        pass

    def close(self) -> None:
        """Cleanup (handled by OpenTelemetry SDK)."""
        pass
