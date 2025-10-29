"""
Official observer implementations for Ondine.

This module contains built-in observers for popular observability platforms:
- OpenTelemetry: Infrastructure monitoring (Jaeger, Datadog, etc.)
- Langfuse: LLM-specific observability (prompts, tokens, costs)
- LoggingObserver: Simple file/console logging
"""

from ondine.observability.observers.logging_observer import LoggingObserver
from ondine.observability.observers.opentelemetry_observer import (
    OpenTelemetryObserver,
)

# Langfuse is optional - only import if available
try:
    from ondine.observability.observers.langfuse_observer import LangfuseObserver

    __all__ = [
        "OpenTelemetryObserver",
        "LangfuseObserver",
        "LoggingObserver",
    ]
except ImportError:
    __all__ = [
        "OpenTelemetryObserver",
        "LoggingObserver",
    ]
