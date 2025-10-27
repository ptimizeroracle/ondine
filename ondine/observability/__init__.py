"""
Observability toolkit for Ondine pipelines.

Provides distributed tracing with OpenTelemetry for production debugging
and performance monitoring.

Usage:
    >>> from ondine.observability import enable_tracing, TracingObserver
    >>> enable_tracing(exporter="jaeger", endpoint="http://localhost:14268")
    >>> # Traces will be exported to Jaeger
"""

# Check if OpenTelemetry is available (optional dependency)
try:
    from opentelemetry import trace  # noqa: F401

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

# Conditional imports based on availability
if OBSERVABILITY_AVAILABLE:
    from .observer import TracingObserver
    from .tracer import disable_tracing, enable_tracing, is_tracing_enabled

    __all__ = [
        "enable_tracing",
        "disable_tracing",
        "is_tracing_enabled",
        "TracingObserver",
    ]
else:
    # Graceful degradation - provide helpful error messages
    def enable_tracing(*args, **kwargs):
        """Placeholder when observability is not installed."""
        raise ImportError(
            "Observability features require OpenTelemetry.\n"
            "Install with: pip install ondine[observability]\n"
            "Or: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-jaeger"
        )

    def disable_tracing(*args, **kwargs):
        """Placeholder when observability is not installed."""
        raise ImportError(
            "Observability features require: pip install ondine[observability]"
        )

    def is_tracing_enabled() -> bool:
        """Always returns False when observability is not installed."""
        return False

    class TracingObserver:  # type: ignore
        """Placeholder when observability is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TracingObserver requires OpenTelemetry.\n"
                "Install with: pip install ondine[observability]"
            )

    __all__ = [
        "enable_tracing",
        "disable_tracing",
        "is_tracing_enabled",
        "TracingObserver",
    ]
