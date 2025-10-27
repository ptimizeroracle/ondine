"""
OpenTelemetry tracer setup and management.

Provides simple API for enabling/disabling distributed tracing with
console or Jaeger exporters.
"""

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# Module-level state (simple global state for opt-in tracing)
_TRACING_ENABLED = False
_TRACER_PROVIDER: TracerProvider | None = None
_TRACER: trace.Tracer | None = None


def get_tracer() -> trace.Tracer:
    """
    Get current tracer instance (or no-op if disabled).

    Returns:
        Active tracer or no-op tracer if tracing is disabled

    Examples:
        >>> tracer = get_tracer()
        >>> with tracer.start_as_current_span("my-operation"):
        ...     # Do work
        ...     pass
    """
    global _TRACER

    if not _TRACING_ENABLED or _TRACER is None:
        # Return no-op tracer that does nothing
        return trace.get_tracer(__name__)

    return _TRACER


def enable_tracing(
    exporter: str = "console",
    endpoint: str | None = None,
    service_name: str = "ondine-pipeline",
) -> None:
    """
    Enable distributed tracing (opt-in).

    Args:
        exporter: Exporter type ("console" or "jaeger")
        endpoint: Jaeger endpoint (required if exporter="jaeger")
        service_name: Service name for traces

    Examples:
        >>> # Console exporter (for development)
        >>> enable_tracing(exporter="console")

        >>> # Jaeger exporter (for production)
        >>> enable_tracing(
        ...     exporter="jaeger",
        ...     endpoint="http://localhost:14268/api/traces"
        ... )
    """
    global _TRACING_ENABLED, _TRACER_PROVIDER, _TRACER

    # Make idempotent - clean up existing tracing if already enabled
    if _TRACING_ENABLED:
        disable_tracing()

    # Create resource with service name
    resource = Resource.create(attributes={"service.name": service_name})

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Configure exporter
    if exporter == "console":
        span_exporter = ConsoleSpanExporter()
    elif exporter == "jaeger":
        if endpoint is None:
            raise ValueError("endpoint is required for Jaeger exporter")
        span_exporter = JaegerExporter(
            collector_endpoint=endpoint,
        )
    else:
        raise ValueError(f"Unknown exporter: {exporter}. Use 'console' or 'jaeger'")

    # Add span processor (gracefully handle export failures)
    try:
        processor = BatchSpanProcessor(span_exporter)
        provider.add_span_processor(processor)
    except Exception as e:
        # Don't break pipeline if exporter setup fails
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to setup span processor: {e}. Tracing will be disabled")
        return

    # Set global provider
    trace.set_tracer_provider(provider)

    # Store module state
    _TRACER_PROVIDER = provider
    _TRACER = trace.get_tracer(__name__)
    _TRACING_ENABLED = True


def disable_tracing() -> None:
    """
    Disable tracing and cleanup resources.

    Examples:
        >>> enable_tracing(exporter="console")
        >>> # ... do work ...
        >>> disable_tracing()
    """
    global _TRACING_ENABLED, _TRACER_PROVIDER, _TRACER

    if _TRACER_PROVIDER is not None:
        # Flush any pending spans
        try:
            _TRACER_PROVIDER.shutdown()
        except Exception:  # nosec B110
            # Silently ignore shutdown errors - tracing cleanup is non-critical
            # Prevents exceptions during cleanup from breaking application shutdown
            pass

    _TRACING_ENABLED = False
    _TRACER_PROVIDER = None
    _TRACER = None


def is_tracing_enabled() -> bool:
    """
    Check if tracing is currently enabled.

    Returns:
        True if tracing is enabled, False otherwise

    Examples:
        >>> is_tracing_enabled()
        False
        >>> enable_tracing(exporter="console")
        >>> is_tracing_enabled()
        True
    """
    return _TRACING_ENABLED
