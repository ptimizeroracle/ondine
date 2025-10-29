"""
Observability toolkit for Ondine pipelines.

Provides event-driven observability with support for multiple backends:
- OpenTelemetry for infrastructure monitoring (Jaeger, Datadog, etc.)
- Langfuse for LLM-specific observability (prompts, tokens, costs)
- Logging for simple console/file output

Usage:
    >>> from ondine import PipelineBuilder
    >>>
    >>> # Add observability to pipeline
    >>> pipeline = (
    ...     PipelineBuilder.create()
    ...     .from_csv("data.csv", ...)
    ...     .with_prompt("...")
    ...     .with_llm(provider="openai", model="gpt-4o-mini")
    ...     .with_observer("langfuse", config={
    ...         "public_key": "pk-lf-...",
    ...         "secret_key": "sk-lf-..."
    ...     })
    ...     .build()
    ... )

    >>> # Create custom observers
    >>> from ondine.observability import PipelineObserver, observer
    >>>
    >>> @observer("custom")
    >>> class CustomObserver(PipelineObserver):
    ...     def on_llm_call(self, event):
    ...         print(f"LLM: {event.model} - ${event.cost}")
"""

# Core observability infrastructure (always available)
from ondine.observability.base import PipelineObserver
from ondine.observability.dispatcher import ObserverDispatcher
from ondine.observability.events import (
    ErrorEvent,
    LLMCallEvent,
    PipelineEndEvent,
    PipelineStartEvent,
    StageEndEvent,
    StageStartEvent,
)

# Import official observers (will auto-register)
from ondine.observability.observers import (  # noqa: F401
    LoggingObserver,
    OpenTelemetryObserver,
)
from ondine.observability.registry import ObserverRegistry, observer
from ondine.observability.sanitizer import sanitize_event, sanitize_text

# Langfuse is optional - only import if available
try:
    from ondine.observability.observers import LangfuseObserver  # noqa: F401

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

# Legacy TracingObserver support (backward compatibility)
try:
    from ondine.observability.observer import TracingObserver  # noqa: F401
    from ondine.observability.tracer import (  # noqa: F401
        disable_tracing,
        enable_tracing,
        is_tracing_enabled,
    )

    LEGACY_OBSERVABILITY = True
except ImportError:
    LEGACY_OBSERVABILITY = False

# Build __all__ dynamically
__all__ = [
    # Core infrastructure
    "PipelineObserver",
    "ObserverRegistry",
    "ObserverDispatcher",
    "observer",
    # Event models
    "PipelineStartEvent",
    "StageStartEvent",
    "LLMCallEvent",
    "StageEndEvent",
    "ErrorEvent",
    "PipelineEndEvent",
    # Sanitization
    "sanitize_text",
    "sanitize_event",
    # Official observers
    "OpenTelemetryObserver",
    "LoggingObserver",
]

if LANGFUSE_AVAILABLE:
    __all__.append("LangfuseObserver")

if LEGACY_OBSERVABILITY:
    __all__.extend(
        [
            "TracingObserver",
            "enable_tracing",
            "disable_tracing",
            "is_tracing_enabled",
        ]
    )
