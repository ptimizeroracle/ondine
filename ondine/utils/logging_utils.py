"""
Structured logging utilities.

Provides consistent logging configuration across the SDK using structlog.
"""

import logging
import os
import sys
from typing import Any

import structlog

# Track if logging has been configured
_logging_configured = False


def configure_logging(
    level: str | None = None,
    json_format: bool = False,
    include_timestamp: bool = True,
) -> None:
    """
    Configure structured logging for the SDK.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to ONDINE_LOG_LEVEL env var, or INFO.
        json_format: Use JSON output format
        include_timestamp: Include timestamps in logs
    """
    global _logging_configured

    # Resolve level from env var if not provided
    if level is None:
        level = os.getenv("ONDINE_LOG_LEVEL", "INFO").upper()

    # Set stdlib logging level
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # CRITICAL: Silence noisy libraries unless ONDINE_TRACE is enabled
    # We only want to see ondine's debug logs, not every HTTP request
    trace_mode = os.getenv("ONDINE_TRACE", "false").lower() in ("true", "1", "yes")

    if not trace_mode:
        for logger_name in [
            "litellm",
            "LiteLLM Router",
            "LiteLLM Proxy",
            "httpx",
            "httpcore",
            "urllib3",
            "asyncio",
            "pydantic",
            "openai",
            "instructor",
        ]:
            # In standard DEBUG/INFO mode, suppress library noise (to WARNING)
            # We only want CRITICAL suppression for internal modules we don't control well
            if logger_name in ["pydantic", "asyncio"]:
                logging.getLogger(logger_name).setLevel(logging.CRITICAL)
            else:
                logging.getLogger(logger_name).setLevel(logging.WARNING)

            logging.getLogger(logger_name).propagate = False
    else:
        # In TRACE mode, let everything through (useful for debugging headers/raw output)
        # We print a warning so the user knows why their console is flooded
        if not _logging_configured:
            print("⚠️ ONDINE_TRACE enabled: External library logs will be visible.")

    # Configure structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]

    if include_timestamp:
        processors.append(structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"))

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Try to use Rich for pretty logging if available (prevents progress bar glitches)
        try:
            from rich.logging import RichHandler

            # Reconfigure basicConfig to use RichHandler
            logging.getLogger().handlers = [
                RichHandler(
                    rich_tracebacks=True,
                    markup=True,
                    show_time=include_timestamp,
                    show_path=False,
                )
            ]
            # Use ConsoleRenderer for structlog which works well with RichHandler
            # Enable padding for aligned logs (DEBUG   , INFO    )
            processors.append(
                structlog.dev.ConsoleRenderer(colors=True, pad_level=True)
            )
        except ImportError:
            # Fallback to custom renderer
            from ondine.utils.logging_utils import _compact_console_renderer

            processors.append(_compact_console_renderer)

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _logging_configured = True


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Auto-configures logging on first use if not already configured.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger
    """
    global _logging_configured

    # Auto-configure logging on first use
    if not _logging_configured:
        configure_logging()

    return structlog.get_logger(name)


def sanitize_for_logging(data: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize sensitive data for logging.

    Args:
        data: Dictionary potentially containing sensitive data

    Returns:
        Sanitized dictionary
    """
    sensitive_keys = {
        "api_key",
        "password",
        "secret",
        "token",
        "authorization",
        "credential",
    }

    sanitized = {}
    for key, value in data.items():
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            sanitized[key] = "***REDACTED***"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_for_logging(value)
        else:
            sanitized[key] = value

    return sanitized
