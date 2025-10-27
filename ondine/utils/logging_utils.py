"""
Structured logging utilities.

Provides consistent logging configuration across the SDK using structlog.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict

# Track if logging has been configured
_logging_configured = False


def _compact_console_renderer(_: Any, __: str, event_dict: EventDict) -> str:
    """
    Custom console renderer with compact formatting (no padding).

    Format: TIMESTAMP [LEVEL] MESSAGE key=value key=value
    """
    # Get timestamp
    timestamp = event_dict.pop("timestamp", "")

    # Get log level and apply color
    level = event_dict.pop("level", "info").upper()
    level_colors = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    reset = "\033[0m"
    colored_level = f"{level_colors.get(level, '')}{level}{reset}"

    # Get event message
    event = event_dict.pop("event", "")

    # Build base message
    parts = []
    if timestamp:
        parts.append(timestamp)
    parts.append(f"[{colored_level}]")
    parts.append(event)

    # Add remaining key-value pairs
    for key, value in event_dict.items():
        if not key.startswith("_"):  # Skip internal keys
            parts.append(f"{key}={value}")

    return " ".join(parts)


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    include_timestamp: bool = True,
) -> None:
    """
    Configure structured logging for the SDK.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON output format
        include_timestamp: Include timestamps in logs
    """
    global _logging_configured

    # Set stdlib logging level
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

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
        # Use custom compact console renderer (no padding)
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
