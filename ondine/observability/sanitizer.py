"""
PII sanitization for trace attributes.

Provides utilities to sanitize sensitive data in prompts and responses
before including them in distributed traces.
"""

import re
from typing import Any, Optional

from ondine.observability.events import LLMCallEvent

# Regex patterns for common PII
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
    "phone_us": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    "phone_intl": re.compile(
        r"\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
    ),
    "api_key": re.compile(
        r"\b(?:api[_-]?key|secret|token)[:\s=]+['\"]?([a-zA-Z0-9_\-]{16,})['\"]?\b",
        re.IGNORECASE,
    ),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def sanitize_text(
    text: str,
    patterns: Optional[dict[str, re.Pattern]] = None,
    replacement: str = "[REDACTED]",
) -> str:
    """
    Sanitize text by replacing PII patterns.

    Args:
        text: Text to sanitize
        patterns: Dictionary of pattern name -> regex pattern (uses defaults if None)
        replacement: Replacement string for matched patterns

    Returns:
        Sanitized text with PII replaced

    Examples:
        >>> sanitize_text("Email me at test@example.com")
        'Email me at [REDACTED]'
        >>> sanitize_text("SSN: 123-45-6789")
        'SSN: [REDACTED]'
    """
    if patterns is None:
        patterns = PII_PATTERNS

    sanitized = text
    for pattern_name, pattern in patterns.items():
        sanitized = pattern.sub(replacement, sanitized)

    return sanitized


def sanitize_prompt(prompt: str, include_prompts: bool = False) -> str:
    """
    Sanitize prompt text for tracing (PII-safe by default).

    Args:
        prompt: The original prompt text
        include_prompts: If True, return original prompt (opt-in)

    Returns:
        Sanitized prompt (hash) or original if opted in

    Examples:
        >>> sanitize_prompt("User email: test@example.com")
        '<sanitized-1234>'
        >>> sanitize_prompt("Test prompt", include_prompts=True)
        'Test prompt'
    """
    if include_prompts:
        return prompt

    # Return hash to detect duplicates without exposing content
    # Use modulo to keep hash short and readable
    hash_value = hash(prompt) % 10000
    return f"<sanitized-{hash_value}>"


def sanitize_response(response: str, include_prompts: bool = False) -> str:
    """
    Sanitize LLM response text for tracing (PII-safe by default).

    Args:
        response: The original response text
        include_prompts: If True, return original response (opt-in)

    Returns:
        Sanitized response (hash) or original if opted in

    Examples:
        >>> sanitize_response("SSN: 123-45-6789")
        '<sanitized-5678>'
        >>> sanitize_response("Safe response", include_prompts=True)
        'Safe response'
    """
    # Reuse same logic as prompt sanitization (DRY principle)
    return sanitize_prompt(response, include_prompts=include_prompts)


def sanitize_event(
    event: LLMCallEvent,
    config: Optional[dict[str, Any]] = None,
) -> LLMCallEvent:
    """
    Sanitize an LLM call event based on configuration.

    Args:
        event: LLM call event to sanitize
        config: Sanitization configuration:
            - sanitize_prompts: bool (default: True)
            - sanitize_completions: bool (default: True)
            - custom_patterns: dict[str, str] (additional regex patterns)
            - replacement: str (default: "[REDACTED]")

    Returns:
        New LLMCallEvent with sanitized fields

    Example:
        config = {
            "sanitize_prompts": True,
            "sanitize_completions": False,
            "custom_patterns": {"account_id": r"ACC-\d{6}"}
        }
        sanitized = sanitize_event(event, config)
    """
    if config is None:
        config = {}

    sanitize_prompts = config.get("sanitize_prompts", True)
    sanitize_completions = config.get("sanitize_completions", True)
    custom_patterns_dict = config.get("custom_patterns", {})
    replacement = config.get("replacement", "[REDACTED]")

    # Build combined patterns
    patterns = PII_PATTERNS.copy()
    for name, pattern_str in custom_patterns_dict.items():
        patterns[name] = re.compile(pattern_str)

    # Create new event with sanitized fields
    sanitized_prompt = event.prompt
    sanitized_completion = event.completion
    sanitized_rag_context = event.rag_context

    if sanitize_prompts:
        sanitized_prompt = sanitize_text(event.prompt, patterns, replacement)
        if event.rag_context:
            sanitized_rag_context = sanitize_text(
                event.rag_context, patterns, replacement
            )

    if sanitize_completions:
        sanitized_completion = sanitize_text(event.completion, patterns, replacement)

    # Return new event (events are immutable dataclasses)
    # We use replace() to create a new instance
    from dataclasses import replace

    return replace(
        event,
        prompt=sanitized_prompt,
        completion=sanitized_completion,
        rag_context=sanitized_rag_context,
    )
