"""
PII sanitization for trace attributes.

Provides utilities to sanitize sensitive data in prompts and responses
before including them in distributed traces.
"""


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
