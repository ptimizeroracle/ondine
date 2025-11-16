"""Model context window limits registry.

Provides context window limits for various LLM models to enable
intelligent batch size validation and optimization.
"""

from ondine.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Model context window limits (in tokens)
# Source: Official provider documentation as of Nov 2024
MODEL_CONTEXT_LIMITS = {
    # OpenAI models
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o-2024-08-06": 128000,
    "gpt-4o-2024-11-20": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385,
    # Anthropic models
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-sonnet-20240620": 200000,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-sonnet-4": 200000,
    "claude-opus-4": 200000,
    # Groq models
    "llama-3.1-70b-versatile": 131072,
    "llama-3.1-8b-instant": 131072,
    "llama-3.3-70b-versatile": 131072,
    "mixtral-8x7b-32768": 32768,
    "gemma-7b-it": 8192,
    # Meta Llama models (general)
    "meta-llama/Meta-Llama-3.1-70B-Instruct": 131072,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 131072,
    "meta-llama/Llama-3.1-70B-Instruct": 131072,
    "meta-llama/Llama-3.1-8B-Instruct": 131072,
    # Moonshot models
    "kimi-k2-0905-preview": 128000,
    "moonshot-v1-8k": 8000,
    "moonshot-v1-32k": 32000,
    "moonshot-v1-128k": 128000,
    # Mistral models
    "mistral-7b-instruct": 8192,
    "mixtral-8x7b-instruct": 32768,
    "mixtral-8x22b-instruct": 65536,
    # Qwen models
    "qwen-7b": 8192,
    "qwen-14b": 8192,
    "qwen-72b": 32768,
    # Default fallback
    "default": 4096,
}


def get_context_limit(model: str) -> int:
    """Get context window limit for a model.

    Args:
        model: Model identifier

    Returns:
        Context window limit in tokens

    Example:
        >>> get_context_limit("gpt-4o-mini")
        128000
        >>> get_context_limit("unknown-model")
        4096  # Default fallback
    """
    # Direct lookup
    if model in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[model]

    # Fuzzy matching for model variants
    model_lower = model.lower()

    # Check for partial matches
    for known_model, limit in MODEL_CONTEXT_LIMITS.items():
        if known_model.lower() in model_lower or model_lower in known_model.lower():
            logger.debug(f"Fuzzy matched '{model}' to '{known_model}' (limit: {limit})")
            return limit

    # Fallback to default
    logger.warning(
        f"Unknown model '{model}', using default context limit: "
        f"{MODEL_CONTEXT_LIMITS['default']} tokens"
    )
    return MODEL_CONTEXT_LIMITS["default"]


def validate_batch_size(
    model: str,
    batch_size: int,
    avg_prompt_tokens: int,
    safety_margin: float = 0.8,
) -> tuple[bool, str | None]:
    """Validate batch size against model context window.

    Args:
        model: Model identifier
        batch_size: Proposed batch size
        avg_prompt_tokens: Average tokens per prompt
        safety_margin: Use only this fraction of context window (default: 0.8)

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, error_message) if invalid

    Example:
        >>> validate_batch_size("gpt-4o-mini", 100, 500)
        (True, None)  # 100 * 500 = 50K < 128K * 0.8
        >>> validate_batch_size("gpt-4o-mini", 1000, 500)
        (False, "Batch too large: ...")  # 1000 * 500 = 500K > 128K * 0.8
    """
    context_limit = get_context_limit(model)
    safe_limit = int(context_limit * safety_margin)

    estimated_tokens = batch_size * avg_prompt_tokens

    if estimated_tokens > safe_limit:
        error = (
            f"Batch size too large: {batch_size} rows × {avg_prompt_tokens} tokens/row "
            f"= {estimated_tokens} tokens, exceeds {safety_margin * 100:.0f}% of "
            f"context window ({safe_limit} tokens). "
            f"Reduce batch_size to {safe_limit // avg_prompt_tokens} or less."
        )
        return False, error

    return True, None


def suggest_optimal_batch_size(
    model: str,
    avg_prompt_tokens: int,
    safety_margin: float = 0.8,
) -> int:
    """Suggest optimal batch size for a model.

    Args:
        model: Model identifier
        avg_prompt_tokens: Average tokens per prompt
        safety_margin: Use only this fraction of context window

    Returns:
        Suggested batch size

    Example:
        >>> suggest_optimal_batch_size("gpt-4o-mini", 500)
        204  # (128000 * 0.8) / 500
    """
    context_limit = get_context_limit(model)
    safe_limit = int(context_limit * safety_margin)

    suggested = safe_limit // avg_prompt_tokens

    # Cap at reasonable maximum
    return min(suggested, 500)
