"""
Intelligent Instructor mode detection for structured LLM output.

This module implements a layered approach to detect the best Instructor mode:
1. User explicit override (highest priority)
2. Special case detection (reasoning_effort, known limitations)
3. LiteLLM model capabilities lookup
4. Provider capability registry (fallback)
5. Safe default: JSON mode
"""

from enum import Enum
from typing import Any, Literal

import instructor
import litellm

from ondine.utils.logging_utils import get_logger

logger = get_logger(__name__)


class InstructorModeStrategy(str, Enum):
    """User-facing mode selection options."""

    AUTO = "auto"
    TOOLS = "tools"
    JSON = "json"
    JSON_SCHEMA = "json_schema"


# Provider capability registry for fallback detection
# This is used when LiteLLM's model info lookup fails
PROVIDER_CAPABILITIES: dict[str, dict[str, Any]] = {
    "openai": {
        "tools": True,
        "json": True,
        "json_schema": True,
    },
    "azure": {
        "tools": True,
        "json": True,
        # Reasoning models (o1, gpt-5-nano with reasoning_effort) don't support TOOLS
        "reasoning_models_tools": False,
    },
    "anthropic": {
        "tools": True,
        "json": True,
    },
    "groq": {
        # Groq has issues with function calling (generates XML)
        "tools": False,
        "json": True,
    },
    "together": {
        "tools": True,
        "json": True,
    },
    "mistral": {
        "tools": True,
        "json": True,
    },
    "cohere": {
        "tools": True,
        "json": True,
    },
    "gemini": {
        "tools": True,
        "json": True,
    },
    "vertex_ai": {
        "tools": True,
        "json": True,
    },
    "bedrock": {
        "tools": True,
        "json": True,
    },
    "ollama": {
        # Most Ollama models don't support function calling well
        "tools": False,
        "json": True,
    },
    "huggingface": {
        "tools": False,
        "json": True,
    },
    "cerebras": {
        "tools": True,
        "json": True,
    },
}


def detect_instructor_mode(
    model: str,
    extra_params: dict | None = None,
    user_override: InstructorModeStrategy
    | Literal["auto", "tools", "json", "json_schema"] = InstructorModeStrategy.AUTO,
    router_model_list: list | None = None,
) -> instructor.Mode:
    """
    Detect the best Instructor mode for structured output.

    Uses a layered approach:
    1. User explicit override (highest priority)
    2. Special case detection (reasoning_effort, known limitations)
    3. LiteLLM model capabilities lookup
    4. Provider capability registry (fallback)
    5. Safe default: JSON mode

    Args:
        model: The model name/identifier
        extra_params: Additional parameters passed to the model (e.g., reasoning_effort)
        user_override: User's explicit mode choice ("auto" for automatic detection)
        router_model_list: List of router deployments (for multi-model setups)

    Returns:
        The appropriate instructor.Mode for the model
    """
    extra_params = extra_params or {}

    # Normalize user_override to enum
    if isinstance(user_override, str):
        user_override = InstructorModeStrategy(user_override.lower())

    # =========================================================================
    # Layer 1: User Override (highest priority)
    # =========================================================================
    if user_override != InstructorModeStrategy.AUTO:
        mode = _strategy_to_instructor_mode(user_override)
        logger.debug(f"Using user-specified Instructor mode: {mode}")
        return mode

    # =========================================================================
    # Layer 2: Special Case Detection
    # =========================================================================

    # Check for reasoning_effort (Azure reasoning models don't support TOOLS)
    has_reasoning_effort = "reasoning_effort" in extra_params
    if router_model_list:
        has_reasoning_effort = has_reasoning_effort or any(
            "reasoning_effort" in d.get("litellm_params", {}) for d in router_model_list
        )

    if has_reasoning_effort:
        logger.debug("Reasoning model detected (reasoning_effort param) -> JSON mode")
        return instructor.Mode.JSON

    # =========================================================================
    # Layer 3: LiteLLM Model Capabilities Lookup
    # =========================================================================
    actual_model = _get_actual_model(model, router_model_list)

    try:
        model_info = litellm.get_model_info(actual_model)
        supports_function_calling = model_info.get("supports_function_calling", False)

        if supports_function_calling:
            logger.debug(
                f"LiteLLM reports '{actual_model}' supports function calling -> TOOLS mode"
            )
            return instructor.Mode.TOOLS
        logger.debug(
            f"LiteLLM reports '{actual_model}' doesn't support function calling -> JSON mode"
        )
        return instructor.Mode.JSON

    except Exception as e:
        logger.debug(f"LiteLLM model info lookup failed for '{actual_model}': {e}")

    # =========================================================================
    # Layer 4: Provider Capability Registry (Fallback)
    # =========================================================================
    provider = _extract_provider(actual_model)
    if provider and provider in PROVIDER_CAPABILITIES:
        caps = PROVIDER_CAPABILITIES[provider]
        if caps.get("tools", False):
            logger.debug(
                f"Provider registry: '{provider}' supports TOOLS -> TOOLS mode"
            )
            return instructor.Mode.TOOLS
        logger.debug(
            f"Provider registry: '{provider}' doesn't support TOOLS -> JSON mode"
        )
        return instructor.Mode.JSON

    # Known provider patterns (legacy fallback)
    model_lower = actual_model.lower()
    if "groq" in model_lower:
        logger.debug("Known provider pattern: Groq -> JSON mode (avoids XML issues)")
        return instructor.Mode.JSON
    if any(p in model_lower for p in ["gpt-4", "gpt-3.5", "claude"]):
        logger.debug("Known provider pattern: GPT/Claude -> TOOLS mode")
        return instructor.Mode.TOOLS

    # =========================================================================
    # Layer 5: Safe Default
    # =========================================================================
    logger.debug(f"Unknown model '{actual_model}', defaulting to JSON mode (safest)")
    return instructor.Mode.JSON


def _strategy_to_instructor_mode(strategy: InstructorModeStrategy) -> instructor.Mode:
    """Convert user strategy to instructor.Mode."""
    mapping = {
        InstructorModeStrategy.TOOLS: instructor.Mode.TOOLS,
        InstructorModeStrategy.JSON: instructor.Mode.JSON,
        InstructorModeStrategy.JSON_SCHEMA: instructor.Mode.JSON_SCHEMA,
    }
    return mapping.get(strategy, instructor.Mode.JSON)


def _get_actual_model(model: str, router_model_list: list | None) -> str:
    """Extract the actual model name from router config if available."""
    if router_model_list and len(router_model_list) > 0:
        # Get the first deployment's actual model
        return router_model_list[0].get("litellm_params", {}).get("model", model)
    return model


def _extract_provider(model: str) -> str | None:
    """Extract provider name from model string (e.g., 'azure/gpt-4' -> 'azure')."""
    model_lower = model.lower()

    # Handle explicit provider prefix (e.g., "azure/gpt-4", "groq/llama")
    if "/" in model_lower:
        provider = model_lower.split("/")[0]
        # Clean up common prefixes
        if provider in PROVIDER_CAPABILITIES:
            return provider

    # Infer provider from model name patterns
    provider_patterns = {
        "gpt-": "openai",
        "o1-": "openai",
        "claude": "anthropic",
        "gemini": "gemini",
        "mistral": "mistral",
        "llama": None,  # Could be many providers
        "command": "cohere",
    }

    for pattern, provider in provider_patterns.items():
        if pattern in model_lower and provider:
            return provider

    return None
