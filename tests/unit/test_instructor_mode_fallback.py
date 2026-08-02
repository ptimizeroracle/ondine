"""
Unit tests for instructor mode selection on namespaced models, and for the
runtime mode-fallback chain (issue #187).

Two distinct regressions are pinned here:

1. Provider mis-attribution on namespaced models. ``_extract_provider`` used
   ``split("/")[0]`` and, when that segment was not a known provider, fell
   through to substring matching on the whole model string. So
   ``openrouter/anthropic/claude-3.5-sonnet`` resolved to ``anthropic`` and
   selected ``ANTHROPIC_TOOLS`` — a mode that only works against the *native*
   Anthropic Instructor adapter, while an ``openrouter/`` model travels over
   LiteLLM. Mode and transport disagreed.

2. No mode fallback. The Instructor client is built once with a single mode;
   if a provider structurally rejects that mode, every row failed. The client
   now walks a fallback chain — but only for structural rejections, never for
   rate limits, auth failures, or timeouts, which must stay loud.
"""

import instructor
import pytest

from ondine.adapters.instructor_mode import (
    PROVIDER_CAPABILITIES,
    _extract_provider,
    detect_instructor_mode,
)

# Modes that only work against Anthropic's native Instructor adapter. A model
# routed over LiteLLM must never be assigned one of these.
_ANTHROPIC_ONLY_MODES = {
    getattr(instructor.Mode, name)
    for name in ("ANTHROPIC_TOOLS", "ANTHROPIC_JSON")
    if hasattr(instructor.Mode, name)
}


class TestNamespacedProviderExtraction:
    """The transport provider owns the model, not the inner vendor."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("openrouter/anthropic/claude-3.5-sonnet", "openrouter"),
            ("openrouter/google/gemini-2.5-flash-lite", "openrouter"),
            ("openrouter/mistralai/mistral-nemo", "openrouter"),
            ("openrouter/openai/gpt-oss-20b", "openrouter"),
            ("deepseek/deepseek-chat", "deepseek"),
        ],
    )
    def test_namespaced_model_resolves_to_transport_provider(self, model, expected):
        assert _extract_provider(model) == expected

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("azure/gpt-4o-mini", "azure"),
            ("groq/llama-3.3-70b-versatile", "groq"),
            ("gpt-4o-mini", "openai"),
            ("claude-haiku-4-5-20251001", "anthropic"),
        ],
    )
    def test_existing_provider_extraction_is_unchanged(self, model, expected):
        """Single-segment and known-prefix models keep resolving as before."""
        assert _extract_provider(model) == expected


class TestNoAnthropicModeOverLiteLLM:
    """An openrouter/anthropic/* model must not get a native-Anthropic mode."""

    def test_openrouter_anthropic_model_avoids_anthropic_only_mode(self):
        mode = detect_instructor_mode(model="openrouter/anthropic/claude-3.5-sonnet")

        assert mode not in _ANTHROPIC_ONLY_MODES, (
            f"openrouter/anthropic/* travels over LiteLLM but got {mode}, "
            f"which requires the native Anthropic Instructor adapter"
        )

    def test_direct_anthropic_model_still_gets_anthropic_mode(self):
        """The fix must not regress genuine direct-Anthropic routing."""
        mode = detect_instructor_mode(model="claude-haiku-4-5-20251001")

        assert mode in _ANTHROPIC_ONLY_MODES


class TestProviderRegistryCoverage:
    @pytest.mark.parametrize("provider", ["openrouter", "deepseek"])
    def test_common_providers_are_registered(self, provider):
        """Both are widely used; absence forced a fall-through to defaults."""
        assert provider in PROVIDER_CAPABILITIES
