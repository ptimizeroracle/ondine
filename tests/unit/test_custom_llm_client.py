"""
Unit tests for custom LLM client support via UnifiedLiteLLMClient.

NOTE: Custom endpoints (Ollama, vLLM, Together.AI) are now handled by
UnifiedLiteLLMClient using LiteLLM's native support for custom base_url.

We test OUR integration code, not LiteLLM internals.
"""

from unittest.mock import patch

from ondine.adapters.unified_litellm_client import UnifiedLiteLLMClient
from ondine.core.specifications import LLMProvider, LLMSpec


class TestCustomEndpointViaUnifiedClient:
    """Test custom OpenAI-compatible endpoints via UnifiedLiteLLMClient."""

    def test_custom_endpoint_model_identifier(self):
        """Test that custom endpoints use correct model identifier."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI_COMPATIBLE,
            model="llama-3.1-70b",
            base_url="https://api.together.xyz/v1",
            api_key="test-key",  # pragma: allowlist secret
        )

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

            # Should use openai_compatible as provider
            assert client.model_identifier == "openai_compatible/llama-3.1-70b"

    def test_ollama_local_model_identifier(self):
        """Test Ollama local endpoint."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI_COMPATIBLE,
            model="llama3.1:70b",
            base_url="http://localhost:11434/v1",
            # No API key needed for local
        )

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

            assert client.model_identifier == "openai_compatible/llama3.1:70b"


# NOTE: Removed extensive OpenAI-compatible client tests
# Rationale: Those tests were testing LlamaIndex's OpenAILike wrapper behavior.
# With UnifiedLiteLLMClient, we use LiteLLM directly, and we don't need to
# test LiteLLM's custom endpoint support - that's their responsibility.
# We only test OUR code: model identifier formatting and factory routing.
