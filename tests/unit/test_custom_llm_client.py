"""
Unit tests for custom LLM provider support.

Tests OpenAI-compatible client and custom client injection.
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from ondine.adapters.llm_client import OpenAICompatibleClient, create_llm_client
from ondine.core.models import LLMResponse
from ondine.core.specifications import LLMProvider, LLMSpec


class TestOpenAICompatibleClient:
    """Test suite for OpenAI-compatible custom providers."""

    def test_requires_base_url(self):
        """Should raise error if base_url not provided."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="base_url required"):
            LLMSpec(
                provider=LLMProvider.OPENAI_COMPATIBLE,
                model="llama-3.1-70b",
                temperature=0.7,
            )

    def test_accepts_valid_base_url(self):
        """Should accept valid base_url."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI_COMPATIBLE,
            model="llama-3.1-70b",
            base_url="https://api.together.xyz/v1",
            api_key="test-key",  # pragma: allowlist secret  # pragma: allowlist secret
        )

        with patch("ondine.adapters.llm_client.OpenAI"):
            client = OpenAICompatibleClient(spec)
            assert client.spec.base_url == "https://api.together.xyz/v1"

    def test_uses_provider_name_for_logging(self):
        """Should use provider_name field for identification."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI_COMPATIBLE,
            model="llama-3.1-70b",
            base_url="https://api.together.xyz/v1",
            provider_name="Together.AI",
            api_key="test-key",  # pragma: allowlist secret  # pragma: allowlist secret
        )

        with patch("ondine.adapters.llm_client.OpenAI"):
            client = OpenAICompatibleClient(spec)
            assert client.provider_name == "Together.AI"

    def test_defaults_provider_name_if_not_provided(self):
        """Should default to 'OpenAI-Compatible' if provider_name not set."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI_COMPATIBLE,
            model="llama-3.1-70b",
            base_url="http://localhost:11434/v1",
            api_key="test-key",  # pragma: allowlist secret  # pragma: allowlist secret
        )

        with patch("ondine.adapters.llm_client.OpenAI"):
            client = OpenAICompatibleClient(spec)
            assert client.provider_name == "OpenAI-Compatible"

    def test_uses_dummy_key_for_local_apis(self):
        """Should allow no API key for local APIs (Ollama)."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI_COMPATIBLE,
            model="llama3.1:70b",
            base_url="http://localhost:11434/v1",
            provider_name="Ollama-Local",
            # No api_key provided
        )

        with patch("ondine.adapters.llm_client.OpenAILike") as mock_openai_like:
            OpenAICompatibleClient(spec)
            # Should pass api_key to OpenAILike (required by library)  # pragma: allowlist secret
            mock_openai_like.assert_called_once()
            call_kwargs = mock_openai_like.call_args.kwargs
            assert "api_key" in call_kwargs  # pragma: allowlist secret
            # Should use "dummy" or env var
            assert (
                call_kwargs["api_key"] in ["dummy", None]  # pragma: allowlist secret
                or len(call_kwargs["api_key"]) > 0  # pragma: allowlist secret
            )

    @patch("ondine.adapters.llm_client.OpenAILike")
    def test_invoke_returns_llm_response(self, mock_openai_like_class):
        """Should invoke API and return LLMResponse."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI_COMPATIBLE,
            model="llama-3.1-70b",
            base_url="https://api.together.xyz/v1",
            provider_name="Together.AI",
            api_key="test-key",  # pragma: allowlist secret
            input_cost_per_1k_tokens=Decimal("0.0006"),
            output_cost_per_1k_tokens=Decimal("0.0006"),
        )

        # Mock the response with raw.usage for token tracking
        mock_response = MagicMock()
        mock_response.__str__ = MagicMock(
            return_value="The capital of France is Paris."
        )
        mock_response.raw = MagicMock()
        mock_response.raw.usage = MagicMock()
        mock_response.raw.usage.prompt_tokens = 10
        mock_response.raw.usage.completion_tokens = 8
        mock_response.raw.usage.prompt_tokens_details = None

        mock_client_instance = MagicMock()
        mock_client_instance.chat.return_value = mock_response
        mock_openai_like_class.return_value = mock_client_instance

        client = OpenAICompatibleClient(spec)
        response = client.invoke("What is the capital of France?")

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text == "The capital of France is Paris."
        assert response.tokens_in > 0
        assert response.tokens_out > 0
        assert "Together.AI" in response.model
        assert response.cost >= 0
        assert response.latency_ms >= 0

    def test_estimate_tokens(self):
        """Should estimate tokens using tiktoken."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI_COMPATIBLE,
            model="test-model",
            base_url="http://localhost:8000",
            api_key="test",  # pragma: allowlist secret  # pragma: allowlist secret
        )

        with patch("ondine.adapters.llm_client.OpenAI"):
            client = OpenAICompatibleClient(spec)
            tokens = client.estimate_tokens("Hello, world!")
            assert tokens > 0
            assert isinstance(tokens, int)

    def test_calculate_cost_with_custom_pricing(self):
        """Should calculate cost using custom pricing."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI_COMPATIBLE,
            model="llama-3.1-70b",
            base_url="https://api.together.xyz/v1",
            api_key="test-key",  # pragma: allowlist secret
            input_cost_per_1k_tokens=Decimal("0.0008"),
            output_cost_per_1k_tokens=Decimal("0.0008"),
        )

        with patch("ondine.adapters.llm_client.OpenAI"):
            client = OpenAICompatibleClient(spec)
            cost = client.calculate_cost(tokens_in=1000, tokens_out=500)
            expected = Decimal("0.0008") + (Decimal("0.0008") * Decimal("0.5"))
            assert cost == expected

    def test_calculate_cost_free_for_local_models(self):
        """Should support $0 cost for local models."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI_COMPATIBLE,
            model="llama3.1:70b",
            base_url="http://localhost:11434/v1",
            provider_name="Ollama-Local",
            input_cost_per_1k_tokens=Decimal("0.0"),
            output_cost_per_1k_tokens=Decimal("0.0"),
        )

        with patch("ondine.adapters.llm_client.OpenAI"):
            client = OpenAICompatibleClient(spec)
            cost = client.calculate_cost(tokens_in=1000, tokens_out=500)
            assert cost == Decimal("0.0")


class TestCustomLLMClientFactory:
    """Test suite for factory function with custom providers."""

    def test_factory_creates_openai_compatible_client(self):
        """Factory should create OpenAICompatibleClient for openai_compatible provider."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI_COMPATIBLE,
            model="test-model",
            base_url="http://localhost:8000",
            api_key="test",  # pragma: allowlist secret  # pragma: allowlist secret
        )

        with patch("ondine.adapters.llm_client.OpenAI"):
            client = create_llm_client(spec)
            assert isinstance(client, OpenAICompatibleClient)

    def test_factory_still_creates_existing_providers(self):
        """Factory should still work with existing providers."""
        from ondine.adapters.llm_client import GroqClient, OpenAIClient

        # OpenAI
        spec_openai = LLMSpec(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test",  # pragma: allowlist secret  # pragma: allowlist secret
        )
        with patch("ondine.adapters.llm_client.OpenAI"):
            client = create_llm_client(spec_openai)
            assert isinstance(client, OpenAIClient)

        # Groq
        spec_groq = LLMSpec(
            provider=LLMProvider.GROQ,
            model="llama-3.3-70b-versatile",
            api_key="test",  # pragma: allowlist secret
        )
        with patch("ondine.adapters.llm_client.Groq"):
            client = create_llm_client(spec_groq)
            assert isinstance(client, GroqClient)


class TestLLMSpecValidation:
    """Test suite for LLMSpec validation with custom providers."""

    def test_base_url_required_for_openai_compatible(self):
        """Should validate that base_url is required for openai_compatible."""
        with pytest.raises(ValueError, match="base_url required"):
            LLMSpec(
                provider=LLMProvider.OPENAI_COMPATIBLE,
                model="test-model",
                # Missing base_url
            )

    def test_base_url_optional_for_standard_providers(self):
        """Should allow base_url to be None for standard providers."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test",  # pragma: allowlist secret
            # base_url not provided - should be fine
        )
        assert spec.base_url is None

    def test_accepts_all_custom_fields(self):
        """Should accept base_url and provider_name fields."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI_COMPATIBLE,
            model="llama-3.1-70b",
            base_url="https://api.together.xyz/v1",
            provider_name="Together.AI",
            api_key="test-key",  # pragma: allowlist secret
            temperature=0.7,
            max_tokens=2000,
            input_cost_per_1k_tokens=Decimal("0.0006"),
            output_cost_per_1k_tokens=Decimal("0.0006"),
        )

        assert spec.provider == LLMProvider.OPENAI_COMPATIBLE
        assert spec.base_url == "https://api.together.xyz/v1"
        assert spec.provider_name == "Together.AI"
        assert spec.model == "llama-3.1-70b"

    def test_base_url_validates_http_scheme(self):
        """Should reject base_url without http(s) scheme."""
        with pytest.raises(ValueError, match="base_url must start with http"):
            LLMSpec(
                provider=LLMProvider.OPENAI_COMPATIBLE,
                model="test-model",
                base_url="ftp://invalid.com/v1",  # Invalid scheme
            )

    def test_base_url_validates_host_presence(self):
        """Should reject base_url without a host."""
        with pytest.raises(ValueError, match="base_url must include a host"):
            LLMSpec(
                provider=LLMProvider.OPENAI_COMPATIBLE,
                model="test-model",
                base_url="http://",  # No host
            )

    def test_base_url_accepts_localhost(self):
        """Should accept localhost URLs."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI_COMPATIBLE,
            model="llama3.1:70b",
            base_url="http://localhost:11434/v1",
        )
        assert spec.base_url == "http://localhost:11434/v1"

    def test_base_url_accepts_https(self):
        """Should accept HTTPS URLs."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI_COMPATIBLE,
            model="test-model",
            base_url="https://api.example.com/v1",
        )
        assert spec.base_url == "https://api.example.com/v1"
