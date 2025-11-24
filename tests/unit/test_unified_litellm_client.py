"""
Unit tests for UnifiedLiteLLMClient (native LiteLLM integration).

Tests the new unified client that uses litellm.acompletion directly
without LlamaIndex wrappers.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ondine.adapters.unified_litellm_client import UnifiedLiteLLMClient
from ondine.core.models import LLMResponse
from ondine.core.specifications import LLMProvider, LLMSpec


@pytest.fixture
def openai_spec():
    """OpenAI specification for testing."""
    return LLMSpec(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        api_key="test-key",  # pragma: allowlist secret  # pragma: allowlist secret
        temperature=0.7,
        max_tokens=1000,
    )


@pytest.fixture
def groq_spec():
    """Groq specification for testing."""
    return LLMSpec(
        provider=LLMProvider.GROQ,
        model="llama-3.3-70b-versatile",
        api_key="test-key",  # pragma: allowlist secret  # pragma: allowlist secret
        temperature=0.1,
        max_tokens=2000,
    )


@pytest.fixture
def azure_spec():
    """Azure OpenAI specification for testing."""
    return LLMSpec(
        provider=LLMProvider.AZURE_OPENAI,
        model="gpt-4",
        azure_endpoint="https://test.openai.azure.com/",
        azure_deployment="gpt-4-deployment",
        api_key="test-key",  # pragma: allowlist secret  # pragma: allowlist secret
    )


class TestUnifiedLiteLLMClientInit:
    """Test client initialization."""

    def test_init_openai(self, openai_spec):
        """Test initialization with OpenAI provider."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(openai_spec)

            assert client.model == "gpt-4o-mini"
            assert client.temperature == 0.7
            assert client.max_tokens == 1000
            assert client.model_identifier == "openai/gpt-4o-mini"

    def test_init_groq(self, groq_spec):
        """Test initialization with Groq provider."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(groq_spec)

            assert client.model == "llama-3.3-70b-versatile"
            assert client.model_identifier == "groq/llama-3.3-70b-versatile"

    def test_init_azure(self, azure_spec):
        """Test initialization with Azure OpenAI."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(azure_spec)

            assert client.model_identifier == "azure/gpt-4-deployment"

    def test_sets_api_key_env_var(self, openai_spec):
        """Test that API key is set in environment."""
        env_dict = {}
        with patch("ondine.adapters.unified_litellm_client.os.environ", env_dict):
            UnifiedLiteLLMClient(openai_spec)

            assert "OPENAI_API_KEY" in env_dict
            assert env_dict["OPENAI_API_KEY"] == "test-key"  # pragma: allowlist secret

    def test_router_not_initialized_by_default(self, openai_spec):
        """Test that Router is not initialized without router_config."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(openai_spec)

            assert client.router is None
            assert client.use_router is False

    def test_cache_not_initialized_by_default(self, openai_spec):
        """Test that cache is not initialized without cache_config."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(openai_spec)

            assert client.cache is None


class TestUnifiedLiteLLMClientInvoke:
    """Test async invoke method."""

    @pytest.mark.asyncio
    async def test_ainvoke_success(self, openai_spec):
        """Test successful async invoke."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(openai_spec)

        # Mock litellm.acompletion
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20

        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch.object(
                client, "_calculate_cost_litellm", return_value=Decimal("0.0001")
            ),
        ):
            response = await client.ainvoke("Test prompt")

        assert isinstance(response, LLMResponse)
        assert response.text == "Test response"
        assert response.tokens_in == 10
        assert response.tokens_out == 20
        assert response.cost == Decimal("0.0001")
        assert response.model == "gpt-4o-mini"
        assert (
            response.latency_ms >= 0
        )  # Latency tracked (can be 0 in fast mocked calls)

    @pytest.mark.asyncio
    async def test_ainvoke_with_system_message(self, openai_spec):
        """Test async invoke with system message."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(openai_spec)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 25

        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_complete,
            patch.object(
                client, "_calculate_cost_litellm", return_value=Decimal("0.0002")
            ),
        ):
            response = await client.ainvoke(
                "User prompt", system_message="System instruction"
            )

        # Verify messages were built correctly
        call_args = mock_complete.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System instruction"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "User prompt"

        assert response.text == "Response"

    def test_invoke_sync_wraps_async(self, openai_spec):
        """Test that sync invoke wraps ainvoke."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(openai_spec)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Sync response"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10

        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch.object(
                client, "_calculate_cost_litellm", return_value=Decimal("0.0001")
            ),
        ):
            response = client.invoke("Test prompt")

        assert response.text == "Sync response"
        assert response.tokens_in == 5
        assert response.tokens_out == 10


class TestModelIdentifierBuilding:
    """Test model identifier construction."""

    def test_openai_identifier(self):
        """Test OpenAI model identifier."""
        spec = LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

        assert client.model_identifier == "openai/gpt-4o-mini"

    def test_groq_identifier(self):
        """Test Groq model identifier."""
        spec = LLMSpec(provider=LLMProvider.GROQ, model="llama-3.3-70b-versatile")
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

        assert client.model_identifier == "groq/llama-3.3-70b-versatile"

    def test_azure_identifier(self, azure_spec):
        """Test Azure model identifier uses deployment name."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(azure_spec)

        assert client.model_identifier == "azure/gpt-4-deployment"

    def test_model_with_existing_prefix(self):
        """Test model that already has provider prefix."""
        spec = LLMSpec(provider=LLMProvider.OPENAI, model="openai/gpt-4o-mini")
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

        assert client.model_identifier == "openai/gpt-4o-mini"


class TestCostCalculation:
    """Test cost calculation methods."""

    def test_calculate_cost_litellm_success(self, openai_spec):
        """Test cost calculation via LiteLLM."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(openai_spec)

        with patch("litellm.completion_cost", return_value=0.00123):
            cost = client._calculate_cost_litellm("prompt", "completion")

        assert cost == Decimal("0.00123")

    def test_calculate_cost_litellm_fallback(self, openai_spec):
        """Test fallback when LiteLLM cost fails."""
        openai_spec.input_cost_per_1k_tokens = Decimal("0.00015")
        openai_spec.output_cost_per_1k_tokens = Decimal("0.0006")

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(openai_spec)

        with (
            patch("litellm.completion_cost", side_effect=Exception("API error")),
            patch.object(client, "estimate_tokens", side_effect=[1000, 500]),
        ):
            cost = client._calculate_cost_litellm("prompt", "completion")

        # Should fallback to spec costs
        # 1000 input tokens * 0.00015/1k + 500 output tokens * 0.0006/1k
        expected = Decimal("0.00015") + Decimal("0.0003")
        assert cost == expected

    def test_calculate_cost_manual(self, openai_spec):
        """Test manual cost calculation."""
        openai_spec.input_cost_per_1k_tokens = Decimal("0.00015")
        openai_spec.output_cost_per_1k_tokens = Decimal("0.0006")

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(openai_spec)

        cost = client.calculate_cost(1000, 500)

        expected = Decimal("0.00015") + Decimal("0.0003")
        assert cost == expected


class TestTokenEstimation:
    """Test token estimation."""

    def test_estimate_tokens_via_litellm(self, openai_spec):
        """Test token estimation using LiteLLM encode."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(openai_spec)

        # Mock litellm.encode to return token list
        with patch("litellm.encode", return_value=[1, 2, 3, 4, 5]):
            tokens = client.estimate_tokens("test text")

        assert tokens == 5

    def test_estimate_tokens_fallback(self, openai_spec):
        """Test fallback token estimation when LiteLLM fails."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(openai_spec)

        # Mock litellm.encode to fail
        with patch("litellm.encode", side_effect=Exception("Encode error")):
            tokens = client.estimate_tokens("hello world test")

        # Should fallback to word count * 1.3
        assert tokens == int(3 * 1.3)  # 3 words → 3 tokens


class TestCacheSupport:
    """Test cache key generation and parsing."""

    def test_generate_cache_key(self, openai_spec):
        """Test cache key generation."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(openai_spec)

        key1 = client._generate_cache_key("prompt", {})
        key2 = client._generate_cache_key("prompt", {})
        key3 = client._generate_cache_key("different", {})

        # Same prompt = same key
        assert key1 == key2
        # Different prompt = different key
        assert key1 != key3
        # Key should be MD5 hash
        assert len(key1) == 32  # MD5 hex digest length

    def test_cache_key_includes_system_message(self, openai_spec):
        """Test that cache key differs with system message."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(openai_spec)

        key1 = client._generate_cache_key("prompt", {})
        key2 = client._generate_cache_key("prompt", {"system_message": "system"})

        assert key1 != key2

    def test_parse_cached_response(self, openai_spec):
        """Test parsing cached response dict into LLMResponse."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(openai_spec)

        cached_dict = {
            "text": "cached response",
            "tokens_in": 10,
            "tokens_out": 20,
            "model": "gpt-4o-mini",
            "cost": "0.0001",
            "latency_ms": 150.0,
        }

        response = client._parse_cached_response(cached_dict)

        assert isinstance(response, LLMResponse)
        assert response.text == "cached response"
        assert response.tokens_in == 10


class TestMessageBuilding:
    """Test message array construction."""

    def test_build_messages_user_only(self, openai_spec):
        """Test building messages with only user prompt."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(openai_spec)

        messages = client._build_messages("Hello", {})

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_build_messages_with_system(self, openai_spec):
        """Test building messages with system message."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(openai_spec)

        messages = client._build_messages(
            "Hello", {"system_message": "You are helpful"}
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"


class TestAPIKeyEnvVars:
    """Test API key environment variable mapping."""

    def test_openai_env_var(self, openai_spec):
        """Test OpenAI API key environment variable."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(openai_spec)

        env_var = client._get_api_key_env_var(LLMProvider.OPENAI)
        assert env_var == "OPENAI_API_KEY"

    def test_groq_env_var(self, groq_spec):
        """Test Groq API key environment variable."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(groq_spec)

        env_var = client._get_api_key_env_var(LLMProvider.GROQ)
        assert env_var == "GROQ_API_KEY"

    def test_azure_env_var(self, azure_spec):
        """Test Azure API key environment variable."""
        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(azure_spec)

        env_var = client._get_api_key_env_var(LLMProvider.AZURE_OPENAI)
        assert env_var == "AZURE_API_KEY"


class TestStructuredInvoke:
    """Test structured output with Instructor (Phase 2)."""

    def test_structured_invoke_implemented(self, openai_spec):
        """Test that structured_invoke is now implemented."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            field: str

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(openai_spec)

        # Should NOT raise NotImplementedError anymore
        # Mock asyncio.run since structured_invoke wraps async
        with patch("asyncio.run") as mock_run:
            mock_run.return_value = MagicMock()
            client.structured_invoke("prompt", TestModel)
            mock_run.assert_called_once()
