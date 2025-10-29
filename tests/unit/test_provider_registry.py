"""
Unit tests for ProviderRegistry.

Tests the plugin system for custom LLM providers.
"""

import pytest

from ondine.adapters.llm_client import LLMClient
from ondine.adapters.provider_registry import ProviderRegistry, provider
from ondine.core.models import LLMResponse
from ondine.core.specifications import LLMSpec


class TestProviderRegistry:
    """Test suite for ProviderRegistry."""

    def setup_method(self):
        """Reset registry before each test."""
        ProviderRegistry._reset()

    def test_register_custom_provider(self):
        """Should register custom provider successfully."""

        class CustomClient(LLMClient):
            def invoke(self, prompt: str, **kwargs) -> LLMResponse:
                return LLMResponse(
                    text="test",
                    tokens_in=10,
                    tokens_out=5,
                    model="custom",
                    cost=0.001,
                    latency_ms=100,
                )

            def estimate_tokens(self, text: str) -> int:
                return len(text.split())

        # Register
        ProviderRegistry.register("custom_provider", CustomClient)

        # Verify registered
        assert ProviderRegistry.is_registered("custom_provider")
        assert "custom_provider" in ProviderRegistry.list_providers()

    def test_register_duplicate_raises_error(self):
        """Should raise error when registering duplicate provider ID."""

        class CustomClient(LLMClient):
            pass

        ProviderRegistry.register("duplicate", CustomClient)

        with pytest.raises(ValueError, match="already registered"):
            ProviderRegistry.register("duplicate", CustomClient)

    def test_get_registered_provider(self):
        """Should retrieve registered provider class."""

        class CustomClient(LLMClient):
            pass

        ProviderRegistry.register("test_provider", CustomClient)

        retrieved_class = ProviderRegistry.get("test_provider")
        assert retrieved_class is CustomClient

    def test_get_unregistered_provider_raises_error(self):
        """Should raise error for unregistered provider."""
        with pytest.raises(ValueError, match="Unknown provider.*nonexistent"):
            ProviderRegistry.get("nonexistent")

    def test_list_providers_returns_all_registered(self):
        """Should list all registered providers."""

        class Custom1(LLMClient):
            pass

        class Custom2(LLMClient):
            pass

        ProviderRegistry.register("provider1", Custom1)
        ProviderRegistry.register("provider2", Custom2)

        providers = ProviderRegistry.list_providers()

        assert "provider1" in providers
        assert "provider2" in providers
        assert providers["provider1"] is Custom1
        assert providers["provider2"] is Custom2

    def test_list_providers_includes_builtins(self):
        """Should include built-in providers after lazy initialization."""
        providers = ProviderRegistry.list_providers()

        # Built-ins should be present
        assert "openai" in providers
        assert "anthropic" in providers
        assert "groq" in providers
        assert "azure_openai" in providers
        assert "openai_compatible" in providers
        assert "mlx" in providers

    def test_builtin_providers_lazy_initialization(self):
        """Should lazy-load built-in providers only when accessed."""
        # Reset to ensure clean state
        ProviderRegistry._reset()
        assert not ProviderRegistry._builtin_registered

        # Access triggers lazy init
        ProviderRegistry.list_providers()
        assert ProviderRegistry._builtin_registered

    def test_unregister_provider(self):
        """Should unregister provider successfully."""

        class CustomClient(LLMClient):
            pass

        ProviderRegistry.register("to_remove", CustomClient)
        assert ProviderRegistry.is_registered("to_remove")

        ProviderRegistry.unregister("to_remove")
        assert not ProviderRegistry.is_registered("to_remove")

    def test_unregister_nonexistent_raises_error(self):
        """Should raise error when unregistering non-existent provider."""
        with pytest.raises(ValueError, match="not registered"):
            ProviderRegistry.unregister("nonexistent")

    def test_provider_decorator(self):
        """Should register provider via decorator."""

        @provider("decorated_provider")
        class DecoratedClient(LLMClient):
            def invoke(self, prompt: str, **kwargs) -> LLMResponse:
                return LLMResponse(
                    text="decorated",
                    tokens_in=1,
                    tokens_out=1,
                    model="test",
                    cost=0.0,
                    latency_ms=0,
                )

            def estimate_tokens(self, text: str) -> int:
                return 1

        assert ProviderRegistry.is_registered("decorated_provider")
        assert ProviderRegistry.get("decorated_provider") is DecoratedClient

    def test_provider_decorator_returns_class(self):
        """Should return decorated class unchanged (enables inheritance)."""

        @provider("test")
        class TestClient(LLMClient):
            custom_attr = "test"

        # Should have access to original class attributes
        assert TestClient.custom_attr == "test"

    def test_create_instance_from_registry(self):
        """Should create working instances from registered providers."""

        @provider("test_instance")
        class TestClient(LLMClient):
            def __init__(self, spec: LLMSpec):
                super().__init__(spec)
                self.initialized = True

            def invoke(self, prompt: str, **kwargs) -> LLMResponse:
                return LLMResponse(
                    text=f"Echo: {prompt}",
                    tokens_in=10,
                    tokens_out=10,
                    model=self.model,
                    cost=0.001,
                    latency_ms=100,
                )

            def estimate_tokens(self, text: str) -> int:
                return len(text.split())

        # Get class and instantiate
        client_class = ProviderRegistry.get("test_instance")
        # Use a valid LLMProvider enum value for validation, but the client won't care
        from ondine.core.specifications import LLMProvider

        spec = LLMSpec(provider=LLMProvider.OPENAI, model="test-model")
        client = client_class(spec)

        # Verify it works
        assert client.initialized
        assert client.model == "test-model"
        response = client.invoke("Hello")
        assert "Hello" in response.text

    def test_registry_isolation_between_tests(self):
        """Should maintain registry state across operations."""

        @provider("test1")
        class Test1(LLMClient):
            pass

        @provider("test2")
        class Test2(LLMClient):
            pass

        providers = ProviderRegistry.list_providers()
        assert "test1" in providers
        assert "test2" in providers

    def test_register_with_same_class_different_ids(self):
        """Should allow registering same class under different IDs."""

        class SharedClient(LLMClient):
            pass

        ProviderRegistry.register("provider_a", SharedClient)
        ProviderRegistry.register("provider_b", SharedClient)

        assert ProviderRegistry.get("provider_a") is SharedClient
        assert ProviderRegistry.get("provider_b") is SharedClient

    def test_reset_clears_all_providers(self):
        """Should clear all providers including built-ins on reset."""

        @provider("custom")
        class CustomClient(LLMClient):
            pass

        # Trigger builtin init
        _ = ProviderRegistry.list_providers()

        # Reset
        ProviderRegistry._reset()

        # Should be empty and not initialized
        assert len(ProviderRegistry._providers) == 0
        assert not ProviderRegistry._builtin_registered

    def test_builtin_providers_are_functional(self):
        """Should have all built-in providers registered and instantiable."""
        from ondine.adapters.llm_client import (
            AnthropicClient,
            AzureOpenAIClient,
            GroqClient,
            MLXClient,
            OpenAIClient,
            OpenAICompatibleClient,
        )

        # Trigger lazy init
        providers = ProviderRegistry.list_providers()

        # Verify all built-ins
        assert providers["openai"] is OpenAIClient
        assert providers["azure_openai"] is AzureOpenAIClient
        assert providers["anthropic"] is AnthropicClient
        assert providers["groq"] is GroqClient
        assert providers["openai_compatible"] is OpenAICompatibleClient
        assert providers["mlx"] is MLXClient
