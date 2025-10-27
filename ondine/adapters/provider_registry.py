"""
Provider registry for extensible LLM client plugins.

Enables custom LLM providers to be registered and discovered without
modifying core code.
"""


class LLMClient:
    """Protocol for LLM client implementations (imported to avoid circular dependency)."""

    pass


class ProviderRegistry:
    """
    Global registry for LLM provider plugins.

    Enables registration and discovery of custom LLM providers without
    modifying core code. Uses lazy initialization for built-in providers.

    Example:
        # Register custom provider
        @ProviderRegistry.register("my_llm")
        class MyLLMClient(LLMClient):
            def invoke(self, prompt: str, **kwargs) -> LLMResponse:
                ...

        # Use in pipeline
        pipeline.with_llm(provider="my_llm", model="my-model")
    """

    _providers: dict[str, type] = {}
    _builtin_registered: bool = False

    @classmethod
    def register(cls, provider_id: str, client_class: type) -> type:
        """
        Register an LLM provider.

        Args:
            provider_id: Unique provider identifier (e.g., "openai", "my_custom_llm")
            client_class: LLM client class implementing LLMClient interface

        Returns:
            The registered client class (enables use as decorator)

        Raises:
            ValueError: If provider_id already registered

        Example:
            @ProviderRegistry.register("replicate")
            class ReplicateClient(LLMClient):
                ...
        """
        if provider_id in cls._providers:
            raise ValueError(
                f"Provider '{provider_id}' already registered. "
                f"Use a different provider_id or unregister first."
            )

        cls._providers[provider_id] = client_class
        return client_class

    @classmethod
    def get(cls, provider_id: str) -> type:
        """
        Get provider class by ID.

        Args:
            provider_id: Provider identifier

        Returns:
            LLM client class

        Raises:
            ValueError: If provider not found

        Example:
            client_class = ProviderRegistry.get("openai")
            client = client_class(spec)
        """
        cls._ensure_builtins_registered()

        if provider_id not in cls._providers:
            available = ", ".join(sorted(cls._providers.keys()))
            raise ValueError(
                f"Unknown provider: '{provider_id}'. Available providers: {available}"
            )

        return cls._providers[provider_id]

    @classmethod
    def list_providers(cls) -> dict[str, type]:
        """
        List all registered providers.

        Returns:
            Dictionary mapping provider IDs to client classes

        Example:
            providers = ProviderRegistry.list_providers()
            print(f"Available: {list(providers.keys())}")
        """
        cls._ensure_builtins_registered()
        return cls._providers.copy()

    @classmethod
    def is_registered(cls, provider_id: str) -> bool:
        """
        Check if provider is registered.

        Args:
            provider_id: Provider identifier

        Returns:
            True if registered, False otherwise
        """
        cls._ensure_builtins_registered()
        return provider_id in cls._providers

    @classmethod
    def unregister(cls, provider_id: str) -> None:
        """
        Unregister a provider (mainly for testing).

        Args:
            provider_id: Provider identifier

        Raises:
            ValueError: If provider not found
        """
        if provider_id not in cls._providers:
            raise ValueError(f"Provider '{provider_id}' not registered")

        del cls._providers[provider_id]

    @classmethod
    def _ensure_builtins_registered(cls) -> None:
        """
        Lazy-register built-in providers on first use.

        This avoids import-time side effects and allows built-ins
        to be registered only when needed.
        """
        if cls._builtin_registered:
            return

        # Import here to avoid circular dependencies
        from ondine.adapters.llm_client import (
            AnthropicClient,
            AzureOpenAIClient,
            GroqClient,
            MLXClient,
            OpenAIClient,
            OpenAICompatibleClient,
        )

        # Register built-in providers
        cls._providers["openai"] = OpenAIClient
        cls._providers["azure_openai"] = AzureOpenAIClient
        cls._providers["anthropic"] = AnthropicClient
        cls._providers["groq"] = GroqClient
        cls._providers["openai_compatible"] = OpenAICompatibleClient
        cls._providers["mlx"] = MLXClient

        cls._builtin_registered = True

    @classmethod
    def _reset(cls) -> None:
        """
        Reset registry (for testing only).

        Clears all registered providers and resets initialization flag.
        """
        cls._providers.clear()
        cls._builtin_registered = False


def provider(provider_id: str):
    """
    Decorator to register a custom LLM provider.

    Args:
        provider_id: Unique provider identifier

    Returns:
        Decorator function

    Example:
        @provider("replicate")
        class ReplicateClient(LLMClient):
            def __init__(self, spec: LLMSpec):
                super().__init__(spec)
                import replicate
                self.client = replicate.Client(api_token=spec.api_key)

            def invoke(self, prompt: str, **kwargs) -> LLMResponse:
                output = self.client.run(self.model, input={"prompt": prompt})
                return LLMResponse(
                    text=output,
                    tokens_in=self.estimate_tokens(prompt),
                    tokens_out=self.estimate_tokens(output),
                    model=self.model,
                    cost=self.calculate_cost(...),
                    latency_ms=...
                )

            def estimate_tokens(self, text: str) -> int:
                return len(text.split())
    """

    def decorator(cls):
        ProviderRegistry.register(provider_id, cls)
        return cls

    return decorator
