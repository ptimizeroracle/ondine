"""
LLM client abstractions and implementations.

Provides unified interface for multiple LLM providers following the
Adapter pattern and Dependency Inversion principle.
"""

import os
import time
import warnings
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any

# Suppress dependency warnings before importing llama_index
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*PyTorch.*TensorFlow.*Flax.*")

# Suppress transformers warnings about missing deep learning frameworks
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import tiktoken
from llama_index.core.llms import ChatMessage
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI

from ondine.core.models import LLMResponse
from ondine.core.specifications import LLMProvider, LLMSpec


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.

    Defines the contract that all LLM provider implementations must follow,
    enabling easy swapping of providers (Strategy pattern).
    """

    def __init__(self, spec: LLMSpec):
        """
        Initialize LLM client.

        Args:
            spec: LLM specification
        """
        self.spec = spec
        self.model = spec.model
        self.temperature = spec.temperature
        self.max_tokens = spec.max_tokens

    @abstractmethod
    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Invoke LLM with a single prompt.

        Args:
            prompt: Text prompt
            **kwargs: Additional model parameters

        Returns:
            LLMResponse with result and metadata
        """
        pass

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        pass

    def batch_invoke(self, prompts: list[str], **kwargs: Any) -> list[LLMResponse]:
        """
        Invoke LLM with multiple prompts.

        Default implementation: sequential invocation.
        Subclasses can override for provider-optimized batch processing.

        Args:
            prompts: List of text prompts
            **kwargs: Additional model parameters

        Returns:
            List of LLMResponse objects
        """
        return [self.invoke(prompt, **kwargs) for prompt in prompts]

    def calculate_cost(self, tokens_in: int, tokens_out: int) -> Decimal:
        """
        Calculate cost for token usage.

        Args:
            tokens_in: Input tokens
            tokens_out: Output tokens

        Returns:
            Total cost in USD
        """
        from ondine.utils.cost_calculator import CostCalculator

        return CostCalculator.calculate(
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            input_cost_per_1k=self.spec.input_cost_per_1k_tokens or Decimal("0.0"),
            output_cost_per_1k=self.spec.output_cost_per_1k_tokens or Decimal("0.0"),
        )


class OpenAIClient(LLMClient):
    """OpenAI LLM client implementation."""

    def __init__(self, spec: LLMSpec):
        """Initialize OpenAI client."""
        super().__init__(spec)

        api_key = spec.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in spec or environment")

        self.client = OpenAI(
            model=spec.model,
            api_key=api_key,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
        )

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(spec.model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Invoke OpenAI API."""
        start_time = time.time()

        message = ChatMessage(role="user", content=prompt)
        response = self.client.chat([message])

        latency_ms = (time.time() - start_time) * 1000

        # Extract token usage
        tokens_in = len(self.tokenizer.encode(prompt))
        tokens_out = len(self.tokenizer.encode(str(response)))

        cost = self.calculate_cost(tokens_in, tokens_out)

        return LLMResponse(
            text=str(response),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=self.model,
            cost=cost,
            latency_ms=latency_ms,
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using tiktoken."""
        return len(self.tokenizer.encode(text))


class AzureOpenAIClient(LLMClient):
    """Azure OpenAI LLM client implementation."""

    def __init__(self, spec: LLMSpec):
        """Initialize Azure OpenAI client with API key or Managed Identity."""
        super().__init__(spec)

        if not spec.azure_endpoint:
            raise ValueError("azure_endpoint required for Azure OpenAI")

        if not spec.azure_deployment:
            raise ValueError("azure_deployment required for Azure OpenAI")

        # Authentication: Three options in priority order
        # 1. Managed Identity (preferred for Azure deployments)
        # 2. Pre-fetched Azure AD token
        # 3. API key (backward compatible)

        if spec.use_managed_identity:
            # Use Azure Managed Identity
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError:
                raise ImportError(
                    "Azure Managed Identity requires azure-identity. "
                    "Install with: pip install ondine[azure]"
                )

            try:
                credential = DefaultAzureCredential()
                token = credential.get_token(
                    "https://cognitiveservices.azure.com/.default"
                )

                self.client = AzureOpenAI(
                    model=spec.model,
                    deployment_name=spec.azure_deployment,
                    azure_ad_token=token.token,
                    azure_endpoint=spec.azure_endpoint,
                    api_version=spec.api_version or "2024-02-15-preview",
                    temperature=spec.temperature,
                    max_tokens=spec.max_tokens,
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to authenticate with Azure Managed Identity: {e}. "
                    "Ensure the resource has a Managed Identity assigned with "
                    "'Cognitive Services OpenAI User' role."
                ) from e

        elif spec.azure_ad_token:
            # Use pre-fetched token
            self.client = AzureOpenAI(
                model=spec.model,
                deployment_name=spec.azure_deployment,
                azure_ad_token=spec.azure_ad_token,
                azure_endpoint=spec.azure_endpoint,
                api_version=spec.api_version or "2024-02-15-preview",
                temperature=spec.temperature,
                max_tokens=spec.max_tokens,
            )

        else:
            # Use API key (existing behavior - backward compatible)
            api_key = spec.api_key or os.getenv("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Azure OpenAI requires either:\n"
                    "  1. use_managed_identity=True (for keyless auth), or\n"
                    "  2. api_key parameter, or\n"
                    "  3. AZURE_OPENAI_API_KEY environment variable"
                )

            self.client = AzureOpenAI(
                model=spec.model,
                deployment_name=spec.azure_deployment,
                api_key=api_key,
                azure_endpoint=spec.azure_endpoint,
                api_version=spec.api_version or "2024-02-15-preview",
                temperature=spec.temperature,
                max_tokens=spec.max_tokens,
            )

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(spec.model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Invoke Azure OpenAI API."""
        start_time = time.time()

        message = ChatMessage(role="user", content=prompt)
        response = self.client.chat([message])

        latency_ms = (time.time() - start_time) * 1000

        # Extract token usage
        tokens_in = len(self.tokenizer.encode(prompt))
        tokens_out = len(self.tokenizer.encode(str(response)))

        cost = self.calculate_cost(tokens_in, tokens_out)

        return LLMResponse(
            text=str(response),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=self.model,
            cost=cost,
            latency_ms=latency_ms,
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using tiktoken."""
        return len(self.tokenizer.encode(text))


class AnthropicClient(LLMClient):
    """Anthropic Claude LLM client implementation."""

    def __init__(self, spec: LLMSpec):
        """Initialize Anthropic client."""
        super().__init__(spec)

        api_key = spec.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in spec or environment")

        self.client = Anthropic(
            model=spec.model,
            api_key=api_key,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens or 1024,
        )

        # Anthropic uses approximate token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Invoke Anthropic API."""
        start_time = time.time()

        message = ChatMessage(role="user", content=prompt)
        response = self.client.chat([message])

        latency_ms = (time.time() - start_time) * 1000

        # Approximate token usage
        tokens_in = len(self.tokenizer.encode(prompt))
        tokens_out = len(self.tokenizer.encode(str(response)))

        cost = self.calculate_cost(tokens_in, tokens_out)

        return LLMResponse(
            text=str(response),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=self.model,
            cost=cost,
            latency_ms=latency_ms,
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens (approximate for Anthropic)."""
        return len(self.tokenizer.encode(text))


class GroqClient(LLMClient):
    """Groq LLM client implementation."""

    def __init__(self, spec: LLMSpec):
        """Initialize Groq client."""
        super().__init__(spec)

        api_key = spec.api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in spec or environment")

        self.client = Groq(
            model=spec.model,
            api_key=api_key,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
        )

        # Use tiktoken for token estimation
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Invoke Groq API."""
        start_time = time.time()

        message = ChatMessage(role="user", content=prompt)
        response = self.client.chat([message])

        latency_ms = (time.time() - start_time) * 1000

        # Extract text from response - handle both string and object responses
        if hasattr(response, "message") and hasattr(response.message, "content"):
            response_text = response.message.content or ""
        elif hasattr(response, "content"):
            response_text = response.content or ""
        else:
            response_text = str(response) if response else ""

        # Extract token usage
        tokens_in = len(self.tokenizer.encode(prompt))
        tokens_out = len(self.tokenizer.encode(response_text))

        cost = self.calculate_cost(tokens_in, tokens_out)

        return LLMResponse(
            text=response_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=self.model,
            cost=cost,
            latency_ms=latency_ms,
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using tiktoken."""
        return len(self.tokenizer.encode(text))


class OpenAICompatibleClient(LLMClient):
    """
    Client for OpenAI-compatible API endpoints.

    Supports custom providers like Ollama, vLLM, Together.ai, Anyscale,
    and any other API that implements the OpenAI chat completions format.
    """

    def __init__(self, spec: LLMSpec):
        """
        Initialize OpenAI-compatible client.

        Args:
            spec: LLM specification with base_url required

        Raises:
            ValueError: If base_url not provided
        """
        super().__init__(spec)

        if not spec.base_url:
            raise ValueError("base_url required for openai_compatible provider")

        # Get API key (optional for local APIs like Ollama)
        api_key = spec.api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY") or "dummy"

        # Initialize OpenAI client with custom base URL
        self.client = OpenAI(
            model=spec.model,
            api_key=api_key,
            api_base=spec.base_url,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
        )

        # Use provider_name for logging/metrics, or default
        self.provider_name = spec.provider_name or "OpenAI-Compatible"

        # Initialize tokenizer (use default encoding for custom providers)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Invoke OpenAI-compatible API.

        Args:
            prompt: Text prompt
            **kwargs: Additional model parameters

        Returns:
            LLMResponse with result and metadata
        """
        start_time = time.time()

        message = ChatMessage(role="user", content=prompt)
        response = self.client.chat([message])

        latency_ms = (time.time() - start_time) * 1000

        # Extract text from response
        response_text = str(response) if response else ""

        # Estimate token usage (approximate for custom providers)
        tokens_in = len(self.tokenizer.encode(prompt))
        tokens_out = len(self.tokenizer.encode(response_text))

        cost = self.calculate_cost(tokens_in, tokens_out)

        return LLMResponse(
            text=response_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=f"{self.provider_name}/{self.model}",  # Show provider in metrics
            cost=cost,
            latency_ms=latency_ms,
        )

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate tokens using tiktoken.

        Note: This is approximate for custom providers.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        return len(self.tokenizer.encode(text))


class MLXClient(LLMClient):
    """
    MLX client for Apple Silicon local inference.

    MLX is Apple's optimized ML framework for M-series chips.
    This client enables fast, local LLM inference without API costs.

    Requires: pip install ondine[mlx]
    Platform: macOS with Apple Silicon only
    """

    def __init__(self, spec: LLMSpec, _mlx_lm_module=None):
        """
        Initialize MLX client and load model.

        Model is loaded once and cached for fast subsequent calls.

        Args:
            spec: LLM specification with model name
            _mlx_lm_module: MLX module (internal/testing only)

        Raises:
            ImportError: If MLX not installed
            Exception: If model loading fails
        """
        super().__init__(spec)

        # Load mlx_lm module (or use injected module for testing)
        if _mlx_lm_module is None:
            try:
                import mlx_lm as _mlx_lm_module
            except ImportError as e:
                raise ImportError(
                    "MLX not installed. Install with:\n"
                    "  pip install ondine[mlx]\n"
                    "or:\n"
                    "  pip install mlx mlx-lm\n\n"
                    "Note: MLX only works on Apple Silicon (M1/M2/M3/M4 chips)"
                ) from e

        # Store mlx_lm module for later use
        self.mlx_lm = _mlx_lm_module

        # Load model once (expensive operation, ~1-2 seconds)
        print(f"🔄 Loading MLX model: {spec.model}...")
        try:
            self.mlx_model, self.mlx_tokenizer = self.mlx_lm.load(spec.model)
            print("✅ Model loaded successfully")
        except Exception as e:
            raise Exception(
                f"Failed to load MLX model '{spec.model}'. "
                f"Ensure the model exists on HuggingFace and you have access. "
                f"Error: {e}"
            ) from e

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Invoke MLX model for inference.

        Args:
            prompt: Text prompt
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse with result and metadata
        """
        start_time = time.time()

        # Generate response using cached model
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        response_text = self.mlx_lm.generate(
            self.mlx_model,
            self.mlx_tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Estimate token usage using MLX tokenizer
        try:
            tokens_in = len(self.mlx_tokenizer.encode(prompt))
            tokens_out = len(self.mlx_tokenizer.encode(response_text))
        except Exception:
            # Fallback to simple estimation if encoding fails
            tokens_in = len(prompt.split())
            tokens_out = len(response_text.split())

        # Calculate cost (typically $0 for local models)
        cost = self.calculate_cost(tokens_in, tokens_out)

        return LLMResponse(
            text=response_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=f"MLX/{self.model}",
            cost=cost,
            latency_ms=latency_ms,
        )

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using MLX tokenizer.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        try:
            return len(self.mlx_tokenizer.encode(text))
        except Exception:
            # Fallback to simple word count
            return len(text.split())


def create_llm_client(spec: LLMSpec) -> LLMClient:
    """
    Factory function to create appropriate LLM client using ProviderRegistry.

    Supports both built-in providers (via LLMProvider enum) and custom
    providers (registered via ProviderRegistry).

    Args:
        spec: LLM specification

    Returns:
        Configured LLM client

    Raises:
        ValueError: If provider not supported

    Example:
        # Built-in provider
        spec = LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
        client = create_llm_client(spec)

        # Custom provider (registered via @provider decorator)
        spec = LLMSpec(provider="my_custom_llm", model="my-model")
        client = create_llm_client(spec)
    """
    from ondine.adapters.provider_registry import ProviderRegistry

    # Check if custom provider ID is specified (from PipelineBuilder.with_llm)
    custom_provider_id = getattr(spec, "_custom_provider_id", None)
    if custom_provider_id:
        provider_id = custom_provider_id
    else:
        # Convert enum to string for registry lookup
        provider_id = (
            spec.provider.value
            if isinstance(spec.provider, LLMProvider)
            else spec.provider
        )

    # Get provider class from registry
    provider_class = ProviderRegistry.get(provider_id)

    # Instantiate and return
    return provider_class(spec)
