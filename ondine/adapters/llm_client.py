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
from llama_index.llms.openai_like import OpenAILike

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
        """Invoke OpenAI API with optional prompt caching."""
        start_time = time.time()

        # Build messages array (OpenAI auto-caches system messages)
        messages = []

        # Extract system message from kwargs
        system_message = kwargs.get("system_message")
        if system_message and self.spec.enable_prefix_caching:
            messages.append(ChatMessage(role="system", content=system_message))

        # User message (dynamic, not cached)
        messages.append(ChatMessage(role="user", content=prompt))

        response = self.client.chat(messages)

        latency_ms = (time.time() - start_time) * 1000

        # Extract token usage from API response (OpenAI returns actual counts)
        tokens_in = 0
        tokens_out = 0
        cached_tokens = 0

        if hasattr(response, "raw") and response.raw and hasattr(response.raw, "usage"):
            usage = response.raw.usage
            tokens_in = getattr(usage, "prompt_tokens", 0)
            tokens_out = getattr(usage, "completion_tokens", 0)

            # Extract cached tokens (OpenAI format: nested in prompt_tokens_details)
            if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0)

            # Debug: Log first response to see what OpenAI returns
            if not hasattr(self, "_debug_logged"):
                self._debug_logged = True
                from ondine.utils import get_logger

                logger = get_logger(f"{__name__}.OpenAIClient")
                logger.debug(
                    f"First API response: {tokens_in} input + {tokens_out} output tokens "
                    f"({cached_tokens} cached)"
                )

            # Log if caching is detected
            if cached_tokens > 0:
                from ondine.utils import get_logger

                logger = get_logger(f"{__name__}.OpenAIClient")
                cache_pct = (cached_tokens / tokens_in * 100) if tokens_in > 0 else 0
                logger.info(
                    f"✅ Cache hit! {cached_tokens}/{tokens_in} tokens cached ({cache_pct:.0f}%)"
                )

        # Fallback to tiktoken estimation if API doesn't provide counts
        if tokens_in == 0:
            total_prompt = prompt
            if system_message and self.spec.enable_prefix_caching:
                total_prompt = system_message + "\n" + prompt
            tokens_in = len(self.tokenizer.encode(total_prompt))
        if tokens_out == 0:
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
        """Invoke Anthropic API with prompt caching."""
        start_time = time.time()

        # Build messages array with optional system message
        messages = []

        # Extract system message from kwargs
        system_message = kwargs.get("system_message")

        # Anthropic uses a separate system parameter with cache_control
        # Until explicit cache_control is wired up, send system message to avoid dropping it
        if system_message:
            if self.spec.enable_prefix_caching:
                # TODO: Wire up explicit cache_control system param when LlamaIndex supports it
                # For now, send as system message (Anthropic caches automatically)
                messages.append(ChatMessage(role="system", content=system_message))
            else:
                # Fallback: prepend to user message if caching disabled
                prompt = f"{system_message}\n\n{prompt}"

        # User message (dynamic, not cached)
        messages.append(ChatMessage(role="user", content=prompt))

        response = self.client.chat(messages)

        latency_ms = (time.time() - start_time) * 1000

        # Approximate token usage (include system message if present)
        total_prompt = prompt
        if system_message:
            total_prompt = system_message + "\n" + prompt
        tokens_in = len(self.tokenizer.encode(total_prompt))
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

        # Initialize logger for debug logging
        from ondine.utils import get_logger

        self.logger = get_logger(f"{__name__}.GroqClient")

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Invoke Groq API with optional system message support."""
        start_time = time.time()

        # Build messages array (support system message for caching)
        messages = []

        system_message = kwargs.get("system_message")
        if system_message and self.spec.enable_prefix_caching:
            messages.append(ChatMessage(role="system", content=system_message))

        messages.append(ChatMessage(role="user", content=prompt))

        # Call Groq API
        response = self.client.chat(messages)

        latency_ms = (time.time() - start_time) * 1000

        # Extract text from response
        if hasattr(response, "message") and hasattr(response.message, "content"):
            response_text = response.message.content or ""
        elif hasattr(response, "content"):
            response_text = response.content or ""
        else:
            response_text = str(response) if response else ""

        # Extract token usage from LlamaIndex response.raw (ChatCompletion object)
        tokens_in = 0
        tokens_out = 0
        cached_tokens = 0

        # LlamaIndex provides actual token counts in response.raw.usage
        if hasattr(response, "raw") and response.raw and hasattr(response.raw, "usage"):
            usage = response.raw.usage
            tokens_in = getattr(usage, "prompt_tokens", 0)
            tokens_out = getattr(usage, "completion_tokens", 0)

            # Extract cached tokens (OpenAI/Groq format: nested in prompt_tokens_details)
            cached_tokens = 0
            if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0)

            # Debug: Log first response to see what Groq returns
            if not hasattr(self, "_debug_logged"):
                self._debug_logged = True
                self.logger.debug(
                    f"First API response: {tokens_in} input + {tokens_out} output tokens "
                    f"({cached_tokens} cached)"
                )

            # Log if caching is detected
            # Track cache hits (use DEBUG level to avoid spam in production)
            if cached_tokens > 0:
                cache_pct = (cached_tokens / tokens_in * 100) if tokens_in > 0 else 0
                self.logger.info(
                    f"✅ Cache hit! {cached_tokens}/{tokens_in} tokens cached ({cache_pct:.0f}%)"
                )

        # Fallback to tiktoken estimation if API doesn't provide counts
        if tokens_in == 0:
            full_prompt = (system_message or "") + prompt
            tokens_in = len(self.tokenizer.encode(full_prompt))
        if tokens_out == 0:
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

        # Use OpenAILike for custom providers (doesn't validate model names)
        self.client = OpenAILike(
            model=spec.model,
            api_key=api_key,
            api_base=spec.base_url,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
            is_chat_model=True,  # Assume chat model for OpenAI-compatible APIs
        )

        # Use provider_name for logging/metrics, or default
        self.provider_name = spec.provider_name or "OpenAI-Compatible"

        # Initialize tokenizer (use default encoding for custom providers)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Initialize logger for debug logging
        from ondine.utils import get_logger

        self.logger = get_logger(f"{__name__}.{self.provider_name}")

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Invoke OpenAI-compatible API with optional system message support.

        Args:
            prompt: Text prompt
            **kwargs: Additional model parameters (including system_message)

        Returns:
            LLMResponse with result and metadata
        """
        start_time = time.time()

        # Build messages array (support system message for caching)
        messages = []

        system_message = kwargs.get("system_message")
        if system_message and self.spec.enable_prefix_caching:
            messages.append(ChatMessage(role="system", content=system_message))

        messages.append(ChatMessage(role="user", content=prompt))

        # Call API
        response = self.client.chat(messages)

        latency_ms = (time.time() - start_time) * 1000

        # Extract text from response
        response_text = str(response) if response else ""

        # Extract token usage from API response (if available)
        tokens_in = 0
        tokens_out = 0
        cached_tokens = 0

        if hasattr(response, "raw") and response.raw and hasattr(response.raw, "usage"):
            usage = response.raw.usage
            tokens_in = getattr(usage, "prompt_tokens", 0)
            tokens_out = getattr(usage, "completion_tokens", 0)

            # Extract cached tokens (OpenAI/Moonshot format: nested in prompt_tokens_details)
            if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0)

            # Debug: Log first response
            if not hasattr(self, "_debug_logged"):
                self._debug_logged = True
                self.logger.debug(
                    f"First API response: {tokens_in} input + {tokens_out} output tokens "
                    f"({cached_tokens} cached)"
                )

            # Log if caching is detected
            # Track cache hits (use DEBUG level to avoid spam in production)
            if cached_tokens > 0:
                cache_pct = (cached_tokens / tokens_in * 100) if tokens_in > 0 else 0
                self.logger.info(
                    f"✅ Cache hit! {cached_tokens}/{tokens_in} tokens cached ({cache_pct:.0f}%)"
                )

        # Fallback to tiktoken estimation if API doesn't provide counts
        if tokens_in == 0:
            full_prompt = (system_message or "") + prompt
            tokens_in = len(self.tokenizer.encode(full_prompt))
        if tokens_out == 0:
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
