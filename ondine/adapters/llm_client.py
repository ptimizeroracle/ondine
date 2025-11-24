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

from pydantic import BaseModel

# Suppress dependency warnings before importing llama_index
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*PyTorch.*TensorFlow.*Flax.*")

# Suppress transformers warnings about missing deep learning frameworks
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


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
    def structured_invoke(
        self,
        prompt: str,
        output_cls: type[BaseModel],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Invoke LLM with structured output enforcement.

        Each implementation provides its own structured output strategy:
        - UnifiedLiteLLMClient: Uses Instructor (Phase 2)
        - MLXClient: Not supported (raises NotImplementedError)

        Args:
            prompt: Text prompt
            output_cls: Pydantic model class for output validation
            **kwargs: Additional model parameters (e.g., system_message)

        Returns:
            LLMResponse with validated structured result (serialized JSON)

        Raises:
            ValueError: If structured prediction fails
            NotImplementedError: If provider doesn't support structured output
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


# =================================================================
# Built-in Provider Implementations
# =================================================================
# AGGRESSIVE REFACTOR (Phase 1):
# - All cloud providers (OpenAI, Groq, Anthropic, Azure, custom) → UnifiedLiteLLMClient
# - Apple Silicon local inference → MLXClient (special case)
# - LlamaIndex kept ONLY for future RAG features (separate concern)
#
# DELETED in Phase 1 (replaced by UnifiedLiteLLMClient):
# - AzureOpenAIClient: ~118 lines (Azure now via LiteLLM: model="azure/deployment")
# - OpenAICompatibleClient: ~140 lines (Custom endpoints via LiteLLM base_url)
#
# This eliminates 258 lines of LlamaIndex wrapper code!
# =================================================================


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

    def structured_invoke(
        self,
        prompt: str,
        output_cls: type[BaseModel],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Structured output not supported for MLX (local models).

        Args:
            prompt: Text prompt
            output_cls: Pydantic model class
            **kwargs: Additional parameters

        Raises:
            NotImplementedError: MLX doesn't support structured output
        """
        raise NotImplementedError(
            "Structured output not supported for MLX local models. "
            "Use cloud providers (OpenAI, Groq, Anthropic) for structured output."
        )


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
    custom_provider_id = spec.custom_provider_id
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
