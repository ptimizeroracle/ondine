"""
Unified LiteLLM client - Direct integration with LiteLLM (no LlamaIndex wrapper).

This client wraps ALL LiteLLM features directly:
- litellm.acompletion() for async-first API calls
- litellm.Router for load balancing and failover
- litellm.completion_cost() for automatic cost tracking
- litellm.encode() for token estimation
- Supports 100+ providers (OpenAI, Groq, Anthropic, Azure, etc.)

Clean abstraction enables future framework swaps while using LiteLLM natively.
"""

import asyncio
import hashlib
import logging
import os
import time
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from ondine.adapters.llm_client import LLMClient
from ondine.core.models import LLMResponse
from ondine.core.specifications import LLMSpec

logger = logging.getLogger(__name__)


class UnifiedLiteLLMClient(LLMClient):
    """
    Unified LLM client using LiteLLM directly (no wrappers).

    Features:
    - Native async: Uses litellm.acompletion() directly
    - All providers: OpenAI, Groq, Anthropic, Azure, 100+ more
    - Router support: Load balancing and failover (optional)
    - Auto-cost: Uses litellm.completion_cost()
    - Redis caching: Optional response caching (optional)
    - Clean abstraction: Can swap to another framework later

    Example:
        # OpenAI
        spec = LLMSpec(provider="openai", model="gpt-4o-mini")
        client = UnifiedLiteLLMClient(spec)

        # Groq
        spec = LLMSpec(provider="groq", model="llama-3.3-70b-versatile")
        client = UnifiedLiteLLMClient(spec)

        # Azure (LiteLLM handles it natively)
        spec = LLMSpec(provider="azure_openai", model="gpt-4",
                       azure_endpoint="...", azure_deployment="...")
        client = UnifiedLiteLLMClient(spec)
    """

    def __init__(self, spec: LLMSpec):
        """
        Initialize unified LiteLLM client.

        Args:
            spec: LLM specification with provider, model, and config
        """
        super().__init__(spec)

        # Suppress verbose LiteLLM logs
        os.environ["LITELLM_LOG"] = "WARNING"
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("litellm").setLevel(logging.WARNING)

        try:
            import litellm

            litellm.suppress_debug_info = True
            litellm.drop_params = True
        except (ImportError, AttributeError):
            pass

        # Build model identifier for LiteLLM
        self.model_identifier = self._build_model_identifier(spec)

        # Set API key if provided
        if spec.api_key:
            env_var = self._get_api_key_env_var(spec.provider)
            os.environ[env_var] = spec.api_key

        # Router support (optional - for load balancing)
        self.router = None
        self.use_router = False
        if hasattr(spec, "router_config") and spec.router_config:
            self._init_router(spec.router_config)

        # Cache support (optional - for response caching)
        self.cache = None
        if hasattr(spec, "cache_config") and spec.cache_config:
            self._init_cache(spec.cache_config)

        logger.debug(
            f"Initialized UnifiedLiteLLMClient with model: {self.model_identifier}"
        )

    def _build_model_identifier(self, spec: LLMSpec) -> str:
        """
        Build LiteLLM model identifier (format: provider/model).

        Examples:
            - openai + gpt-4o-mini → "openai/gpt-4o-mini"
            - groq + llama-3.3-70b → "groq/llama-3.3-70b-versatile"
            - azure_openai + gpt-4 → "azure/deployment-name"
        """
        # Get provider name
        provider_name = (
            spec.provider.value
            if hasattr(spec.provider, "value")
            else str(spec.provider)
        )

        # Remove litellm_ prefix if present
        provider_name = provider_name.replace("litellm_", "")

        # Special handling for Azure
        if provider_name == "azure_openai":
            # Azure uses deployment name, not model name
            if hasattr(spec, "azure_deployment") and spec.azure_deployment:
                return f"azure/{spec.azure_deployment}"
            return f"azure/{spec.model}"

        # Check if model already has provider prefix
        if spec.model.startswith(f"{provider_name}/"):
            return spec.model

        # Add provider prefix
        return f"{provider_name}/{spec.model}"

    def _get_api_key_env_var(self, provider) -> str:
        """Get environment variable name for provider API key."""
        provider_str = provider.value if hasattr(provider, "value") else str(provider)

        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
            "azure_openai": "AZURE_API_KEY",
        }

        return env_var_map.get(provider_str, f"{provider_str.upper()}_API_KEY")

    def _init_router(self, router_config):
        """Initialize LiteLLM Router for load balancing."""
        try:
            from litellm import Router

            self.router = Router(
                model_list=router_config.get("model_list", []),
                routing_strategy=router_config.get(
                    "routing_strategy", "latency-based-routing"
                ),
                num_retries=router_config.get("num_retries", 3),
                timeout=router_config.get("timeout", 30),
            )
            self.use_router = True
            logger.info(
                f"Initialized LiteLLM Router with {len(router_config.get('model_list', []))} models"
            )
        except ImportError:
            logger.warning("LiteLLM Router requested but litellm not installed")

    def _init_cache(self, cache_config):
        """Initialize Redis cache for response caching."""
        try:
            from ondine.adapters.cache_client import RedisResponseCache

            self.cache = RedisResponseCache(
                redis_url=cache_config.get("redis_url"),
                ttl=cache_config.get("ttl", 3600),
            )
            logger.info("Initialized Redis response cache")
        except ImportError:
            logger.warning("Redis cache requested but redis not installed")

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Async invoke LLM (native async, not wrapped).

        Args:
            prompt: Text prompt
            **kwargs: Additional parameters (system_message, etc.)

        Returns:
            LLMResponse with result and metadata
        """
        import litellm

        start_time = time.time()

        # Check cache first
        if self.cache:
            cache_key = self._generate_cache_key(prompt, kwargs)
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug("Cache hit for prompt")
                return self._parse_cached_response(cached)

        # Build messages
        messages = self._build_messages(prompt, kwargs)

        # Call LiteLLM (Router or direct)
        if self.use_router:
            response = await self.router.acompletion(
                model=self.model_identifier,
                messages=messages,
                temperature=self.spec.temperature,
                max_tokens=self.spec.max_tokens,
            )
        else:
            response = await litellm.acompletion(
                model=self.model_identifier,
                messages=messages,
                temperature=self.spec.temperature,
                max_tokens=self.spec.max_tokens,
            )

        latency_ms = (time.time() - start_time) * 1000

        # Extract response
        response_text = response.choices[0].message.content
        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0

        # Calculate cost using LiteLLM
        cost = self._calculate_cost_litellm(prompt, response_text)

        llm_response = LLMResponse(
            text=response_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=self.model,
            cost=cost,
            latency_ms=latency_ms,
        )

        # Cache response
        if self.cache:
            self.cache.set(cache_key, llm_response.model_dump())

        return llm_response

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Sync invoke (wraps async).

        Args:
            prompt: Text prompt
            **kwargs: Additional parameters

        Returns:
            LLMResponse with result and metadata
        """
        return asyncio.run(self.ainvoke(prompt, **kwargs))

    def _build_messages(self, prompt: str, kwargs: dict) -> list[dict]:
        """Build messages array for LiteLLM."""
        messages = []

        # Add system message if provided
        system_message = kwargs.get("system_message")
        if system_message:
            messages.append({"role": "system", "content": system_message})

        # Add user prompt
        messages.append({"role": "user", "content": prompt})

        return messages

    def _generate_cache_key(self, prompt: str, kwargs: dict) -> str:
        """Generate cache key from prompt and parameters."""
        # Include model, prompt, temperature in cache key
        key_parts = [
            self.model_identifier,
            prompt,
            str(self.spec.temperature),
            str(kwargs.get("system_message", "")),
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode(), usedforsecurity=False).hexdigest()

    def _parse_cached_response(self, cached: dict) -> LLMResponse:
        """Parse cached response into LLMResponse."""
        return LLMResponse(**cached)

    def _calculate_cost_litellm(self, prompt: str, completion: str) -> Decimal:
        """Calculate cost using LiteLLM's pricing database."""
        try:
            from litellm import completion_cost

            cost = completion_cost(
                model=self.model_identifier,
                prompt=prompt,
                completion=completion,
            )
            return Decimal(str(cost)) if cost else Decimal("0.0")
        except Exception as e:
            logger.debug(f"Could not calculate cost via LiteLLM: {e}")
            # Fallback to spec costs if available
            if (
                self.spec.input_cost_per_1k_tokens
                and self.spec.output_cost_per_1k_tokens
            ):
                tokens_in = self.estimate_tokens(prompt)
                tokens_out = self.estimate_tokens(completion)
                return self.calculate_cost(tokens_in, tokens_out)
            return Decimal("0.0")

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using LiteLLM.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        try:
            from litellm import encode

            return len(encode(model=self.model_identifier, text=text))
        except Exception:
            # Fallback to word-based estimation
            return int(len(text.split()) * 1.3)

    def calculate_cost(self, tokens_in: int, tokens_out: int) -> Decimal:
        """
        Calculate cost from token counts.

        Args:
            tokens_in: Input tokens
            tokens_out: Output tokens

        Returns:
            Cost in USD
        """
        if (
            not self.spec.input_cost_per_1k_tokens
            or not self.spec.output_cost_per_1k_tokens
        ):
            return Decimal("0.0")

        input_cost = Decimal(str(tokens_in / 1000)) * self.spec.input_cost_per_1k_tokens
        output_cost = (
            Decimal(str(tokens_out / 1000)) * self.spec.output_cost_per_1k_tokens
        )
        return input_cost + output_cost

    def structured_invoke(
        self,
        prompt: str,
        output_cls: type[BaseModel],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Structured output invocation using Instructor (sync wrapper).

        Wraps async structured_invoke_async for compatibility with sync pipeline.
        Uses asyncio.run() which may show cleanup warnings - this is expected.

        Args:
            prompt: Text prompt
            output_cls: Pydantic model class
            **kwargs: Additional parameters

        Returns:
            LLMResponse with structured output
        """
        # Use asyncio.run() to wrap async call
        # Note: This may show "event loop closed" warnings during cleanup
        # This is a known LiteLLM issue and doesn't affect functionality
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return asyncio.run(
                self.structured_invoke_async(prompt, output_cls, **kwargs)
            )

    async def structured_invoke_async(
        self,
        prompt: str,
        output_cls: type[BaseModel],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Async structured output invocation using Instructor.

        Supports dual-path strategy:
        1. Instructor with Mode.JSON (Groq and safe default)
        2. Instructor with Mode.TOOLS (OpenAI/Anthropic function calling)
        3. Native LiteLLM function calling (future)

        Auto-detects best mode based on provider.

        Args:
            prompt: Text prompt
            output_cls: Pydantic model class
            **kwargs: Additional parameters

        Returns:
            LLMResponse with structured output

        Raises:
            ValueError: If structured prediction fails
        """
        import instructor
        from litellm import acompletion

        start_time = time.time()

        # Determine mode (auto-detect if not specified)
        mode = self._get_structured_output_mode()

        # Initialize Instructor client with appropriate mode
        instructor_client = instructor.from_litellm(acompletion, mode=mode)

        # Build messages
        messages = self._build_messages(prompt, kwargs)

        # Call Instructor (async)
        try:
            result = await instructor_client.chat.completions.create(
                model=self.model_identifier,
                messages=messages,
                response_model=output_cls,
                temperature=self.spec.temperature,
                max_tokens=self.spec.max_tokens,
                max_retries=3,  # Built-in validation retry!
            )

            latency_ms = (time.time() - start_time) * 1000

            # Serialize result to JSON
            response_text = result.model_dump_json()

            # Estimate tokens (Instructor doesn't expose usage)
            full_prompt = prompt
            if kwargs.get("system_message"):
                full_prompt = f"{kwargs['system_message']}\n\n{prompt}"

            tokens_in = self.estimate_tokens(full_prompt)
            tokens_out = self.estimate_tokens(response_text)

            # Calculate cost
            cost = self._calculate_cost_litellm(full_prompt, response_text)

            return LLMResponse(
                text=response_text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model=self.model,
                cost=cost,
                latency_ms=latency_ms,
            )

        except Exception as e:
            raise ValueError(f"Structured prediction failed: {e}") from e

    def _get_structured_output_mode(self):
        """
        Determine which Instructor mode to use for structured output.

        Auto-detection logic:
        - Groq: Mode.JSON (no native function calling)
        - OpenAI/Anthropic: Mode.TOOLS (native function calling)
        - Others: Mode.JSON (safe default)

        Can be overridden via spec.structured_output_mode if implemented.

        Returns:
            instructor.Mode enum value
        """
        import instructor

        # Check if mode is explicitly configured
        if hasattr(self.spec, "structured_output_mode"):
            mode_str = self.spec.structured_output_mode
            if mode_str == "instructor_json":
                return instructor.Mode.JSON
            if mode_str == "instructor_tools":
                return instructor.Mode.TOOLS
            if mode_str == "native":
                logger.warning(
                    "Native function calling not yet implemented, using Instructor Mode.TOOLS"
                )
                return instructor.Mode.TOOLS

        # Auto-detect based on provider
        provider_str = (
            self.spec.provider.value
            if hasattr(self.spec.provider, "value")
            else str(self.spec.provider)
        )

        if provider_str in ["groq"]:
            # Groq doesn't have reliable function calling, use JSON mode
            return instructor.Mode.JSON
        if provider_str in ["openai", "azure_openai", "anthropic"]:
            # These providers have good function calling support
            return instructor.Mode.TOOLS
        # Safe default for unknown providers
        return instructor.Mode.JSON
