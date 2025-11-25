"""
Minimal LiteLLM wrapper - thin abstraction for easy backend swaps.

Philosophy:
- LiteLLM handles ALL providers, routing, caching, retries
- This wrapper ONLY provides: sync/async normalization, structured output, Ondine response format
- ZERO provider-specific logic - just pass through to LiteLLM
- Clean interface makes future backend swaps trivial

Usage:
    # Any LiteLLM model format works (provider/model):
    spec = LLMSpec(model="openai/gpt-4o-mini", api_key="sk-...")  # pragma: allowlist secret
    spec = LLMSpec(model="groq/llama-3.3-70b-versatile", api_key="gsk_...")  # pragma: allowlist secret
    spec = LLMSpec(model="moonshot/kimi-k2-thinking-turbo", api_key="sk-...")  # pragma: allowlist secret

    # Advanced: Use extra_params for ANY LiteLLM feature
    spec = LLMSpec(
        model="openai/gpt-4o-mini",
        api_key="sk-...",  # pragma: allowlist secret
        extra_params={
            'stream': True,              # Streaming
            'caching': True,             # Enable caching
            'max_retries': 3,            # Built-in retries
            'top_p': 0.9,                # Sampling params
            'response_format': {...},    # JSON mode
            'tools': [...],              # Function calling
            # ... ANY LiteLLM param works!
        }
    )
    client = UnifiedLiteLLMClient(spec)

The extra_params pattern means:
- No code changes needed when LiteLLM adds new features
- User has full control over LiteLLM's capabilities
- Wrapper stays minimal and maintainable
"""

import asyncio
import logging
import time
import warnings
from decimal import Decimal
from typing import Any

import instructor
import litellm
from pydantic import BaseModel

from ondine.adapters.llm_client import LLMClient
from ondine.core.models import LLMResponse
from ondine.core.specifications import LLMSpec

# CRITICAL: Suppress Pydantic serialization warnings at module level
# LiteLLM's internal models trigger these harmless warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*Expected.*fields but got.*")
warnings.filterwarnings("ignore", message=".*serialized value may not be as expected.*")

logger = logging.getLogger(__name__)


class UnifiedLiteLLMClient(LLMClient):
    """
    Thin wrapper around LiteLLM - minimal abstraction.

    Just provides:
    - Async/sync interface
    - Structured output via Instructor
    - Ondine response format
    - Optional Router support

    Everything else (providers, retries, caching) = LiteLLM's job.
    """

    def __init__(self, spec: LLMSpec):
        super().__init__(spec)

        # Build model identifier for LiteLLM
        # If model has "/", use as-is (e.g., "moonshot/kimi-k2")
        # Otherwise, prepend provider (e.g., "groq" + "llama-3.3" → "groq/llama-3.3")
        if "/" in spec.model:
            self.model = spec.model  # Already has provider prefix
        else:
            # Get provider name
            provider_name = (
                spec.provider.value
                if hasattr(spec.provider, "value")
                else str(spec.provider)
            )
            provider_name = (
                provider_name.replace("litellm_", "").replace("litellm", "").strip()
            )

            # Only prepend if we have a non-empty provider
            if provider_name:
                self.model = f"{provider_name}/{spec.model}"
            else:
                self.model = spec.model  # No provider, hope model is complete

        self.api_key = spec.api_key

        # Suppress LiteLLM noise and async cleanup warnings
        litellm.suppress_debug_info = True
        litellm.drop_params = True
        litellm.set_verbose = False  # CRITICAL: Disable ALL LiteLLM internal logging
        logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
        logging.getLogger("LiteLLM Router").setLevel(logging.CRITICAL)
        logging.getLogger("LiteLLM Proxy").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.ERROR)

        # Suppress harmless async cleanup warnings that occur on script exit
        # (LiteLLM spawns background tasks that may not complete before event loop closes)
        import warnings

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings(
            "ignore", category=UserWarning
        )  # Pydantic serialization warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*Event loop is closed.*")
        warnings.filterwarnings("ignore", message=".*coroutine.*never awaited.*")
        warnings.filterwarnings(
            "ignore", message=".*PydanticSerializationUnexpectedValue.*"
        )
        warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")
        warnings.filterwarnings("ignore", message=".*Expected.*fields but got.*")
        warnings.filterwarnings(
            "ignore", message=".*serialized value may not be as expected.*"
        )

        # Suppress asyncio SSL transport errors (harmless cleanup noise)
        logging.getLogger("asyncio").setLevel(logging.CRITICAL)

        # Suppress LiteLLM's internal async client cleanup warnings
        logging.getLogger("litellm.llms.custom_httpx.async_client_cleanup").setLevel(
            logging.CRITICAL
        )

        # CRITICAL: Suppress Pydantic's internal warnings logger
        logging.getLogger("pydantic").setLevel(logging.CRITICAL)
        logging.getLogger("pydantic.warnings").setLevel(logging.CRITICAL)
        logging.getLogger("pydantic._internal").setLevel(logging.CRITICAL)

        # Router support (optional)
        self.router = None
        if hasattr(spec, "router_config") and spec.router_config:
            self._init_router(spec.router_config)

        # Initialize Instructor client ONCE (not per-call!)
        # Auto-detect mode based on provider:
        # - Groq: Use JSON mode (function calling is unreliable, generates XML)
        # - OpenAI/Anthropic: Use TOOLS mode (native function calling)
        # - Default: JSON mode (safest fallback)
        completion_func = (
            self.router.acompletion if self.router else litellm.acompletion
        )

        # Detect provider from model string
        model_lower = self.model.lower()
        if "groq" in model_lower:
            instructor_mode = instructor.Mode.JSON
            logger.debug("Using Instructor JSON mode for Groq (avoids XML issues)")
        elif any(p in model_lower for p in ["gpt", "openai", "claude", "anthropic"]):
            instructor_mode = instructor.Mode.TOOLS
            logger.debug("Using Instructor TOOLS mode for OpenAI/Anthropic")
        else:
            instructor_mode = instructor.Mode.JSON  # Safest default
            logger.debug("Using Instructor JSON mode (default)")

        self.instructor_client = instructor.from_litellm(
            completion_func, mode=instructor_mode
        )

        logger.debug(f"Initialized LiteLLM client: {self.model}")

    def _init_router(self, config: dict):
        """
        Initialize LiteLLM Router - generic wrapper behavior.

        Safety checks:
        - All deployments must share the same model_name (required for load balancing)
        - Maps 'debug' config to LiteLLM's 'set_verbose'
        """
        from litellm import Router

        from ondine.utils.rich_utils import display_router_deployments

        try:
            # Validate all deployments share same model_name (required for Router load balancing)
            model_names = {m["model_name"] for m in config["model_list"]}
            if len(model_names) != 1:
                raise ValueError(
                    f"Router requires all deployments to share the same model_name. "
                    f"Found {len(model_names)} different names: {model_names}. "
                    f"Use unique 'model_id' for tracking instead."
                )

            # Map our 'debug' to LiteLLM's 'set_verbose'
            router_kwargs = dict(config)
            verbose_mode = router_kwargs.pop("debug", False)
            if verbose_mode:
                router_kwargs["set_verbose"] = True

            self.router = Router(**router_kwargs)
            self.model = model_names.pop()  # Use shared model_name

            # Display Router info using Rich utilities
            strategy = router_kwargs.get("routing_strategy", "simple-shuffle")
            display_router_deployments(
                model_name=self.model,
                strategy=strategy,
                deployments=config["model_list"],
                verbose=verbose_mode or router_kwargs.get("set_verbose", False),
            )

            # ============================================================
            # CRITICAL: Aggressively suppress Router's internal JSON logging
            # ============================================================
            router_logger = logging.getLogger("LiteLLM Router")
            router_logger.setLevel(logging.CRITICAL)
            router_logger.propagate = False

            # Suppress litellm's underlying logger
            litellm_logger = logging.getLogger("litellm")
            litellm_logger.setLevel(logging.CRITICAL)
            litellm_logger.propagate = False

            # Remove all handlers to prevent JSON dumps
            for handler in router_logger.handlers[:]:
                router_logger.removeHandler(handler)
            for handler in litellm_logger.handlers[:]:
                litellm_logger.removeHandler(handler)
            # ============================================================
        except Exception as e:
            logger.error(f"Router init failed: {e}")
            self.router = None

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Async call to LLM - pass through to LiteLLM."""
        start = time.time()

        # Build messages
        messages = [{"role": "user", "content": prompt}]
        if kwargs.get("system_message"):
            messages.insert(0, {"role": "system", "content": kwargs["system_message"]})

        # Build call kwargs
        call_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.spec.temperature,
            "max_tokens": self.spec.max_tokens,
        }

        # CRITICAL: Only pass api_key if NOT using Router
        # Router has keys embedded in its model_list config!
        if not self.router and self.api_key:
            call_kwargs["api_key"] = self.api_key

        # Add api_base if custom endpoint
        if hasattr(self.spec, "base_url") and self.spec.base_url:
            call_kwargs["api_base"] = self.spec.base_url

        # CRITICAL: Pass through any extra params from spec (streaming, caching, retries, etc.)
        # This makes the wrapper truly "thin" - ANY LiteLLM param works without code changes!
        if hasattr(self.spec, "extra_params") and self.spec.extra_params:
            call_kwargs.update(self.spec.extra_params)

        # Call LiteLLM (Router or direct)
        if self.router:
            response = await self.router.acompletion(**call_kwargs)
        else:
            response = await litellm.acompletion(**call_kwargs)

        # Extract response
        text = response.choices[0].message.content
        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0
        latency_ms = (time.time() - start) * 1000

        # Calculate cost (LiteLLM has pricing DB) - pass response object for accurate cost
        cost = self._calc_cost_from_response(response)

        return LLMResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=self.model,
            cost=cost,
            latency_ms=latency_ms,
        )

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Sync wrapper that works in both scripts and Jupyter notebooks.

        Automatically detects if running in an async context (Jupyter, FastAPI, etc.)
        and handles it appropriately.
        """
        try:
            # Check if we're already in an event loop (Jupyter, FastAPI, etc.)
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop running - safe to use asyncio.run() (script mode)
            return asyncio.run(self.ainvoke(prompt, **kwargs))
        else:
            # Already in a loop - schedule and wait (Jupyter mode)
            future = asyncio.run_coroutine_threadsafe(
                self.ainvoke(prompt, **kwargs), loop
            )
            return future.result()

    def _calc_cost_from_response(self, response: Any) -> Decimal:
        """
        Calculate cost from LiteLLM response object.

        Uses LiteLLM's completion_cost() with the full response for accurate pricing.
        """
        try:
            # CRITICAL: Pass the full response AND our original model string
            # Response.model may have provider prefix stripped by LiteLLM
            cost = litellm.completion_cost(
                completion_response=response,
                model=self.model,  # Use our stored model string with provider prefix
            )
            logger.debug(f"LiteLLM cost for {self.model}: ${cost}")
            return Decimal(str(cost)) if cost else Decimal("0")
        except Exception as e:
            # Fallback to token-based calculation
            logger.warning(
                f"completion_cost failed for {self.model}: {e}, falling back to manual calculation"
            )
            tokens_in = response.usage.prompt_tokens if response.usage else 0
            tokens_out = response.usage.completion_tokens if response.usage else 0
            return self._calc_cost(tokens_in, tokens_out)

    def _calc_cost(self, tokens_in: int, tokens_out: int) -> Decimal:
        """
        Calculate cost using token counts (fallback method).

        Framework-wide behavior:
        1. Falls back to manual calculation if pricing available in spec
        2. Returns $0 if pricing unavailable (prevents pipeline errors)

        Pipelines consume response.cost without needing to know calculation method.
        """
        try:
            # Fallback to manual calculation if specified in spec
            if (
                self.spec.input_cost_per_1k_tokens
                and self.spec.output_cost_per_1k_tokens
            ):
                input_cost = (
                    Decimal(tokens_in / 1000) * self.spec.input_cost_per_1k_tokens
                )
                output_cost = (
                    Decimal(tokens_out / 1000) * self.spec.output_cost_per_1k_tokens
                )
                return input_cost + output_cost
            # Return $0 if pricing unavailable (avoids breaking pipelines)
            return Decimal("0")
        except Exception:
            return Decimal("0")

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using LiteLLM."""
        try:
            return len(litellm.encode(model=self.model, text=text))
        except Exception:
            return int(len(text.split()) * 1.3)

    def calculate_cost(self, tokens_in: int, tokens_out: int) -> Decimal:
        """Public cost calculation."""
        return self._calc_cost(tokens_in, tokens_out)

    def structured_invoke(
        self, prompt: str, output_cls: type[BaseModel], **kwargs: Any
    ) -> LLMResponse:
        """
        Sync wrapper for structured output that works in both scripts and Jupyter.

        Automatically detects if running in an async context and handles it appropriately.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop - use asyncio.run()
            return asyncio.run(
                self.structured_invoke_async(prompt, output_cls, **kwargs)
            )
        else:
            # Loop exists - use run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(
                self.structured_invoke_async(prompt, output_cls, **kwargs), loop
            )
            return future.result()

    async def structured_invoke_async(
        self, prompt: str, output_cls: type[BaseModel], **kwargs: Any
    ) -> LLMResponse:
        """Structured output via Instructor - use pre-initialized client."""
        start = time.time()

        # Build messages
        messages = [{"role": "user", "content": prompt}]
        if kwargs.get("system_message"):
            messages.insert(0, {"role": "system", "content": kwargs["system_message"]})

        # Build call kwargs
        call_kwargs = {
            "model": self.model,
            "messages": messages,
            "response_model": output_cls,
            "temperature": self.spec.temperature,
            "max_tokens": self.spec.max_tokens,
        }

        # CRITICAL: Only pass api_key if NOT using Router
        # Router has keys embedded in its model_list config!
        if not self.router and self.api_key:
            call_kwargs["api_key"] = self.api_key

        # Retry strategy: Default to 1 (Ondine handles retries at pipeline level)
        # Allow override via extra_params if needed
        call_kwargs["max_retries"] = (
            self.spec.extra_params.get("max_retries", 1)
            if hasattr(self.spec, "extra_params") and self.spec.extra_params
            else 1
        )

        if hasattr(self.spec, "base_url") and self.spec.base_url:
            call_kwargs["api_base"] = self.spec.base_url

        # CRITICAL: Pass through extra params (caching, response_format, tools, etc.)
        if hasattr(self.spec, "extra_params") and self.spec.extra_params:
            call_kwargs.update(self.spec.extra_params)

        # Call with pre-initialized Instructor client
        try:
            result = await self.instructor_client.chat.completions.create(**call_kwargs)
        except Exception as e:
            raise ValueError(f"Structured prediction failed: {e}") from e

        # Serialize for backward compatibility (text field)
        text = result.model_dump_json()

        # Estimate tokens (Instructor doesn't expose usage)
        full_prompt = (
            f"{kwargs.get('system_message', '')}\n\n{prompt}"
            if kwargs.get("system_message")
            else prompt
        )
        tokens_in = self.estimate_tokens(full_prompt)
        tokens_out = self.estimate_tokens(text)
        latency_ms = (time.time() - start) * 1000

        cost = self._calc_cost(tokens_in, tokens_out)

        return LLMResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=self.model,
            cost=cost,
            latency_ms=latency_ms,
            structured_result=result,  # CRITICAL: Keep Pydantic object (avoids re-parsing!)
        )
