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
        api_key="sk-...",
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
from decimal import Decimal
from typing import Any

import instructor
import litellm
from pydantic import BaseModel

from ondine.adapters.llm_client import LLMClient
from ondine.core.models import LLMResponse
from ondine.core.specifications import LLMSpec

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

        # Suppress LiteLLM noise
        litellm.suppress_debug_info = True
        litellm.drop_params = True
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)

        # Router support (optional)
        self.router = None
        if hasattr(spec, "router_config") and spec.router_config:
            self._init_router(spec.router_config)

        # Initialize Instructor client ONCE (not per-call!)
        completion_func = (
            self.router.acompletion if self.router else litellm.acompletion
        )
        self.instructor_client = instructor.from_litellm(completion_func)

        logger.debug(f"Initialized LiteLLM client: {self.model}")

    def _init_router(self, config: dict):
        """Initialize LiteLLM Router - just pass config through."""
        from litellm import Router

        try:
            # Map our 'debug' to LiteLLM's 'set_verbose'
            router_kwargs = dict(config)
            if router_kwargs.pop("debug", False):
                router_kwargs["set_verbose"] = True

            self.router = Router(**router_kwargs)
            self.model = config["model_list"][0][
                "model_name"
            ]  # Use router's model_name
            logger.info(f"Router initialized: {len(config['model_list'])} deployments")
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
            "api_key": self.api_key,  # Pass key directly (no env var!)
            "temperature": self.spec.temperature,
            "max_tokens": self.spec.max_tokens,
        }

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

        # Calculate cost (LiteLLM has pricing DB)
        cost = self._calc_cost(tokens_in, tokens_out)

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

    def _calc_cost(self, tokens_in: int, tokens_out: int) -> Decimal:
        """Calculate cost - try LiteLLM first, fallback to spec."""
        try:
            # LiteLLM has pricing for most models
            cost = litellm.completion_cost(
                model=self.model,
                prompt=tokens_in,
                completion=tokens_out,
            )
            return Decimal(str(cost)) if cost else Decimal("0")
        except Exception:
            # Fallback to manual if specified
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
            "api_key": self.api_key,  # Pass key directly
            "temperature": self.spec.temperature,
            "max_tokens": self.spec.max_tokens,
            "max_retries": 1,  # Ondine handles retries at pipeline level
        }

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

        # Serialize
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
        )
