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
import os
import time
import warnings
from decimal import Decimal
from typing import Any

import instructor
import litellm
from pydantic import BaseModel

# Import Ondine Exceptions for mapping
from ondine.core.exceptions import (
    InvalidAPIKeyError,
    ModelNotFoundError,
    QuotaExceededError,
)
from ondine.utils.retry_handler import NetworkError
from ondine.utils.retry_handler import RateLimitError as OndineRateLimitError

try:
    import aiohttp
    from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler

    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False

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

        # Initialize Cache (optional)
        # Supports Redis, Disk, S3, etc. via LiteLLM native caching
        if hasattr(spec, "cache_config") and spec.cache_config:
            try:
                from litellm.caching import Cache

                litellm.cache = Cache(**spec.cache_config)
                logger.debug(f"Initialized LiteLLM cache: {spec.cache_config}")
            except Exception as e:
                logger.warning(f"Failed to initialize LiteLLM cache: {e}")

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

        # Global connection pooling state
        self._aiohttp_session = None

    async def start(self):
        """
        Initialize global high-performance connection pool (aiohttp).

        This enables Proxy-level performance (1000+ concurrent requests) by reusing
        connections across all LiteLLM calls.
        """
        if not _HAS_AIOHTTP:
            logger.warning("aiohttp not installed. Global connection pooling disabled.")
            return

        if self._aiohttp_session:
            return  # Already started

        try:
            # Load limits from env or default to high performance
            total_limit = int(os.getenv("LITELLM_POOL_LIMIT", "1000"))
            host_limit = int(os.getenv("LITELLM_POOL_LIMIT_PER_HOST", "100"))

            # Create high-performance session
            # Limits match LiteLLM Proxy defaults for high throughput
            self._aiohttp_session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    limit=total_limit,  # Max total connections
                    limit_per_host=host_limit,  # Max per provider (e.g. OpenAI)
                    ttl_dns_cache=600,  # Cache DNS for 10 mins
                    keepalive_timeout=60,  # Keep idle connections open
                ),
                timeout=aiohttp.ClientTimeout(total=600),  # 10 min timeout for batches
            )

            # Inject into LiteLLM Global Handler
            # This makes ALL Router/Completion calls use this session
            litellm.base_llm_aiohttp_handler = BaseLLMAIOHTTPHandler(
                client_session=self._aiohttp_session
            )
            logger.info(
                f"🚀 Initialized global high-performance connection pool (limit={total_limit}, per_host={host_limit})"
            )

            # Perform connectivity check to prune dead providers
            await self.verify_connectivity()

        except Exception as e:
            logger.error(f"Failed to initialize global connection pool: {e}")
            # Graceful degradation: litellm will create its own sessions

    async def verify_connectivity(self):
        """
        Verify connectivity for all configured providers.

        Prunes dead/invalid providers from the Router's model_list to prevent
        runtime errors during batch processing.
        """
        if not self.router:
            return

        logger.info("🏥 Performing Pre-flight Health Check...")

        # Access internal model list (LiteLLM Router stores it here)
        if not hasattr(self.router, "model_list") or not self.router.model_list:
            return

        working_models = []
        failed_models = []

        # We need to iterate a copy because we might modify the list
        for model in self.router.model_list:
            # Get model alias for display
            model_info = model.get("litellm_params", {})
            model_name = model_info.get("model", "unknown")
            alias = model.get("model_name", model_name)  # The routing alias

            # Create a friendly display name
            display_name = f"{model_name} ({alias})"

            # Skip if no API key (unless it's local/no-auth)
            # Actually, just try pinging it.

            try:
                # Send minimal ping (1 token)
                # Use litellm.acompletion directly with specific params
                # We bypass the router to test the specific deployment
                logger.info(f"  👉 Testing {display_name}...")

                # Construct clean ping args to avoid conflicts (e.g. user max_tokens vs ping max_tokens)
                # Explicitly select ONLY what we need for a ping
                ping_kwargs = {
                    "model": model_info.get("model"),
                    "api_key": model_info.get("api_key"),
                    "api_base": model_info.get("api_base"),
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1,
                    "timeout": 10,  # Fast fail
                }
                # Filter out None values
                ping_kwargs = {k: v for k, v in ping_kwargs.items() if v is not None}

                await litellm.acompletion(**ping_kwargs)
                logger.info("     ✅ OK")
                working_models.append(model)

            except Exception as e:
                error_msg = str(e).split("\n")[0][:100]
                logger.warning(f"     ❌ FAILED: {error_msg}")
                failed_models.append(model)

        # If we found failures, just warn (User requested to NOT remove them)
        if failed_models:
            logger.warning(
                f"⚠️ Found {len(failed_models)} unhealthy providers, but keeping them in Router as requested."
            )
            if not working_models:
                logger.error(
                    "❌ ALL providers failed health check! Pipeline will likely fail."
                )

            # self.router.model_list = working_models # DISABLED: Keep all providers

    async def stop(self):
        """Cleanup global connection pool."""
        if self._aiohttp_session:
            try:
                # Unset global handler first
                if getattr(litellm, "base_llm_aiohttp_handler", None):
                    litellm.base_llm_aiohttp_handler = None

                await self._aiohttp_session.close()
                self._aiohttp_session = None
                logger.info("🔌 Closed global connection pool")
            except Exception as e:
                logger.error(f"Error closing connection pool: {e}")

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

    def _map_provider_error(self, error: Exception) -> Exception:
        """
        Map provider-specific exceptions to Ondine domain exceptions.

        Centralizes error classification logic to decouple stages from provider details.
        Distinguishes between Fatal (NonRetryable) and Transient (Retryable) errors.

        Args:
            error: The raw exception from LiteLLM/Provider

        Returns:
            Mapped Ondine exception or original exception if no mapping exists
        """
        error_str = str(error).lower()

        # 1. Unwrap Instructor Retry Exceptions (recursively)
        try:
            from instructor.core.exceptions import InstructorRetryException

            if isinstance(error, InstructorRetryException):
                # Try to find the underlying cause
                if hasattr(error, "last_attempt") and error.last_attempt:
                    if hasattr(error.last_attempt, "exception"):
                        inner_exc = error.last_attempt.exception()
                        if inner_exc:
                            # Recurse to map the inner exception
                            return self._map_provider_error(inner_exc)
                elif hasattr(error, "args") and error.args:
                    if isinstance(error.args[0], Exception):
                        return self._map_provider_error(error.args[0])
        except ImportError:
            pass

        # 2. Check for Network errors (Retryable) - CHECK FIRST
        if (
            "network" in error_str
            or "timeout" in error_str
            or "connection" in error_str
            or "service unavailable" in error_str
            or "503" in error_str
            or "502" in error_str
        ):
            provider_info = ""
            if hasattr(error, "model") and error.model and error.model != "mixed-llm":
                provider_info = f" [Provider: {error.model}]"
            return NetworkError(f"{str(error)}{provider_info}")

        # 3. Check for Quota/Billing errors (Fatal) - CHECK BEFORE RATE LIMIT
        # Because Providers often return 429 for BOTH Rate Limit and Quota
        quota_patterns = [
            "quota exceeded",
            "insufficient_quota",
            "billing",
            "credits exhausted",
            "account suspended",
            "payment required",
            "tokens per day limit exceeded",  # Cerebras Quota
            "tokens per hour limit exceeded",  # Cerebras Quota
            "tokens per month limit exceeded",  # Cerebras Quota
        ]
        if any(p in error_str for p in quota_patterns):
            return QuotaExceededError(f"Quota error: {error}")

        # 4. Check for Rate Limit (Retryable)
        # LiteLLM usually wraps these in its own RateLimitError, but check string too
        if (
            "rate" in error_str
            or "429" in error_str
            or isinstance(error, litellm.RateLimitError)
        ):
            return OndineRateLimitError(str(error))

        # 5. Check for Authentication errors (Fatal)
        auth_patterns = [
            "invalid api key",
            "authentication failed",
            "401",
            "403",
            "unauthorized",
            "invalid credentials",
            "permission denied",
        ]
        # Check OpenAI/Anthropic specific auth errors types if available
        try:
            from openai import AuthenticationError as OpenAIAuthError

            if isinstance(error, OpenAIAuthError):
                return InvalidAPIKeyError(f"OpenAI authentication failed: {error}")
        except ImportError:
            pass

        try:
            from anthropic import AuthenticationError as AnthropicAuthError

            if isinstance(error, AnthropicAuthError):
                return InvalidAPIKeyError(f"Anthropic authentication failed: {error}")
        except ImportError:
            pass

        if any(p in error_str for p in auth_patterns):
            return InvalidAPIKeyError(f"Authentication error: {error}")

        # 6. Check for Model Not Found (Fatal)
        model_patterns = [
            "decommissioned",
            "not found",
            "does not exist",
            "invalid model",
            "unknown model",
        ]
        if any(p in error_str for p in model_patterns):
            return ModelNotFoundError(f"Model error: {error}")

        # Return original if no mapping matches
        return error

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
        try:
            if self.router:
                response = await self.router.acompletion(**call_kwargs)
            else:
                response = await litellm.acompletion(**call_kwargs)
        except Exception as e:
            raise self._map_provider_error(e)

        # Extract response
        text = response.choices[0].message.content
        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0
        latency_ms = (time.time() - start) * 1000

        # Calculate cost (LiteLLM has pricing DB) - pass response object for accurate cost
        cost = self._calc_cost_from_response(response)

        # Check for prompt caching (OpenAI/Azure/Anthropic/Groq)
        self._check_cache_hit(response, tokens_in)

        # Extract Router deployment info (if available)
        # LiteLLM Router stores actual deployment ID in _hidden_params
        metadata = {}
        if self.router:
            # Try multiple methods to extract deployment info
            if hasattr(response, "_hidden_params") and response._hidden_params:
                hidden = response._hidden_params
                if isinstance(hidden, dict):
                    # Method 1: model_id field
                    if "model_id" in hidden:
                        metadata["model_id"] = hidden["model_id"]
                    # Method 2: model_region or custom_llm_provider
                    elif "model_region" in hidden:
                        metadata["model_id"] = hidden["model_region"]

        return LLMResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=self.model,
            cost=cost,
            latency_ms=latency_ms,
            metadata=metadata,  # Pass deployment info to stage
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

        Simple logic: LiteLLM already calculated the cost, just use it!
        """
        # SIMPLE: Check if LiteLLM already calculated cost (it's in _hidden_params)
        if hasattr(response, "_hidden_params") and response._hidden_params:
            hidden = response._hidden_params
            if isinstance(hidden, dict) and "response_cost" in hidden:
                cost = hidden["response_cost"]
                if cost and cost > 0:
                    return Decimal(str(cost))

        # Fallback: Calculate ourselves
        try:
            model_to_use = self.model
            if self.router and hasattr(response, "model") and response.model:
                model_to_use = response.model

            cost = litellm.completion_cost(
                completion_response=response,
                model=model_to_use,
            )
            if cost and cost > 0:
                return Decimal(str(cost))
        except Exception:
            pass

        # Last resort: Manual calculation from spec
        if self.spec.input_cost_per_1k_tokens or self.spec.output_cost_per_1k_tokens:
            tokens_in = response.usage.prompt_tokens if response.usage else 0
            tokens_out = response.usage.completion_tokens if response.usage else 0
            return self._calc_cost(tokens_in, tokens_out)

        return Decimal("0")

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
        raw_response = None
        try:
            # Try to get raw response for metadata extraction
            # Instructor >= 1.0.0 supports create_with_completion
            if hasattr(
                self.instructor_client.chat.completions, "create_with_completion"
            ):
                (
                    result,
                    raw_response,
                ) = await self.instructor_client.chat.completions.create_with_completion(
                    **call_kwargs
                )
            else:
                # Fallback for older versions
                result = await self.instructor_client.chat.completions.create(
                    **call_kwargs
                )
        except Exception as e:
            raise self._map_provider_error(e)

        # Serialize for backward compatibility (text field)
        text = result.model_dump_json()

        # Calculate tokens and cost
        # If we have raw_response, use it for accurate usage/cost!
        if raw_response:
            tokens_in = raw_response.usage.prompt_tokens if raw_response.usage else 0
            tokens_out = (
                raw_response.usage.completion_tokens if raw_response.usage else 0
            )
            # Use LiteLLM's cost calculation if available
            cost = self._calc_cost_from_response(raw_response)

            # Check for prompt caching (OpenAI/Azure/Anthropic/Groq)
            self._check_cache_hit(raw_response, tokens_in)
        else:
            # Fallback estimation
            full_prompt = (
                f"{kwargs.get('system_message', '')}\n\n{prompt}"
                if kwargs.get("system_message")
                else prompt
            )
            tokens_in = self.estimate_tokens(full_prompt)
            tokens_out = self.estimate_tokens(text)
            cost = self._calc_cost(tokens_in, tokens_out)

        latency_ms = (time.time() - start) * 1000

        # Extract Router deployment info (Instructor path)
        metadata = {}
        if self.router:
            # Use raw_response if available
            source = raw_response if raw_response else result

            # Check for _hidden_params
            if hasattr(source, "_hidden_params") and source._hidden_params:
                hidden = source._hidden_params
                if isinstance(hidden, dict):
                    # Method 1: model_id field
                    if "model_id" in hidden:
                        metadata["model_id"] = hidden["model_id"]
                    # Method 2: model_region or custom_llm_provider
                    elif "model_region" in hidden:
                        metadata["model_id"] = hidden["model_region"]

            # Fallback: Check if result has _raw_response (some instructor versions)
            if not metadata and hasattr(result, "_raw_response"):
                raw = result._raw_response
                if hasattr(raw, "_hidden_params") and raw._hidden_params:
                    hidden = raw._hidden_params
                    if isinstance(hidden, dict) and "model_id" in hidden:
                        metadata["model_id"] = hidden["model_id"]

        return LLMResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=self.model,
            cost=cost,
            latency_ms=latency_ms,
            structured_result=result,  # CRITICAL: Keep Pydantic object (avoids re-parsing!)
            metadata=metadata,  # Pass deployment info
        )

    def _check_cache_hit(self, response: Any, tokens_in: int) -> None:
        """
        Check for provider-side prompt caching (OpenAI/Azure/Anthropic).

        If cached tokens are found, logs a DEBUG message.
        Uses ONDINE_LOG_LEVEL environment variable (default INFO) to control visibility.
        """
        try:
            cached_tokens = 0
            # 1. Check standard OpenAI/Groq format (usage.prompt_tokens_details.cached_tokens)
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                if (
                    hasattr(usage, "prompt_tokens_details")
                    and usage.prompt_tokens_details
                ):
                    cached_tokens = getattr(
                        usage.prompt_tokens_details, "cached_tokens", 0
                    )

            # 2. Check Anthropic format (usage.cache_creation_input_tokens / cache_read_input_tokens)
            # LiteLLM normalizes this, but checking raw just in case
            if cached_tokens == 0 and hasattr(response, "usage"):
                cached_tokens = getattr(response.usage, "cache_read_input_tokens", 0)

            # Log if hit
            if cached_tokens > 0:
                # Try to get actual model name from response
                # 1. response.model (usually the human-readable model name, e.g., 'gpt-4o-mini')
                # 2. hidden params (deployment ID, sometimes a hash)
                # 3. self.model (fallback to 'mixed-llm')
                actual_model = getattr(response, "model", None)

                if not actual_model or "mixed-llm" in actual_model:
                    if hasattr(response, "_hidden_params"):
                        hidden = response._hidden_params
                        if isinstance(hidden, dict):
                            # Prefer model_region (often provider/model) over model_id (hash)
                            if "model_region" in hidden:
                                actual_model = hidden["model_region"]
                            elif "model_id" in hidden:
                                actual_model = hidden["model_id"]

                if not actual_model:
                    actual_model = self.model

                cache_pct = (cached_tokens / tokens_in * 100) if tokens_in > 0 else 0
                logger.debug(
                    f"✅ Cache hit! ({actual_model}) {cached_tokens}/{tokens_in} tokens cached ({cache_pct:.0f}%)"
                )
        except Exception:
            # Don't crash on logging errors
            pass
