"""
LiteLLM unified client implementation using LlamaIndex's LiteLLM wrapper.

Provides a single client class that supports 100+ LLM providers through
LlamaIndex's LiteLLM integration, with automatic cost tracking and full
compatibility with LlamaIndex features (structured_predict, observability, etc.).
"""

import os
import time
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from ondine.adapters.llm_client import LLMClient
from ondine.core.models import LLMResponse
from ondine.core.specifications import LLMSpec


class LiteLLMClient(LLMClient):
    """
    Unified LLM client using LlamaIndex's LiteLLM wrapper.

    This is the BEST of both worlds:
    - Uses LlamaIndex's LiteLLM class (maintains compatibility)
    - Gets LiteLLM features (auto cost tracking, 100+ providers, fallbacks)
    - Supports structured_predict() natively (Pydantic validation)
    - Works with existing observability, caching, and tooling

    Replaces provider-specific clients with a single, unified implementation.

    Example:
        # OpenAI (same as before, but through LiteLLM)
        spec = LLMSpec(provider="openai", model="gpt-4o-mini")
        client = LiteLLMClient(spec)

        # Groq (no special XML workaround needed!)
        spec = LLMSpec(provider="groq", model="llama-3.3-70b-versatile")
        client = LiteLLMClient(spec)

        # Anthropic (no validation bug workaround needed!)
        spec = LLMSpec(provider="anthropic", model="claude-3-5-haiku-20241022")
        client = LiteLLMClient(spec)
    """

    def __init__(self, spec: LLMSpec):
        """
        Initialize LiteLLM client using LlamaIndex's wrapper.

        Args:
            spec: LLM specification with provider, model, and credentials
        """
        import logging

        from llama_index.llms.litellm import LiteLLM

        # Suppress verbose LiteLLM logs (keep only warnings/errors)
        # LiteLLM logs every completion() call at INFO level which is too noisy
        # Also suppress print statements from litellm library

        os.environ["LITELLM_LOG"] = "WARNING"  # Set log level via env var
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("litellm").setLevel(logging.WARNING)

        # Suppress litellm's success_callback logging
        try:
            import litellm

            litellm.suppress_debug_info = True
            litellm.drop_params = True
        except (ImportError, AttributeError):
            pass  # Older version of litellm

        self.spec = spec
        self.model = spec.model
        self.temperature = spec.temperature
        self.max_tokens = spec.max_tokens

        # Build LiteLLM model identifier (format: "provider/model")
        # Use custom_provider_id if available (e.g., "litellm_groq")
        # Otherwise fall back to the enum value
        if spec.custom_provider_id:
            provider_name = spec.custom_provider_id
        else:
            provider_name = (
                spec.provider.value
                if hasattr(spec.provider, "value")
                else str(spec.provider)
            )

        # Remove "litellm_" prefix to get actual provider (e.g., "litellm_groq" → "groq")
        provider_name = provider_name.replace("litellm_", "")

        # Build LiteLLM model identifier (format: "provider/model")
        if spec.model.startswith(f"{provider_name}/"):
            # Already has CORRECT provider prefix
            model_identifier = spec.model
        else:
            # Add provider prefix to ensure correct LiteLLM routing
            # Even if model has other slashes (e.g. "openai/gpt-oss"), we must prefix with provider
            # Example: provider="groq", model="openai/gpt-oss" -> "groq/openai/gpt-oss"
            model_identifier = f"{provider_name}/{spec.model}"

        # Set API key environment variable if provided
        if spec.api_key:
            env_var = self._get_api_key_env_var(provider_name)
            os.environ[env_var] = spec.api_key

        # Initialize LlamaIndex's LiteLLM wrapper
        # CRITICAL: Pass the FULL "provider/model" string
        # This tells LlamaIndex to route through LiteLLM, not OpenAI client
        self.client = LiteLLM(
            model=model_identifier,  # e.g., "groq/llama-3.3-70b-versatile"
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
        )

        # Store for cost calculation and token counting
        self.model_identifier = model_identifier

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Invoke LLM through LlamaIndex's LiteLLM wrapper.

        Uses the same ChatMessage API as other LlamaIndex providers,
        ensuring full compatibility with observability and caching.

        Args:
            prompt: Text prompt
            **kwargs: Additional parameters (system_message, etc.)

        Returns:
            LLMResponse with standardized output and automatic cost tracking
        """
        from llama_index.core.llms import ChatMessage

        start_time = time.time()

        # Build messages array (same pattern as OpenAIClient)
        messages = []

        # Add system message if provided (supports prompt caching)
        system_message = kwargs.get("system_message")
        if system_message and self.spec.enable_prefix_caching:
            messages.append(ChatMessage(role="system", content=system_message))

        # Add user prompt
        messages.append(ChatMessage(role="user", content=prompt))

        # Call LlamaIndex's LiteLLM wrapper (same API as OpenAI, Groq, etc.)
        response = self.client.chat(messages)

        latency_ms = (time.time() - start_time) * 1000

        # Extract token usage from response
        tokens_in = 0
        tokens_out = 0

        if hasattr(response, "raw") and response.raw and hasattr(response.raw, "usage"):
            usage = response.raw.usage
            tokens_in = getattr(usage, "prompt_tokens", 0)
            tokens_out = getattr(usage, "completion_tokens", 0)

        # Calculate cost using LiteLLM's pricing database
        cost = self._calculate_cost_from_litellm(tokens_in, tokens_out)

        return LLMResponse(
            text=response.message.content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=self.model,
            cost=cost,
            latency_ms=latency_ms,
        )

    def structured_invoke(
        self,
        prompt: str,
        output_cls: type[BaseModel],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Invoke LLM with structured output using LlamaIndex's structured_predict.

        For Groq: Uses LLMTextCompletionProgram workaround to avoid XML tool-use bug.
        For others: Uses native structured_predict with function calling.

        Args:
            prompt: Text prompt
            output_cls: Pydantic model class for output validation
            **kwargs: Additional parameters

        Returns:
            LLMResponse with validated structured output
        """
        from llama_index.core import PromptTemplate
        from llama_index.core.llms import ChatMessage

        start_time = time.time()

        # Extract system_message from kwargs
        system_message = kwargs.pop("system_message", None)

        # GROQ WORKAROUND: Avoid XML tool-use bug
        # Groq's LlamaIndex integration produces XML-wrapped tool calls that fail validation
        # Use LLMTextCompletionProgram with JSON mode instead
        if "groq/" in self.model_identifier.lower():
            try:
                from llama_index.core.program import LLMTextCompletionProgram

                program = LLMTextCompletionProgram.from_defaults(
                    output_cls=output_cls,
                    llm=self.client,
                    prompt_template_str="{prompt}",
                )

                # Wrap prompt
                if isinstance(prompt, str):
                    prompt_tmpl = PromptTemplate(prompt)
                else:
                    prompt_tmpl = prompt

                # Call with JSON mode
                result_obj = program(
                    prompt=prompt_tmpl,
                    llm_kwargs={"response_format": {"type": "json_object"}},
                )

                latency_ms = (time.time() - start_time) * 1000

                # Handle validation bug
                if isinstance(result_obj, str):
                    raise ValueError(
                        f"Model returned validation error: {result_obj[:200]}"
                    )

                response_text = result_obj.model_dump_json()

                # Estimate tokens
                tokens_in = self.estimate_tokens(prompt)
                if system_message:
                    tokens_in += self.estimate_tokens(system_message)
                tokens_out = self.estimate_tokens(response_text)

                cost = self._calculate_cost_from_litellm(tokens_in, tokens_out)

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

        # Use LlamaIndex's native structured prediction for non-Groq providers
        try:
            if hasattr(self.client, "structured_predict"):
                # Wrap prompt in PromptTemplate
                if isinstance(prompt, str):
                    prompt_tmpl = PromptTemplate(prompt)
                else:
                    prompt_tmpl = prompt

                # Build messages array if system_message is provided
                if system_message:
                    messages = [
                        ChatMessage(role="system", content=system_message),
                        ChatMessage(role="user", content=prompt),
                    ]
                    result_obj = self.client.structured_predict(
                        output_cls,
                        messages=messages,
                    )
                else:
                    result_obj = self.client.structured_predict(
                        output_cls,
                        prompt=prompt_tmpl,
                    )
            else:
                # Fallback: Use LLMTextCompletionProgram
                from llama_index.core.program import LLMTextCompletionProgram

                program = LLMTextCompletionProgram.from_defaults(
                    output_cls=output_cls,
                    llm=self.client,
                    prompt_template_str="{prompt}",
                )
                result_obj = program(prompt=prompt)

            latency_ms = (time.time() - start_time) * 1000

            # Handle Anthropic validation bug (returns string on error)
            if isinstance(result_obj, str):
                raise ValueError(
                    f"Model returned validation error instead of structured object: {result_obj[:200]}"
                )

            # Serialize result to JSON for pipeline consistency
            response_text = result_obj.model_dump_json()

            # Estimate tokens (LiteLLM provides better counting)
            tokens_in = self.estimate_tokens(prompt)
            if system_message:
                tokens_in += self.estimate_tokens(system_message)
            tokens_out = self.estimate_tokens(response_text)

            # Calculate cost using LiteLLM's pricing database
            cost = self._calculate_cost_from_litellm(tokens_in, tokens_out)

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

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using LiteLLM's token counting.

        LiteLLM uses provider-specific logic, more accurate than generic tiktoken.
        """
        try:
            from litellm import token_counter

            return token_counter(model=self.model_identifier, text=text)
        except Exception:
            # Fallback to word-based estimation
            return int(len(text.split()) * 1.3)

    def calculate_cost(self, tokens_in: int, tokens_out: int) -> Decimal:
        """
        Calculate cost using LiteLLM's pricing database.

        AUTO-MAGIC: No need to manually configure costs!
        LiteLLM knows pricing for 1818+ models.
        """
        return self._calculate_cost_from_litellm(tokens_in, tokens_out)

    def _calculate_cost_from_litellm(self, tokens_in: int, tokens_out: int) -> Decimal:
        """Calculate cost using LiteLLM's pricing database."""
        try:
            from litellm import completion_cost

            # LiteLLM has built-in pricing for 1818+ models
            cost = completion_cost(
                model=self.model_identifier,
                prompt_tokens=tokens_in,
                completion_tokens=tokens_out,
            )
            return Decimal(str(cost)) if cost else Decimal("0.0")
        except Exception:
            # Fallback to manual calculation if model not in database
            input_cost = Decimal(str(tokens_in / 1000)) * (
                self.spec.input_cost_per_1k_tokens or Decimal("0.0")
            )
            output_cost = Decimal(str(tokens_out / 1000)) * (
                self.spec.output_cost_per_1k_tokens or Decimal("0.0")
            )
            return input_cost + output_cost

    def _get_api_key_env_var(self, provider_name: str) -> str:
        """Get environment variable name for provider API key."""
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
            "azure_openai": "AZURE_API_KEY",
        }
        return env_var_map.get(provider_name, f"{provider_name.upper()}_API_KEY")
