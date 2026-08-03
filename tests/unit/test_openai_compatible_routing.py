"""Model routing for OpenAI-compatible endpoints (issue #207).

``provider="openai_compatible"`` is the caller saying "this endpoint speaks
OpenAI". LiteLLM has no such provider, so the client has to translate that
into the ``openai/`` routing prefix it does understand. Before this fix it
did not, and both spellings failed:

* ``inclusionai/ling-2.6-flash`` was passed through untouched, because the
  old rule was "model contains a slash → already namespaced" — but here the
  slash belongs to the *model*, not a provider, so LiteLLM answered
  ``LLM Provider NOT provided``.
* ``ling-flash`` became ``openai_compatible/ling-flash``, naming a provider
  LiteLLM has never heard of.

Only ``openai/<model>`` worked, which forces every caller to say
"OpenAI-shaped" twice.
"""

import pytest

from ondine.adapters.unified_litellm_client import UnifiedLiteLLMClient
from ondine.core.specifications import LLMSpec

BASE_URL = "https://openrouter.ai/api/v1"


def _client(model: str, provider: str = "openai_compatible") -> UnifiedLiteLLMClient:
    return UnifiedLiteLLMClient(
        LLMSpec(
            provider=provider,
            model=model,
            base_url=BASE_URL,
            api_key="sk-test",  # pragma: allowlist secret
        )
    )


class TestOpenAICompatibleRouting:
    """provider="openai_compatible" routes through LiteLLM's openai/ prefix."""

    def test_vendor_namespaced_model_is_prefixed(self):
        """A vendor/model name is a model, not a provider — prefix it.

        This is the exact call from #207 that returned
        "LLM Provider NOT provided".
        """
        assert _client("inclusionai/ling-2.6-flash").model == (
            "openai/inclusionai/ling-2.6-flash"
        )

    def test_bare_model_is_prefixed_with_openai_not_the_ondine_name(self):
        """The prefix must be LiteLLM's provider, not ondine's own label."""
        assert _client("ling-flash").model == "openai/ling-flash"

    def test_already_prefixed_model_is_left_alone(self):
        """Callers who worked around this must not end up with openai/openai/."""
        assert _client("openai/gpt-4o-mini").model == "openai/gpt-4o-mini"

    def test_provider_name_reports_openai_for_capability_lookups(self):
        """Capability detection keys off the provider; it must see openai.

        Left as "openai_compatible" it matches nothing in the capability
        registry, so structured-output mode selection silently falls through.
        """
        assert _client("inclusionai/ling-2.6-flash").provider_name == "openai"


class TestOtherProvidersAreUnaffected:
    """The openai_compatible translation must not disturb normal routing."""

    def test_bare_model_still_takes_its_own_provider_prefix(self):
        client = UnifiedLiteLLMClient(
            LLMSpec(provider="groq", model="llama-3.3-70b-versatile", api_key="gsk-x")
        )
        assert client.model == "groq/llama-3.3-70b-versatile"

    @pytest.mark.parametrize(
        "model", ["groq/llama-3.3-70b-versatile", "openrouter/anthropic/claude-3.5"]
    )
    def test_litellm_format_model_is_passed_through(self, model):
        """A real provider prefix is already correct — don't touch it."""
        client = UnifiedLiteLLMClient(LLMSpec(provider="litellm", model=model))
        assert client.model == model
