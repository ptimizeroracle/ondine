"""A named provider decides where a request goes, whatever the model is called.

Hosts namespace the models they serve by the *vendor* that made them. Groq
serves ``openai/gpt-oss-20b`` and ``qwen/qwen3-32b``; Together serves
``meta-llama/...``. The client used to read any slash as "this model already
names its LiteLLM provider" and pass it through untouched, so

    .with_llm(provider="groq", model="openai/gpt-oss-20b")

was sent to **OpenAI** — a provider the caller did not choose — with whatever
``OPENAI_API_KEY`` happened to be in the environment. With no such key the
call failed with a "model does not exist" error naming a model the caller
never asked OpenAI for.

The rule these tests pin: when the caller names a provider, that provider is
the route. A slash in the model belongs to the model unless the model already
starts with that same provider's prefix.
"""

import pytest

from ondine.adapters.unified_litellm_client import UnifiedLiteLLMClient
from ondine.core.specifications import LLMSpec


def _routed_model(provider: str, model: str) -> str:
    """The model id the client will hand to LiteLLM for this spec."""
    return UnifiedLiteLLMClient(LLMSpec(provider=provider, model=model)).model


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        # The reported case: a Groq-hosted model whose vendor is OpenAI.
        ("groq", "openai/gpt-oss-20b", "groq/openai/gpt-oss-20b"),
        # A vendor that is not a LiteLLM provider at all.
        ("groq", "qwen/qwen3-32b", "groq/qwen/qwen3-32b"),
    ],
)
def test_vendor_namespaced_model_goes_to_the_named_provider(provider, model, expected):
    assert _routed_model(provider, model) == expected


def test_vendor_namespaced_model_reports_the_named_provider():
    """Capability lookups key off the provider; they must not see the vendor."""
    client = UnifiedLiteLLMClient(LLMSpec(provider="groq", model="openai/gpt-oss-20b"))
    assert client.provider_name == "groq"


def test_model_already_carrying_its_provider_prefix_is_not_doubled():
    """``groq/...`` under ``provider="groq"`` is the documented long form."""
    assert _routed_model("groq", "groq/llama-3.1-8b-instant") == (
        "groq/llama-3.1-8b-instant"
    )


def test_litellm_provider_still_passes_a_full_model_id_through():
    """``provider="litellm"`` is how a caller names the route in the model."""
    assert _routed_model("litellm", "openai/gpt-oss-20b") == "openai/gpt-oss-20b"
