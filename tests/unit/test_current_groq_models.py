"""Groq's current catalogue must be usable through every convenience path.

Groq retired its Llama and Mixtral models; what it serves now is namespaced by
vendor (``openai/gpt-oss-120b``). Three conveniences still knew only the
retired names, and each failed quietly rather than loudly:

* the context-window table fell back to 4,096 tokens for a 131,072-token
  model, so batch sizing rejected batches that fit thirty times over;
* the only Groq preset named a model that no longer exists;
* ``QuickPipeline`` read ``groq/openai/gpt-oss-20b`` as an OpenAI model.
"""

from decimal import Decimal

import pytest

from ondine.api.quick import QuickPipeline
from ondine.core.specifications import LLMProvider, LLMProviderPresets
from ondine.utils.model_context_limits import get_context_limit, validate_batch_size


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "groq/openai/gpt-oss-120b",
    ],
)
def test_current_groq_models_have_their_real_context_window(model):
    assert get_context_limit(model) == 131072


def test_a_batch_that_fits_a_current_groq_model_is_accepted():
    """50 prompts of 500 tokens is 25K — a fifth of the real window."""
    is_valid, message = validate_batch_size("openai/gpt-oss-120b", 50, 500)

    assert is_valid, message


@pytest.mark.parametrize(
    ("preset_name", "model", "input_per_1k", "output_per_1k"),
    [
        ("GROQ_GPT_OSS_120B", "openai/gpt-oss-120b", "0.00015", "0.0006"),
        ("GROQ_GPT_OSS_20B", "openai/gpt-oss-20b", "0.000075", "0.0003"),
    ],
)
def test_groq_presets_name_models_groq_serves(
    preset_name, model, input_per_1k, output_per_1k
):
    spec = getattr(LLMProviderPresets, preset_name)

    assert spec.provider == LLMProvider.GROQ
    assert spec.model == model
    assert spec.input_cost_per_1k_tokens == Decimal(input_per_1k)
    assert spec.output_cost_per_1k_tokens == Decimal(output_per_1k)


def test_quickpipeline_reads_a_groq_prefixed_model_as_groq():
    assert QuickPipeline._detect_provider("groq/openai/gpt-oss-20b") == "groq"
