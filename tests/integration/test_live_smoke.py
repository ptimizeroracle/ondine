"""A few real calls a day, so the outside world cannot break ondine unnoticed.

Everything else in the suite runs against fakes or skips without an API key,
and CI has no keys — so for weeks nothing noticed that a provider had retired
the models the docs recommended, or that a named provider was being routed to
a different one (#271). Both were visible to the first user who tried, and to no test.

This file is the tripwire. It is deliberately small: one provider path per
test, a handful of rows, answers checked by value. The scheduled
``live-smoke`` workflow runs it against the *published* package with freshly
resolved dependencies, which is what a new user gets from ``pip install``.

Two rules keep it honest:

* **A required provider never skips.** ``ONDINE_LIVE_REQUIRED`` lists the
  providers the run must exercise; a missing key for one of them is a
  failure. A smoke test that skips itself green is how the gap opened.
* **Answers are checked, not just counted.** Each row asks a different
  question, so a batch that comes back misaligned or half-empty fails here
  rather than returning three non-null cells.
"""

import os
import re

import pandas as pd
import pytest
from pydantic import BaseModel

from ondine import PipelineBuilder
from tests.integration.live_models import FREE_KEY_ENV, FREE_MODEL, FREE_PROVIDER

pytestmark = pytest.mark.integration

#: Questions with one-token answers no competent model gets wrong, each
#: different so a row holding its neighbour's answer is detectable.
ARITHMETIC = pd.DataFrame({"question": ["2 + 2", "10 - 3", "3 * 3"]})
EXPECTED = ["4", "7", "9"]

#: ``(provider, model, api_key_env)``; the id is the name ONDINE_LIVE_REQUIRED
#: uses. OpenRouter's free tier allows 50 requests a day, and this file makes
#: seven, so it fits with room for a manual re-run.
PROVIDERS = [
    pytest.param(FREE_PROVIDER, FREE_MODEL, FREE_KEY_ENV, id="openrouter"),
    pytest.param("openai", "gpt-4o-mini", "OPENAI_API_KEY", id="openai"),
]


def _api_key(name: str, key_env: str) -> str:
    """The key for the provider called ``name``, or a skip — unless required."""
    api_key = os.getenv(key_env)
    if api_key:
        return api_key

    required = {
        name.strip()
        for name in os.getenv("ONDINE_LIVE_REQUIRED", "").split(",")
        if name.strip()
    }
    if name in required:
        pytest.fail(
            f"{key_env} is not set, but ONDINE_LIVE_REQUIRED names {name!r}. "
            f"Add the {key_env} repository secret, or the live smoke test "
            f"proves nothing."
        )
    pytest.skip(f"{key_env} not set")


def _numbers(column: pd.Series) -> list[str]:
    """The last number in each answer.

    Models decorate: "9", "9.", "3 * 3 = 9". The result is whatever comes
    last, so restating the question does not turn a right answer wrong.
    """
    return [(re.findall(r"\d+", str(value)) or [""])[-1] for value in column]


@pytest.mark.parametrize(("provider", "model", "key_env"), PROVIDERS)
def test_one_prompt_per_row_answers_every_row(provider, model, key_env, request):
    api_key = _api_key(request.node.callspec.id, key_env)

    result = (
        PipelineBuilder.create()
        .from_dataframe(
            ARITHMETIC, input_columns=["question"], output_columns=["answer"]
        )
        .with_prompt("Compute {question}. Reply with the number only.")
        .with_llm(provider=provider, model=model, api_key=api_key, temperature=0.0)
        .build()
        .execute()
    )

    assert result.is_complete, result.error_summary()
    assert _numbers(result.to_pandas()["answer"]) == EXPECTED
    assert result.costs.total_tokens > 0


@pytest.mark.parametrize(("provider", "model", "key_env"), PROVIDERS)
def test_a_batched_prompt_keeps_each_answer_on_its_own_row(
    provider, model, key_env, request
):
    api_key = _api_key(request.node.callspec.id, key_env)

    result = (
        PipelineBuilder.create()
        .from_dataframe(
            ARITHMETIC, input_columns=["question"], output_columns=["answer"]
        )
        .with_prompt("Compute {question}. Reply with the number only.")
        .with_llm(provider=provider, model=model, api_key=api_key, temperature=0.0)
        .with_batch_size(3)
        .build()
        .execute()
    )

    assert result.is_complete, result.error_summary()
    assert _numbers(result.to_pandas()["answer"]) == EXPECTED


class _Sum(BaseModel):
    answer: int


@pytest.mark.parametrize(("provider", "model", "key_env"), PROVIDERS)
def test_structured_output_parses_into_the_declared_type(
    provider, model, key_env, request
):
    api_key = _api_key(request.node.callspec.id, key_env)

    result = (
        PipelineBuilder.create()
        .from_dataframe(
            ARITHMETIC, input_columns=["question"], output_columns=["answer"]
        )
        .with_prompt("Compute {question}.")
        .with_llm(provider=provider, model=model, api_key=api_key, temperature=0.0)
        .with_structured_output(_Sum)
        .build()
        .execute()
    )

    assert result.is_complete, result.error_summary()
    assert _numbers(result.to_pandas()["answer"]) == EXPECTED
