"""The models the live integration tests call, named in one place.

Hosted models are retired without notice. Groq dropped every Llama model it
served, and twenty-eight tests across fifteen files went red at once because
each of them spelled ``llama-3.3-70b-versatile`` out by hand. A retired model
should cost one edit here, not a sweep of the suite.

Two models are named:

- ``FREE_*`` — a zero-cost OpenRouter model, for every test that only needs
  "some real model answering" (batching, rate limiting, routing, YAML, CLI).
  It is reached through the ``litellm`` provider, which resolves the
  ``openrouter/`` prefix and reads ``OPENROUTER_API_KEY`` on its own.
- ``GROQ_MODEL`` — a model Groq still serves, for the few tests whose subject
  is the ``groq`` provider path itself.

A free model reports a cost of exactly zero, so tests running on it assert on
tokens, never on ``total_cost > 0``. Anything that checks priced cost
accounting belongs on a paid model.

OpenRouter caps free models at 50 requests a day for an account with no
purchased credits (1000 once it holds 10). The full live suite needs more than
50, so on an uncredited key it runs out partway and the remaining tests fail
with ``free-models-per-day``. Every name below can be overridden from the
environment — to finish a run on another host, or when a model disappears and
the suite has to run before this file is updated:

    ONDINE_TEST_FREE_MODEL=groq/openai/gpt-oss-20b \
        ONDINE_TEST_FREE_KEY_ENV=GROQ_API_KEY pytest tests/integration

Keep ``FREE_MODEL`` a full LiteLLM id (route prefix included): the router
tests hand it to LiteLLM directly, with no provider beside it.
"""

import os

FREE_PROVIDER = os.getenv("ONDINE_TEST_FREE_PROVIDER", "litellm")
FREE_MODEL = os.getenv(
    "ONDINE_TEST_FREE_MODEL", "openrouter/nvidia/nemotron-3-super-120b-a12b:free"
)
FREE_KEY_ENV = os.getenv("ONDINE_TEST_FREE_KEY_ENV", "OPENROUTER_API_KEY")

#: The ``(provider, model, api_key_env)`` triple the parametrized tests take.
FREE_LLM = (FREE_PROVIDER, FREE_MODEL, FREE_KEY_ENV)

GROQ_MODEL = os.getenv("ONDINE_TEST_GROQ_MODEL", "openai/gpt-oss-20b")
