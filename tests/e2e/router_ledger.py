"""Failover conformance below the client, at LiteLLM's deployment layer.

`LedgerClient` (see conformance_harness) replaces the whole `LLMClient`, so it
never exercises the Router: failover, load balancing and cooldown all live
*inside* LiteLLM, under `UnifiedLiteLLMClient`. To put a ledger there, the fake
has to be a LiteLLM deployment, not an Ondine client.

That is what this module provides. Each deployment is a `litellm.CustomLLM`
that answers a prompt from the prompt's own row markers — the same trick the
client-level ledger uses, so a reordered or dropped row is a value mismatch
rather than an invisible one. Deployments record which rows they answered, so a
test can assert *which* deployment served each row: the essence of failover.

The deployments speak both prompt dialects (one row per call, and the JSON
mega-batch of `with_batch_size()`) by reusing the client ledger's parser, so a
single fake covers batched and unbatched routing.

Registering a CustomLLM mutates three process-global LiteLLM lists. The tests
must leave them exactly as found — a leaked fake provider would change how an
unrelated test resolves a model — so registration is a context manager that
snapshots and restores all three.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections import Counter
from typing import Any

import litellm
from litellm import CustomLLM
from litellm.types.utils import Usage

from tests.e2e.conformance_harness import (
    _batch_items,
    _token_in,
    answer_for,
)

# Fixed per-call usage so token accounting is predictable if a test wants it.
_TOKENS_IN = 8
_TOKENS_OUT = 4


def _answer_for_prompt(prompt: str) -> str:
    """The completion text for a prompt, in whichever dialect it arrived.

    A batched prompt earns a JSON array keyed by the batch ids; a plain prompt
    earns the single answer for its marker. Identical mapping to the
    client-level ledger, so both ledgers agree on what a correct answer is.
    """
    items = _batch_items(prompt)
    if items is not None:
        return json.dumps(
            [
                {"id": item["id"], "result": answer_for(_token_in(str(item["input"])))}
                for item in items
            ]
        )
    return answer_for(_token_in(prompt))


class LedgerDeployment(CustomLLM):
    """A healthy deployment that answers from the prompt and records its work.

    Args:
        deployment_id: The value reported back as the answering deployment, so a
            test can tell which deployment served a row.
        fail_tokens: Row markers this deployment refuses. A prompt carrying one
            raises a connection error — the deployment is up, but this row will
            never succeed here. A batched prompt fails whole if *any* of its
            rows is refused, which is how a batch actually behaves when one of
            its members is poison.

    Thread-safe: the async executor drives many concurrent calls, and a ledger
    that dropped entries under load would fake the very omission it exists to
    detect.
    """

    def __init__(
        self, deployment_id: str, *, fail_tokens: set[str] | None = None
    ) -> None:
        self.deployment_id = deployment_id
        self._fail_tokens = set(fail_tokens or ())
        self._lock = threading.Lock()
        self.answered: Counter[str] = Counter()

    @property
    def answered_tokens(self) -> set[str]:
        """Every row marker this deployment answered at least once."""
        return set(self.answered)

    def _record(self, prompt: str) -> None:
        items = _batch_items(prompt)
        tokens = (
            [_token_in(str(item["input"])) for item in items]
            if items is not None
            else [_token_in(prompt)]
        )
        refused = [token for token in tokens if token in self._fail_tokens]
        if refused:
            raise litellm.APIConnectionError(
                message=f"{self.deployment_id} refuses {sorted(refused)}",
                llm_provider="ledger",
                model=self.deployment_id,
            )
        with self._lock:
            self.answered.update(tokens)

    def _build(self, model: str, messages: list, model_response: Any) -> Any:
        prompt = messages[-1]["content"]
        self._record(prompt)
        model_response.choices[0].message.content = _answer_for_prompt(prompt)
        model_response.model = model
        model_response.usage = Usage(
            prompt_tokens=_TOKENS_IN,
            completion_tokens=_TOKENS_OUT,
            total_tokens=_TOKENS_IN + _TOKENS_OUT,
        )
        model_response._hidden_params = {"model_id": self.deployment_id}
        return model_response

    def completion(self, *args: Any, **kwargs: Any) -> Any:
        return self._build(
            kwargs["model"], kwargs["messages"], kwargs["model_response"]
        )

    async def acompletion(self, *args: Any, **kwargs: Any) -> Any:
        return self._build(
            kwargs["model"], kwargs["messages"], kwargs["model_response"]
        )


class AlwaysDownDeployment(CustomLLM):
    """A deployment that is entirely unreachable.

    Every call raises the transient connection error the Router treats as
    grounds for failover, so this stands in for a provider that is down for the
    whole run. Counts attempts only to prove the Router did try it.
    """

    def __init__(self, deployment_id: str) -> None:
        self.deployment_id = deployment_id
        self._lock = threading.Lock()
        self.attempts = 0

    def _boom(self, model: str) -> None:
        with self._lock:
            self.attempts += 1
        raise litellm.APIConnectionError(
            message=f"{self.deployment_id} is down",
            llm_provider="ledger",
            model=model,
        )

    def completion(self, *args: Any, **kwargs: Any) -> Any:
        self._boom(kwargs["model"])

    async def acompletion(self, *args: Any, **kwargs: Any) -> Any:
        self._boom(kwargs["model"])


def deployment_entry(provider: str, deployment_id: str, **litellm_params: Any) -> dict:
    """One `model_list` entry pointing the shared group at a fake deployment.

    All entries in a group must share `model_name` for the Router to load
    balance across them; `deployment_id` becomes the LiteLLM `model` suffix and
    doubles as the provider-map key, so each fake is addressable on its own.
    """
    return {
        "model_name": "svc",
        "litellm_params": {"model": f"{provider}/{deployment_id}", **litellm_params},
    }


@contextlib.contextmanager
def registered_deployments(mapping: dict[str, CustomLLM]):
    """Register fake providers for the block, then restore LiteLLM's globals.

    `mapping` is provider-name → handler. Registration appends to three
    module-global lists that LiteLLM never expects to shrink; we snapshot and
    restore all three so no fake leaks into another test's model resolution.
    """
    from litellm.utils import custom_llm_setup

    saved_map = list(litellm.custom_provider_map)
    saved_custom = list(litellm._custom_providers)
    saved_providers = list(litellm.provider_list)
    try:
        litellm.custom_provider_map = [
            {"provider": name, "custom_handler": handler}
            for name, handler in mapping.items()
        ]
        custom_llm_setup()
        yield
    finally:
        litellm.custom_provider_map = saved_map
        litellm._custom_providers = saved_custom
        litellm.provider_list = saved_providers
