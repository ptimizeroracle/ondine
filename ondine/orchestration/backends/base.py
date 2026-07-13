"""Execution backend protocol — the pluggable middle of the pipeline.

Architecture (see /plans/ARCHITECTURE_PROPOSAL.md §3):

* The front half (Load → Format → Aggregate) and the back half
  (Disaggregate → Parse → Write) are SHARED across execution modes.
* Only the middle — the LLM call layer — swaps. ``LiveBackend`` drives
  the existing asyncio engine (request-per-call); ``ProviderBatchBackend``
  compiles a JSONL, submits a provider-native Batch job, and collects
  results later (see :mod:`ondine.orchestration.backends.provider_batch`).

This module defines the contract both backends satisfy. The interface is
**job-lifecycle shaped** (submit → poll → collect), NOT request-shaped,
because the whole point of provider batch mode is that submit and collect
are separated in time — a request-stream interface would force the batch
backend to block inside ``invoke()``, destroying the non-blocking UX the
RunRegistry and CLI depend on.

Design (deep module, information hiding):

* ``ExecutionBackend`` exposes the minimum a pipeline needs to route a
  run: ``submit`` to start, ``poll`` for progress, ``collect`` for
  results. Every provider-specific detail (JSONL encoding, batch API
  endpoints, file upload/download, response envelope decoding) lives
  behind the concrete backend and never reaches the caller.
* ``BatchProgress`` is a provider-agnostic snapshot: total / completed /
  failed counts and a terminal flag. The registry and CLI read this one
  shape regardless of whether the job runs on OpenAI or Anthropic, so
  adding a third provider later touches exactly one module.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ondine.core.models import LLMResponse, PromptBatch
    from ondine.core.specifications import LLMSpec


@dataclass(frozen=True)
class BatchProgress:
    """Provider-agnostic snapshot of a batch job's current state.

    ``total`` / ``completed`` / ``failed`` are request counts (one
    request per prompt row). ``is_terminal`` lets a poller stop without
    re-deriving the provider's own status vocabulary (OpenAI uses
    ``completed``/``failed``/``expired``; Anthropic uses
    ``ended`` — the caller should not have to know).
    """

    provider_job_id: str
    status: str
    total: int = 0
    completed: int = 0
    failed: int = 0
    cost_so_far: Decimal = Decimal("0")
    is_terminal: bool = False


@runtime_checkable
class ExecutionBackend(Protocol):
    """Pluggable middle of the pipeline: how prompts become responses.

    A backend owns ONE concern — turning a list of :class:`PromptBatch`
    into a stream of :class:`LLMResponse` — and hides every detail of
    the transport (live HTTP vs. provider batch job) behind three
    methods. The pipeline's front and back halves are identical for
    every backend; only the middle swaps.

    Lifecycle:

    * ``submit(prompts)`` — non-blocking. Compiles and uploads the
      request payload, kicks off the job, returns a ``provider_job_id``
      the caller persists (in :class:`RunHandle`) for later polling.
    * ``poll(provider_job_id)`` — read-only progress snapshot.
    * ``collect(provider_job_id)`` — terminal only. Downloads and
      decodes the results, yielding one :class:`LLMResponse` per
      original request, ordered to match the submit order.

    The protocol is ``@runtime_checkable`` so the pipeline can assert
    conformance at build time (a misconfigured backend is caught
    immediately rather than failing mid-run).
    """

    @property
    def llm_spec(self) -> LLMSpec:
        """The LLM configuration this backend was built with."""
        ...

    def submit(self, prompts: list[PromptBatch]) -> str:
        """Compile + upload + start the job; return provider_job_id.

        Must NOT block on results — the caller relies on the immediate
        id to persist in the RunRegistry and return to the user.
        """
        ...

    def poll(self, provider_job_id: str) -> BatchProgress:
        """Return a provider-agnostic progress snapshot."""
        ...

    def collect(self, provider_job_id: str) -> Iterator[LLMResponse]:
        """Download finished results and yield decoded LLMResponses.

        Only valid once ``poll(...)`` reports ``is_terminal``; calling
        on an in-flight job raises ``ValueError``.
        """
        ...


__all__ = ["BatchProgress", "ExecutionBackend"]
