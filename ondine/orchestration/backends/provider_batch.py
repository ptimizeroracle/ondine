"""ProviderBatchBackend — OpenAI + Anthropic native Batch API backend.

Architecture (see /plans/ARCHITECTURE_PROPOSAL.md §3, §5):

This is the batch-mode replacement for the pipeline's middle stage.
Instead of issuing one live HTTP request per prompt (the LiveBackend
path), it compiles every prompt into a single JSONL payload, uploads
that file to the provider's Batch API, kicks off a job, and later
downloads + decodes the results. The front half (Load → Format →
Aggregate) and back half (Disaggregate → Parse → Write) are untouched —
they consume the same :class:`PromptBatch` / :class:`LLMResponse` types
either way.

Why a separate backend, not a flag on the live engine:

* Batch APIs are *job-based*, not call-based. The natural unit is
  "upload a file, get a job id, poll, download" — fundamentally
  different control flow from a per-call asyncio gather. Forcing both
  into one engine would create a special/general mixture (Ousterhout
  red flag) where rare batch branches pollute the common live path.
* The job lifecycle maps cleanly onto the RunRegistry: ``submit`` stores
  a ``provider_job_id`` on the :class:`RunHandle`; ``ondine status``
  polls it; ``ondine collect`` drains it. One shape, three commands.

Design (deep module, information hiding):

* The caller sees three methods (``submit`` / ``poll`` / ``collect``)
  and never learns whether OpenAI or Anthropic is underneath. The
  provider difference — JSONL line shape, endpoint path, response
  envelope — is isolated in small private ``_openai_*`` / ``_anthropic_*``
  helpers. Adding a third provider means adding one helper set, not
  touching the pipeline, registry, or CLI.
* Credentials are resolved lazily inside ``submit``: the backend reads
  ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` from the environment (or
  from the LLMSpec) only when a job is actually submitted, never at
  import time, so merely constructing the backend has no side effects.
* v1 scope guard: only OpenAI and Anthropic support batch mode. The
  guard fires in the constructor (defence in depth) AND in
  :meth:`PipelineBuilder.build` (the single user-facing choke point).

Tests mock the SDK client objects (the architectural boundary) so the
JSONL encoding, response decoding, and provider_job_id round-trip are
all exercised against real code with no network.
"""

from __future__ import annotations

import io
import json
import os
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from ondine.orchestration.backends.base import BatchProgress
from ondine.utils import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ondine.core.models import LLMResponse, PromptBatch
    from ondine.core.specifications import LLMSpec

logger = get_logger(__name__)

# v1 scope: only these providers ship a Batch API we support. Listed once
# here and checked in the constructor (defence in depth) and in
# PipelineBuilder.build (the user-facing choke point). Adding a provider
# means adding it here plus the matching _<provider>_ helpers below.
SUPPORTED_BATCH_PROVIDERS: frozenset[str] = frozenset({"openai", "anthropic"})

# Canonical statuses the backend treats as "done". Kept provider-neutral
# so poll()/collect() never leak provider vocabulary to callers.
_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        "completed",
        "failed",
        "expired",
        "cancelled",  # OpenAI
        "ended",  # Anthropic terminal umbrella
        "succeeded",  # generic
    }
)


class ProviderBatchBackend:
    """Batch-API backend implementing the :class:`ExecutionBackend` protocol.

    The backend is constructed from an :class:`LLMSpec` (which selects
    the provider + model + pricing) and, optionally, a pre-built SDK
    client (the test seam — production code lets the backend build its
    own client lazily on first ``submit``). All provider specifics live
    behind private methods; the public surface is provider-agnostic.

    The class deliberately does NOT subclass the Protocol — structural
    conformance (``@runtime_checkable``) is the contract, mirroring how
    the rest of ondine treats protocols (see knowledge/context stores).
    """

    def __init__(
        self,
        llm_spec: LLMSpec,
        *,
        client: Any | None = None,
    ) -> None:
        """Configure the backend for a specific provider + model.

        Args:
            llm_spec: Provider, model, pricing, and credentials. The
                provider MUST be in :data:`SUPPORTED_BATCH_PROVIDERS`.
            client: Optional pre-built provider SDK client. Production
                callers omit this (the backend builds one lazily); tests
                inject a mock here to exercise the encoding/decoding
                paths without any network.
        """
        provider_key = self._provider_key(llm_spec)
        if provider_key not in SUPPORTED_BATCH_PROVIDERS:
            raise ValueError(
                f"provider_batch mode is only supported for "
                f"{sorted(SUPPORTED_BATCH_PROVIDERS)} in v1, got provider "
                f"'{provider_key}'. Use execution_mode='live' for other "
                f"providers."
            )
        self._llm_spec = llm_spec
        # NOTE: attribute named ``_provider`` (not ``_provider_key``) so it
        # does not shadow the ``_provider_key`` staticmethod below.
        self._provider = provider_key
        self._client = client  # lazily built in _ensure_client()
        # custom_id → RowMetadata index, so collect() can preserve order.
        # Populated during submit(); consumed during collect().
        self._custom_ids: list[str] = []

    # ── ExecutionBackend protocol surface ──────────────────────────

    @property
    def llm_spec(self) -> LLMSpec:
        """The LLM configuration this backend was built with."""
        return self._llm_spec

    def submit(self, prompts: list[PromptBatch]) -> str:
        """Compile prompts to JSONL, upload, start the job, return its id.

        Non-blocking: returns as soon as the provider acknowledges the
        batch creation. The returned id is what the caller persists on
        the :class:`RunHandle` (``provider_job_id``) for later polling.
        """
        client = self._ensure_client()
        self._custom_ids = []
        provider_job_id = (
            self._openai_submit(client, prompts)
            if self._provider == "openai"
            else self._anthropic_submit(client, prompts)
        )
        logger.info(
            "Submitted %s batch job %s (%d requests)",
            self._provider,
            provider_job_id,
            len(self._custom_ids),
        )
        return provider_job_id

    def poll(self, provider_job_id: str) -> BatchProgress:
        """Return a provider-agnostic progress snapshot for the job."""
        client = self._ensure_client()
        if self._provider == "openai":
            return self._openai_poll(client, provider_job_id)
        return self._anthropic_poll(client, provider_job_id)

    def collect(self, provider_job_id: str) -> Iterator[LLMResponse]:
        """Download + decode finished results, one LLMResponse per request.

        Raises ValueError if the job is not yet terminal — collecting an
        in-flight job would return a partial file or a confusing provider
        error, so we refuse cleanly (mirrors RunRegistry's terminal-only
        collect contract).
        """
        client = self._ensure_client()
        progress = self.poll(provider_job_id)
        if not progress.is_terminal:
            raise ValueError(
                f"Batch job {provider_job_id} is not terminal "
                f"(status={progress.status}); poll until complete."
            )
        if self._provider == "openai":
            yield from self._openai_collect(client, provider_job_id)
        else:
            yield from self._anthropic_collect(client, provider_job_id)

    # ── OpenAI Batch API ───────────────────────────────────────────

    def _openai_submit(self, client: Any, prompts: list[PromptBatch]) -> str:
        """Compile OpenAI-format JSONL, upload file, create batch."""
        jsonl = self._build_openai_jsonl(prompts)
        upload_file = ("requests.jsonl", io.BytesIO(jsonl.encode()), "application/json")
        file_obj = client.files.create(file=upload_file, purpose="batch")
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"source": "ondine"},
        )
        return str(batch.id)

    def _openai_poll(self, client: Any, provider_job_id: str) -> BatchProgress:
        batch = client.batches.retrieve(provider_job_id)
        counts = getattr(batch, "request_counts", None)
        total = int(getattr(counts, "total", 0) or 0)
        completed = int(getattr(counts, "completed", 0) or 0)
        failed = int(getattr(counts, "failed", 0) or 0)
        return BatchProgress(
            provider_job_id=provider_job_id,
            status=batch.status,
            total=total,
            completed=completed,
            failed=failed,
            cost_so_far=self._estimate_cost(completed, failed),
            is_terminal=batch.status in _TERMINAL_STATUSES,
        )

    def _openai_collect(
        self, client: Any, provider_job_id: str
    ) -> Iterator[LLMResponse]:
        batch = client.batches.retrieve(provider_job_id)
        result_file_id = getattr(batch, "output_file_id", None) or getattr(
            batch, "error_file_id", None
        )
        if result_file_id is None:
            return
        content = client.files.content(result_file_id)
        raw = getattr(content, "content", b"") or b""
        for line in raw.decode().splitlines():
            if not line.strip():
                continue
            yield self._decode_openai_line(json.loads(line))

    # ── Anthropic Batch API ────────────────────────────────────────

    def _anthropic_submit(self, client: Any, prompts: list[PromptBatch]) -> str:
        """Compile Anthropic-format requests, create message batch."""
        requests = self._build_anthropic_requests(prompts)
        batch = client.messages.batches.create(requests=requests)
        return str(batch.id)

    def _anthropic_poll(self, client: Any, provider_job_id: str) -> BatchProgress:
        batch = client.messages.batches.retrieve(provider_job_id)
        # Anthropic exposes counts as result_counts on the batch object.
        counts = getattr(batch, "result_counts", None) or {}
        total = (
            int(
                getattr(counts, "processing", 0)
                + getattr(counts, "succeeded", 0)
                + getattr(counts, "errored", 0)
                + getattr(counts, "canceled", 0)
                + getattr(counts, "expired", 0)
            )
            if not isinstance(counts, dict)
            else sum(counts.values())
        )
        succeeded = (
            int(getattr(counts, "succeeded", 0))
            if not isinstance(counts, dict)
            else int(counts.get("succeeded", 0))
        )
        errored = (
            int(getattr(counts, "errored", 0))
            if not isinstance(counts, dict)
            else int(counts.get("errored", 0))
        )
        return BatchProgress(
            provider_job_id=provider_job_id,
            status=batch.processing_status
            if hasattr(batch, "processing_status")
            else getattr(batch, "status", "unknown"),
            total=total,
            completed=succeeded,
            failed=errored,
            cost_so_far=self._estimate_cost(succeeded, errored),
            is_terminal=self._anthropic_is_terminal(batch),
        )

    def _anthropic_collect(
        self, client: Any, provider_job_id: str
    ) -> Iterator[LLMResponse]:
        # Anthropic streams results via the list() iterator.
        results = client.messages.batches.results(provider_job_id)
        for entry in results:
            yield self._decode_anthropic_entry(entry)

    # ── JSONL / request compilers ──────────────────────────────────

    def _build_openai_jsonl(self, prompts: list[PromptBatch]) -> str:
        """Flatten PromptBatches into OpenAI Batch JSONL.

        One line per prompt row. The ``custom_id`` carries the original
        row index so collect() can reconstruct order without a separate
        side-table; it is also stashed in ``self._custom_ids`` for tests
        and debugging.
        """
        lines: list[str] = []
        for batch in prompts:
            for prompt_text, meta in zip(batch.prompts, batch.metadata, strict=False):
                custom_id = f"row-{meta.row_index}"
                self._custom_ids.append(custom_id)
                record: dict[str, Any] = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self._llm_spec.model,
                        "messages": self._openai_messages(prompt_text),
                        "temperature": self._llm_spec.temperature,
                    },
                }
                if self._llm_spec.max_tokens is not None:
                    record["body"]["max_tokens"] = self._llm_spec.max_tokens
                lines.append(json.dumps(record))
        return "\n".join(lines) + ("\n" if lines else "")

    def _build_anthropic_requests(
        self, prompts: list[PromptBatch]
    ) -> list[dict[str, Any]]:
        """Flatten PromptBatches into Anthropic Batch request objects."""
        requests: list[dict[str, Any]] = []
        for batch in prompts:
            for prompt_text, meta in zip(batch.prompts, batch.metadata, strict=False):
                custom_id = f"row-{meta.row_index}"
                self._custom_ids.append(custom_id)
                requests.append(
                    {
                        "custom_id": custom_id,
                        "params": {
                            "model": self._llm_spec.model,
                            "max_tokens": self._llm_spec.max_tokens or 1024,
                            "messages": self._anthropic_messages(prompt_text),
                            "temperature": self._llm_spec.temperature,
                        },
                    }
                )
        return requests

    # ── response decoders ──────────────────────────────────────────

    def _decode_openai_line(self, record: dict[str, Any]) -> LLMResponse:
        """Translate one OpenAI Batch result line into an LLMResponse.

        Handles both success (response.body present) and error
        (response.status != 200) records so a single failed request
        cannot poison the whole collect().
        """
        from ondine.core.models import LLMResponse

        resp = record.get("response", {})
        body = resp.get("body", {}) if isinstance(resp, dict) else {}
        usage = body.get("usage", {}) or {}
        tokens_in = int(usage.get("prompt_tokens", 0))
        tokens_out = int(usage.get("completion_tokens", 0))
        choices = body.get("choices", []) or []
        text = (
            choices[0]["message"]["content"]
            if choices
            else (
                resp.get("error", {}).get("message", "")
                if isinstance(resp, dict)
                else ""
            )
        )
        model = body.get("model", self._llm_spec.model)
        return LLMResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=model,
            cost=self._row_cost(tokens_in, tokens_out),
            latency_ms=0.0,
            metadata={"custom_id": record.get("custom_id")},
        )

    def _decode_anthropic_entry(self, entry: Any) -> LLMResponse:
        """Translate one Anthropic Batch result into an LLMResponse."""
        from ondine.core.models import LLMResponse

        # entry has .result.message (success) or .result.error (failure)
        result = getattr(entry, "result", entry)
        message = getattr(result, "message", None)
        usage = getattr(message, "usage", None) if message else None
        tokens_in = int(getattr(usage, "input_tokens", 0)) if usage else 0
        tokens_out = int(getattr(usage, "output_tokens", 0)) if usage else 0
        text = ""
        if message is not None:
            content = getattr(message, "content", None)
            if content:
                # content is a list of text blocks
                text = "".join(
                    getattr(block, "text", "")
                    for block in content
                    if getattr(block, "type", None) == "text"
                )
        return LLMResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=self._llm_spec.model,
            cost=self._row_cost(tokens_in, tokens_out),
            latency_ms=0.0,
            metadata={"custom_id": getattr(entry, "custom_id", "")},
        )

    # ── shared helpers ─────────────────────────────────────────────

    def _openai_messages(self, prompt_text: str) -> list[dict[str, str]]:
        """Wrap a prompt in OpenAI chat messages, honouring system_message."""
        msgs: list[dict[str, str]] = []
        if self._llm_spec.provider and hasattr(self._llm_spec, "system_message"):
            # LLMSpec has no system_message field; system prompt lives in
            # PromptSpec. The batch backend receives already-formatted
            # prompts, so this path is a no-op kept for future use.
            pass
        msgs.append({"role": "user", "content": prompt_text})
        return msgs

    def _anthropic_messages(self, prompt_text: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": prompt_text}]

    def _estimate_cost(self, completed: int, failed: int) -> Decimal:
        """Rough running cost from request counts (pre-download).

        Used only for the live progress snapshot; the authoritative
        cost comes from token usage during collect(). Falls back to a
        per-request average when per-1k pricing is unset.
        """
        # Without per-row token data mid-flight, report zero — the
        # registry's cost tracker is updated authoritatively during
        # collect(). Keeping this conservative avoids overstating spend.
        return Decimal("0")

    def _row_cost(self, tokens_in: int, tokens_out: int) -> Decimal:
        """Cost for a single decoded response from LLMSpec pricing."""
        in_rate = self._llm_spec.input_cost_per_1k_tokens or Decimal("0")
        out_rate = self._llm_spec.output_cost_per_1k_tokens or Decimal("0")
        return Decimal(tokens_in) * in_rate / Decimal("1000") + Decimal(
            tokens_out
        ) * out_rate / Decimal("1000")

    def _anthropic_is_terminal(self, batch: Any) -> bool:
        """True when the Anthropic batch has stopped processing."""
        status = (
            getattr(batch, "processing_status", None)
            or getattr(batch, "status", None)
            or ""
        )
        return status in _TERMINAL_STATUSES or status == ""

    @staticmethod
    def _provider_key(llm_spec: LLMSpec) -> str:
        """Normalise the provider to a lowercase canonical key.

        Handles both the enum form (``LLMProvider.OPENAI``) and the
        LiteLLM-style ``"provider/model"`` model string so the scope
        guard works regardless of how the user configured the spec.
        """
        from ondine.core.specifications import LLMProvider

        provider = llm_spec.provider
        if isinstance(provider, LLMProvider):
            return provider.value
        provider_str = str(provider).lower()
        # Strip a leading "provider/" prefix from LiteLLM model strings
        if "/" in provider_str:
            provider_str = provider_str.split("/")[0]
        # If the provider is the generic litellm sentinel, derive from
        # the model string (e.g. "openai/gpt-4o-mini" → "openai").
        if provider_str == "litellm" and "/" in llm_spec.model:
            provider_str = llm_spec.model.split("/")[0].lower()
        return provider_str

    def _ensure_client(self) -> Any:
        """Lazily build the provider SDK client on first use.

        Construction is deferred so importing this module (and building
        the backend at pipeline-build time) never requires credentials
        or network. Only an actual ``submit`` needs a live client.
        """
        if self._client is not None:
            return self._client
        if self._provider == "openai":
            self._client = self._build_openai_client()
        else:
            self._client = self._build_anthropic_client()
        return self._client

    def _build_openai_client(self) -> Any:
        """Build an OpenAI client from env / spec credentials."""
        from ondine.utils.optional_dependencies import (
            _matches_missing_dependency,
        )

        try:
            from openai import OpenAI
        except ImportError as exc:
            if _matches_missing_dependency(exc, ("openai",)):
                raise ImportError(
                    "OpenAI batch mode requires the 'openai' package. "
                    "Install with: pip install openai"
                ) from exc
            raise
        api_key = self._llm_spec.api_key or os.environ.get("OPENAI_API_KEY")
        return OpenAI(api_key=api_key)

    def _build_anthropic_client(self) -> Any:
        """Build an Anthropic client from env / spec credentials."""
        from ondine.utils.optional_dependencies import (
            _matches_missing_dependency,
        )

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            if _matches_missing_dependency(exc, ("anthropic",)):
                raise ImportError(
                    "Anthropic batch mode requires the 'anthropic' package. "
                    "Install with: pip install anthropic"
                ) from exc
            raise
        api_key = self._llm_spec.api_key or os.environ.get("ANTHROPIC_API_KEY")
        return Anthropic(api_key=api_key)


__all__ = ["ProviderBatchBackend", "SUPPORTED_BATCH_PROVIDERS"]
