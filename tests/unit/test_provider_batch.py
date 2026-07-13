"""Tests for ProviderBatchBackend (§5) — OpenAI + Anthropic Batch API mode.

Every test pins a concrete regression. Contract under test:

* ``ProviderBatchBackend`` implements the ``ExecutionBackend`` protocol
  (runtime-checkable ``@runtime_checkable``) — so it can be swapped in as
  the pipeline's middle stage without changing the front/back halves.
* ``submit()`` compiles PromptBatches to a provider JSONL, uploads it,
  kicks off a batch job, and returns a ``provider_job_id`` **without
  blocking** on the results (the whole reason batch mode exists).
* ``poll()`` returns a ``BatchProgress`` snapshot (status + counts +
  partial cost) so the registry/CLI can show live progress.
* ``collect()`` downloads the finished results file and yields one
  ``LLMResponse`` per request, decoding provider-specific JSON into the
  ondine domain model — never leaking the raw provider envelope.
* The v1 scope guard rejects unsupported providers at ``build()``: only
  OpenAI and Anthropic may run in provider_batch mode. This is the single
  choke point so an unsupported provider can never reach the backend.

The OpenAI/Anthropic clients are mocked at the architectural boundary
(the SDK client objects) so no network call is made and no real API key
is required. Everything else (JSONL encoding, response decoding,
provider_job_id round-trip) is exercised against real code paths.
"""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from ondine.core.models import LLMResponse, PromptBatch, RowMetadata
from ondine.core.specifications import (
    LLMProvider,
    LLMSpec,
    ProcessingSpec,
)

# ── protocol conformance ─────────────────────────────────────────────


def test_provider_batch_implements_execution_backend_protocol() -> None:
    """Regression: ProviderBatchBackend MUST be recognised as an
    ExecutionBackend by runtime-checkable isinstance. If the protocol
    were not satisfied, the pipeline would silently fall back to the
    live backend and the user's provider_batch intent would be ignored
    without any error."""
    from ondine.orchestration.backends.base import ExecutionBackend
    from ondine.orchestration.backends.provider_batch import (
        ProviderBatchBackend,
    )

    backend = ProviderBatchBackend(
        llm_spec=_openai_spec(),
    )
    assert isinstance(backend, ExecutionBackend)


def test_provider_batch_exposes_submit_poll_collect() -> None:
    """Regression: the job-lifecycle surface (submit/poll/collect) is
    the contract the RunRegistry provider_job_id and the CLI
    submit/status/collect commands are built against. If any method is
    missing or renamed, the registry would store a provider_job_id that
    nothing can resolve."""
    from ondine.orchestration.backends.provider_batch import (
        ProviderBatchBackend,
    )

    backend = ProviderBatchBackend(llm_spec=_openai_spec())
    for method in ("submit", "poll", "collect"):
        assert callable(getattr(backend, method, None)), (
            f"ProviderBatchBackend missing job-lifecycle method: {method}"
        )


# ── scope guard (v1: OpenAI + Anthropic only) ─────────────────────────


def test_scope_guard_allows_openai() -> None:
    """Regression: OpenAI is a v1-supported provider for batch mode.
    Rejecting it would break the documented happy path."""
    from ondine.orchestration.backends.provider_batch import (
        SUPPORTED_BATCH_PROVIDERS,
    )

    assert LLMProvider.OPENAI in SUPPORTED_BATCH_PROVIDERS


def test_scope_guard_allows_anthropic() -> None:
    """Regression: Anthropic is a v1-supported provider for batch mode."""
    from ondine.orchestration.backends.provider_batch import (
        SUPPORTED_BATCH_PROVIDERS,
    )

    assert LLMProvider.ANTHROPIC in SUPPORTED_BATCH_PROVIDERS


def test_scope_guard_rejects_unsupported_provider_in_builder() -> None:
    """Regression: a user who selects provider_batch with an unsupported
    provider (e.g. Groq) would otherwise get a silent, confusing
    failure deep in the batch submit path. The guard at build() is the
    single choke point — catching it early with a clear message."""
    from ondine.api.pipeline_builder import PipelineBuilder

    with pytest.raises(ValueError, match="provider_batch.*groq"):
        (
            PipelineBuilder.create()
            .from_csv("data.csv", input_columns=["text"], output_columns=["r"])
            .with_prompt("P: {text}")
            .with_llm(provider="groq", model="llama-3.3-70b-versatile")
            .with_execution_mode("provider_batch")
            .build()
        )


def test_scope_guard_in_backend_constructor_for_groq() -> None:
    """Regression: even if a caller constructs the backend directly
    (bypassing the builder), an unsupported provider must still raise —
    defence in depth so a programmatic misuse cannot slip through."""
    from ondine.orchestration.backends.provider_batch import (
        ProviderBatchBackend,
    )

    groq_spec = LLMSpec(provider=LLMProvider.GROQ, model="llama-3.3-70b-versatile")
    with pytest.raises(ValueError, match="provider_batch"):
        ProviderBatchBackend(llm_spec=groq_spec)


# ── submit/poll/collect flow — OpenAI (mocked SDK) ────────────────────


def test_openai_submit_compiles_jsonl_and_returns_job_id() -> None:
    """Regression: submit() must (a) write one JSONL line per prompt,
    (b) hand the file to the OpenAI Batch API, and (c) return the
    provider's batch id immediately. If it blocked on results, the
    entire non-blocking batch UX (ondine submit → later ondine collect)
    collapses back into a synchronous call."""
    from ondine.orchestration.backends.provider_batch import (
        ProviderBatchBackend,
    )

    prompts = _two_prompt_batches()
    mock_client = MagicMock()
    mock_batches = MagicMock()
    mock_batches.id = "batch_abc123"
    mock_batches.status = "validating"
    mock_client.batches.create.return_value = mock_batches
    mock_files = MagicMock()
    mock_files.id = "file_xyz"
    mock_files.status = "processed"
    mock_client.files.create.return_value = mock_files
    mock_client.files.content.return_value = MagicMock(content=b"{}")

    backend = ProviderBatchBackend(llm_spec=_openai_spec(), client=mock_client)

    job_id = backend.submit(prompts)

    assert job_id == "batch_abc123"
    # One upload + one batch create call
    mock_client.files.create.assert_called_once()
    mock_client.batches.create.assert_called_once()
    # The uploaded file content must be valid JSONL with one line per prompt
    uploaded_bytes = mock_client.files.create.call_args.kwargs["file"][1]
    lines = [ln for ln in uploaded_bytes.read().decode().splitlines() if ln.strip()]
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert record["custom_id"]
        assert record["method"] == "POST"
        assert record["url"] == "/v1/chat/completions"


def test_openai_poll_returns_progress_snapshot() -> None:
    """Regression: poll() translates the provider's job status into a
    canonical BatchProgress so the registry/CLI never see provider-
    specific field names. If the mapping leaked, changing providers
    would break every downstream consumer."""
    from ondine.orchestration.backends.provider_batch import (
        BatchProgress,
        ProviderBatchBackend,
    )

    mock_client = MagicMock()
    mock_batch = MagicMock()
    mock_batch.id = "batch_abc"
    mock_batch.status = "in_progress"
    mock_batch.request_counts = MagicMock(total=100, completed=40, failed=2)
    mock_client.batches.retrieve.return_value = mock_batch

    backend = ProviderBatchBackend(llm_spec=_openai_spec(), client=mock_client)
    progress = backend.poll("batch_abc")

    assert isinstance(progress, BatchProgress)
    assert progress.total == 100
    assert progress.completed == 40
    assert progress.failed == 2
    assert progress.is_terminal is False


def test_openai_collect_yields_llm_responses() -> None:
    """Regression: collect() must decode the provider result JSONL into
    ondine LLMResponse objects — one per request — keeping token/cost
    accounting consistent with the live path. Leaking the raw provider
    envelope would force every downstream stage to know two formats."""
    from ondine.orchestration.backends.provider_batch import (
        ProviderBatchBackend,
    )

    mock_client = MagicMock()
    mock_batch = MagicMock()
    mock_batch.id = "batch_abc"
    mock_batch.status = "completed"
    mock_batch.request_counts = MagicMock(total=2, completed=2, failed=0)
    mock_client.batches.retrieve.return_value = mock_batch

    result_jsonl = (
        json.dumps(
            {
                "id": "req_1",
                "custom_id": "row-0",
                "response": {
                    "status_code": 200,
                    "body": {
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 3,
                        },
                        "choices": [{"message": {"content": "positive"}}],
                        "model": "gpt-4o-mini",
                    },
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "req_2",
                "custom_id": "row-1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "usage": {
                            "prompt_tokens": 8,
                            "completion_tokens": 3,
                        },
                        "choices": [{"message": {"content": "negative"}}],
                        "model": "gpt-4o-mini",
                    },
                },
            }
        )
        + "\n"
    )
    mock_content = MagicMock()
    mock_content.content = result_jsonl.encode()
    mock_client.files.content.return_value = mock_content

    backend = ProviderBatchBackend(llm_spec=_openai_spec(), client=mock_client)
    responses = list(backend.collect("batch_abc"))

    assert len(responses) == 2
    assert all(isinstance(r, LLMResponse) for r in responses)
    assert responses[0].text == "positive"
    assert responses[0].tokens_in == 10
    assert responses[0].tokens_out == 3
    assert responses[1].text == "negative"


def test_openai_collect_refuses_non_terminal() -> None:
    """Regression: collect() on an in-flight job would return a partial
    file or raise a confusing provider error. The contract — matching
    RunRegistry's terminal-only collect — is to refuse cleanly."""
    from ondine.orchestration.backends.provider_batch import (
        ProviderBatchBackend,
    )

    mock_client = MagicMock()
    mock_batch = MagicMock()
    mock_batch.status = "in_progress"
    mock_batch.request_counts = MagicMock(total=2, completed=1, failed=0)
    mock_client.batches.retrieve.return_value = mock_batch

    backend = ProviderBatchBackend(llm_spec=_openai_spec(), client=mock_client)
    with pytest.raises(ValueError, match="not terminal"):
        list(backend.collect("batch_abc"))


# ── Anthropic variant — proves provider complexity is hidden ──────────


def test_anthropic_submit_uses_messages_batch_endpoint() -> None:
    """Regression: Anthropic's Batch API has a different request shape
    (``/v1/messages/batches`` with ``requests``) than OpenAI. The
    backend must translate once, internally, so callers see one
    submit() regardless of provider."""
    from ondine.orchestration.backends.provider_batch import (
        ProviderBatchBackend,
    )

    prompts = _two_prompt_batches()
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.id = "msgbatch_001"
    mock_client.messages.batches.create.return_value = mock_resp

    backend = ProviderBatchBackend(llm_spec=_anthropic_spec(), client=mock_client)
    job_id = backend.submit(prompts)

    assert job_id == "msgbatch_001"
    mock_client.messages.batches.create.assert_called_once()
    # The Anthropic batch wraps each prompt as a request object
    sent = mock_client.messages.batches.create.call_args
    assert "requests" in sent.kwargs


# ── spec wiring: execution_mode + builder ─────────────────────────────


def test_processing_spec_has_execution_mode_default_live() -> None:
    """Regression: adding the field must not change default behaviour.
    Existing pipelines (which never set it) must stay in live mode — a
    silent flip to batch would surprise every current user."""
    spec = ProcessingSpec()
    assert spec.execution_mode == "live"


def test_processing_spec_accepts_provider_batch() -> None:
    spec = ProcessingSpec(execution_mode="provider_batch")
    assert spec.execution_mode == "provider_batch"


def test_processing_spec_rejects_unknown_execution_mode() -> None:
    """Regression: a typo like 'batch' must fail fast at validation,
    not silently default to live (defeating the feature) or crash later
    in the backend."""
    with pytest.raises(Exception):  # pydantic ValidationError
        ProcessingSpec(execution_mode="batch")  # type: ignore[arg-type]


def test_builder_with_execution_mode_sets_spec() -> None:
    """Regression: the builder method must propagate the choice into
    the ProcessingSpec so the pipeline's build() scope guard and the
    backend selection both read the same source of truth."""
    from ondine.api.pipeline_builder import PipelineBuilder

    builder = (
        PipelineBuilder.create()
        .from_csv("data.csv", input_columns=["text"], output_columns=["r"])
        .with_prompt("P: {text}")
        .with_llm(provider="openai", model="gpt-4o-mini")
        .with_execution_mode("provider_batch")
    )
    assert builder._processing_spec.execution_mode == "provider_batch"


# ── helpers ───────────────────────────────────────────────────────────


def _openai_spec() -> LLMSpec:
    return LLMSpec(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        api_key="sk-test",  # pragma: allowlist secret
        temperature=0.0,
        input_cost_per_1k_tokens=Decimal("0.00015"),
        output_cost_per_1k_tokens=Decimal("0.0006"),
    )


def _anthropic_spec() -> LLMSpec:
    return LLMSpec(
        provider=LLMProvider.ANTHROPIC,
        model="claude-sonnet-4-20250514",
        api_key="sk-ant-test",  # pragma: allowlist secret
        max_tokens=8192,
        temperature=0.0,
        input_cost_per_1k_tokens=Decimal("0.003"),
        output_cost_per_1k_tokens=Decimal("0.015"),
    )


def _two_prompt_batches() -> list[PromptBatch]:
    return [
        PromptBatch(
            prompts=["Classify: alpha", "Classify: beta"],
            metadata=[
                RowMetadata(row_index=0),
                RowMetadata(row_index=1),
            ],
            batch_id=0,
        ),
    ]
