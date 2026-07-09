"""Tests for LiveBackend — the asyncio engine behind the submit/poll/collect
protocol (degenerate lifecycle).

Every test pins a concrete regression. Contract under test:

* ``LiveBackend`` implements the ``ExecutionBackend`` protocol
  (runtime-checkable ``@runtime_checkable``), so it can stand in as the
  pipeline's middle stage alongside ``ProviderBatchBackend``.
* ``submit()`` runs the concurrent asyncio engine **synchronously** and
  returns a ``live-*`` job id whose results are cached in-process.
  This is the deliberate degenerate-lifecycle trade-off documented in
  the protocol docstring: live backends block in submit, batch backends
  do not.
* ``poll()`` always reports ``is_terminal=True`` for a live job id —
  the work is already done, there is nothing to wait for.
* ``collect()`` yields the cached ``LLMResponse`` objects (flattened
  from the engine's ``ResponseBatch`` output), one per original row.
* The CLI ``collect`` command refuses ``live-*`` job ids: live results
  live in memory and cannot survive into a fresh collect process.

The LLM client is mocked at the architectural boundary
(``create_llm_client``) so no network call is made and no real API key
is required. The engine wiring (rate limiter, retry handler, observer
lifecycle) is exercised against real code paths.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ondine.core.models import LLMResponse, PromptBatch, RowMetadata
from ondine.core.specifications import (
    DatasetSpec,
    LLMSpec,
    PipelineSpecifications,
    PromptSpec,
)


# ── protocol conformance ─────────────────────────────────────────────


def test_live_backend_implements_execution_backend_protocol() -> None:
    """Regression: LiveBackend MUST be recognised as an ExecutionBackend
    by runtime-checkable isinstance. If the protocol were not satisfied,
    the pipeline could not treat live and batch backends as
    interchangeable, defeating the whole point of the abstraction."""
    from ondine.orchestration.backends.base import ExecutionBackend
    from ondine.orchestration.backends.live import LiveBackend

    backend = LiveBackend(llm_spec=_openai_spec(), specs=_specs(), context=_context())
    assert isinstance(backend, ExecutionBackend)


def test_live_backend_exposes_submit_poll_collect() -> None:
    """Regression: the job-lifecycle surface (submit/poll/collect) is the
    contract the CLI submit/status/collect commands share across live
    and batch modes. If any method is missing or renamed, the CLI's
    single plumbing path breaks for live jobs."""
    from ondine.orchestration.backends.live import LiveBackend

    backend = LiveBackend(llm_spec=_openai_spec(), specs=_specs(), context=_context())
    for method in ("submit", "poll", "collect"):
        assert callable(getattr(backend, method, None)), (
            f"LiveBackend missing job-lifecycle method: {method}"
        )


def test_live_backend_exposes_llm_spec_property() -> None:
    """Regression: the protocol requires an ``llm_spec`` property so the
    pipeline/CLI can read which provider a backend was built with
    without coupling to the concrete class. If it disappears, every
    backend-specific code path must grow an isinstance check."""
    from ondine.orchestration.backends.live import LiveBackend

    spec = _openai_spec()
    backend = LiveBackend(llm_spec=spec, specs=_specs(), context=_context())
    assert backend.llm_spec is spec


# ── degenerate lifecycle: submit blocks, poll terminal, collect cached ──


def test_submit_returns_live_prefixed_job_id() -> None:
    """Regression: submit() must return an id with the ``live-`` prefix
    so the CLI collect command can distinguish in-process live jobs
    from cross-process batch jobs and refuse to collect the former in a
    fresh process (their results live in memory and are gone)."""
    from ondine.orchestration.backends.live import LiveBackend

    backend = LiveBackend(llm_spec=_openai_spec(), specs=_specs(), context=_context())
    with _mock_engine(responses=_two_llm_responses()):
        job_id = backend.submit(_two_prompt_batches())

    assert job_id.startswith("live-"), (
        f"live job id must be 'live-'-prefixed, got {job_id!r}"
    )


def test_submit_runs_engine_synchronously_and_caches_results() -> None:
    """Regression: the degenerate lifecycle contract — submit() runs the
    engine to completion NOW and caches the flattened results keyed by
    job id, so a subsequent collect() in the same process returns them.
    If submit did not block, collect would find an empty cache and the
    CLI's live path would silently return zero results."""
    from ondine.orchestration.backends.live import LiveBackend

    expected = _two_llm_responses()
    backend = LiveBackend(llm_spec=_openai_spec(), specs=_specs(), context=_context())
    with _mock_engine(responses=expected) as mock_stage:
        job_id = backend.submit(_two_prompt_batches())

    # The engine must have actually executed (not been deferred).
    assert mock_stage.execute.called, "submit() must run the engine synchronously"
    # collect() must yield the same number of results without re-running.
    collected = list(backend.collect(job_id))
    assert len(collected) == len(expected)
    assert [r.text for r in collected] == [r.text for r in expected]


def test_poll_always_reports_terminal() -> None:
    """Regression: a live job is already complete by the time submit()
    returns, so poll() must report is_terminal=True immediately. If it
    reported in-flight, a poll loop would spin forever waiting for a
    job that can never progress."""
    from ondine.orchestration.backends.base import BatchProgress
    from ondine.orchestration.backends.live import LiveBackend

    backend = LiveBackend(llm_spec=_openai_spec(), specs=_specs(), context=_context())
    with _mock_engine(responses=_two_llm_responses()):
        job_id = backend.submit(_two_prompt_batches())

    progress = backend.poll(job_id)

    assert isinstance(progress, BatchProgress)
    assert progress.is_terminal is True
    assert progress.status == "completed"
    assert progress.completed == 2
    assert progress.failed == 0


def test_collect_yields_one_llmresponse_per_row() -> None:
    """Regression: collect() must yield flat LLMResponse objects (one per
    original prompt row), NOT ResponseBatch objects. The CLI collect
    path writes LLMResponse.text to CSV; handing it a ResponseBatch
    would either crash or write nothing."""
    from ondine.orchestration.backends.live import LiveBackend

    backend = LiveBackend(llm_spec=_openai_spec(), specs=_specs(), context=_context())
    with _mock_engine(responses=_two_llm_responses()):
        job_id = backend.submit(_two_prompt_batches())

    for r in backend.collect(job_id):
        assert isinstance(r, LLMResponse), (
            "collect() must yield LLMResponse, not ResponseBatch"
        )


def test_submit_empty_batches_returns_job_with_no_results() -> None:
    """Regression: an empty prompt list is a valid degenerate input (the
    pipeline front half produced no batches). submit() must still
    return a live-* id and collect() must yield nothing, not raise."""
    from ondine.orchestration.backends.live import LiveBackend

    backend = LiveBackend(llm_spec=_openai_spec(), specs=_specs(), context=_context())
    job_id = backend.submit([])

    assert job_id.startswith("live-")
    assert list(backend.collect(job_id)) == []


def test_collect_for_unknown_job_id_yields_nothing() -> None:
    """Regression: collect() must not raise on a job id it has never
    seen (defensive — a caller may hold a stale id). It yields nothing
    rather than KeyError-ing the whole CLI."""
    from ondine.orchestration.backends.live import LiveBackend

    backend = LiveBackend(llm_spec=_openai_spec(), specs=_specs(), context=_context())
    assert list(backend.collect("live-999-nonexistent")) == []


# ── flattening: ResponseBatch (str | LLMResponse) → LLMResponse ───────


def test_flatten_handles_raw_string_responses() -> None:
    """Regression: the engine may emit ResponseBatch.responses as raw
    strings (e.g. structured-output-disabled path). collect() must
    synthesise LLMResponse objects from them rather than crashing on a
    type mismatch with the CLI's CSV writer."""
    from ondine.orchestration.backends.live import LiveBackend

    backend = LiveBackend(llm_spec=_openai_spec(), specs=_specs(), context=_context())
    with _mock_engine(responses=["plain text one", "plain text two"]):
        job_id = backend.submit(_two_prompt_batches())

    collected = list(backend.collect(job_id))
    assert len(collected) == 2
    for r in collected:
        assert isinstance(r, LLMResponse)
        assert isinstance(r.text, str)


def test_submit_preserves_response_order() -> None:
    """Regression: collect() must yield responses in submit-order so the
    back half (Disaggregate → Parse → Write) can correlate each response
    with its original row. If the flattening re-ordered, row 0's answer
    would land on row 1."""
    from ondine.orchestration.backends.live import LiveBackend

    ordered = [
        LLMResponse(text="first", tokens_in=1, tokens_out=1, model="m", cost=Decimal("0"), latency_ms=1.0),
        LLMResponse(text="second", tokens_in=2, tokens_out=2, model="m", cost=Decimal("0"), latency_ms=2.0),
    ]
    backend = LiveBackend(llm_spec=_openai_spec(), specs=_specs(), context=_context())
    with _mock_engine(responses=ordered):
        job_id = backend.submit(_two_prompt_batches())

    collected = list(backend.collect(job_id))
    assert [r.text for r in collected] == ["first", "second"]


def test_each_submit_returns_distinct_job_id() -> None:
    """Regression: multiple submits in one process must not collide on a
    shared job id, or collect() would return the first run's results
    for every subsequent run."""
    from ondine.orchestration.backends.live import LiveBackend

    backend = LiveBackend(llm_spec=_openai_spec(), specs=_specs(), context=_context())
    with _mock_engine(responses=_two_llm_responses()):
        job_a = backend.submit(_two_prompt_batches())
        job_b = backend.submit(_two_prompt_batches())

    assert job_a != job_b
    assert len(list(backend.collect(job_a))) == 2
    assert len(list(backend.collect(job_b))) == 2


# ── CLI guard: collect refuses live-* job ids ─────────────────────────


def test_cli_collect_refuses_live_job_id(capsys: pytest.CaptureFixture) -> None:
    """Regression: live results live in the submitting process's memory
    and cannot be rebuilt in a fresh collect process (unlike batch jobs,
    which are reconstructed from the spec snapshot). If the CLI did not
    refuse live-* ids, ``ondine collect`` would rebuild an empty
    ProviderBatchBackend, call ``collect("live-1")`` on it, get nothing
    back, and silently report "Collected 0 responses" — confusing the
    user who did get results synchronously during the live run."""
    from ondine.cli.main import cli

    # Build a fake registry entry whose provider_job_id is a live-* id.
    fake_handle = MagicMock()
    fake_handle.provider_job_id = "live-1"
    fake_handle.status.value = "succeeded"
    fake_handle.run_id = MagicMock()
    fake_handle.spec_snapshot = {}

    with patch("ondine.orchestration.RunRegistry") as mock_registry_cls:
        mock_registry_cls.return_value.get.return_value = fake_handle
        with pytest.raises(SystemExit) as exc_info:
            cli.main(
                ["collect", "00000000-0000-0000-0000-000000000001"],
                standalone_mode=False,
            )

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    out = captured.out.lower()
    # Must be an explicit refusal naming the live job, not a silent
    # "0 responses" success or an unrelated API-key error.
    assert "live" in out, (
        "CLI collect must explain why a live-* job id cannot be collected"
    )
    assert "cannot be collected" in out or "not a batch" in out, (
        "CLI collect must clearly state the live job cannot be collected cross-process"
    )


# ── helpers ───────────────────────────────────────────────────────────


def _openai_spec() -> LLMSpec:
    return LLMSpec(
        model="openai/gpt-4o-mini",
        api_key="sk-test",  # pragma: allowlist secret
        temperature=0.0,
    )


def _specs() -> PipelineSpecifications:
    from ondine.core.specifications import DataSourceType

    return PipelineSpecifications(
        dataset=DatasetSpec(
            source_type=DataSourceType.DATAFRAME,
            input_columns=["text"],
            output_columns=["r"],
        ),
        prompt=PromptSpec(template="P: {text}"),
        llm=_openai_spec(),
    )


def _context() -> Any:
    """A minimal ExecutionContext with the fields the live engine reads."""
    from ondine.orchestration.execution_context import ExecutionContext

    return ExecutionContext()


def _two_prompt_batches() -> list[PromptBatch]:
    return [
        PromptBatch(
            prompts=["Classify: alpha", "Classify: beta"],
            metadata=[RowMetadata(row_index=0), RowMetadata(row_index=1)],
            batch_id=0,
        ),
    ]


def _two_llm_responses() -> list[LLMResponse]:
    return [
        LLMResponse(text="A", tokens_in=3, tokens_out=1, model="gpt-4o-mini", cost=Decimal("0"), latency_ms=10.0),
        LLMResponse(text="B", tokens_in=3, tokens_out=1, model="gpt-4o-mini", cost=Decimal("0"), latency_ms=12.0),
    ]


class _FakeStage:
    """Stand-in for LLMInvocationStage that returns canned responses.

    The real stage is built by the live backend; we only intercept
    ``execute()`` so we control the engine's output without a network
    call. ``called`` lets tests assert submit() actually ran the engine.
    """

    def __init__(self, responses: list[Any]) -> None:
        from ondine.core.models import ResponseBatch

        self._batch = ResponseBatch(
            responses=responses,
            metadata=[RowMetadata(row_index=i) for i in range(len(responses))],
        )
        self.execute = MagicMock(return_value=[self._batch])


class _mock_engine:
    """Patch create_llm_client + LLMInvocationStage so the live backend
    builds a fake stage returning ``responses``."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses
        self._patches: list[Any] = []

    def __enter__(self) -> _FakeStage:
        fake_stage = _FakeStage(self._responses)
        self._patches = [
            patch(
                "ondine.orchestration.backends.live.create_llm_client",
                return_value=MagicMock(),
            ),
            patch(
                "ondine.orchestration.backends.live.LLMInvocationStage",
                return_value=fake_stage,
            ),
        ]
        for p in self._patches:
            p.start()
        return fake_stage

    def __exit__(self, *args: Any) -> None:
        for p in self._patches:
            p.stop()
