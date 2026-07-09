"""Tests for RunRegistry — persistent job index for pipeline runs.

Every test pins a concrete regression. The contract under test:

* `create(spec)` returns a RunHandle with status PENDING, durable on disk.
* `get(run_id)` returns the up-to-date handle. `list(status?)` filters.
* `transition(run_id, status, **fields)` validates the transition, persists,
  and notifies RegistryObservers exactly once with (run_id, old, new).
* Pipelines that pass `run_id=` to `execute()` register and progress through
  PENDING → RUNNING → SUCCEEDED|FAILED|PARTIAL. Default `run_id=None` leaves
  behaviour unchanged (no registry artifact on disk).

The SUT is `RunRegistry` against a real on-disk SQLite DB in a temp dir —
the entire motivation is crash-safe durability, which mocks cannot prove.
"""

from __future__ import annotations

import os
import subprocess
import sys
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from ondine.core.specifications import (
    DatasetSpec,
    DataSourceType,
    LLMProvider,
    LLMSpec,
    PipelineSpecifications,
    PromptSpec,
)
from ondine.orchestration.run_registry import (
    RegistryObserver,
    RunRegistry,
    RunSpec,
    RunStatus,
)

# ── regression #1: create persists a PENDING run ──────────────────────


def test_create_returns_pending_handle(registry: RunRegistry) -> None:
    """Regression: if create did not persist, the MCP ``ondine_run`` tool
    would return a run_id that no later ``ondine_status`` could resolve —
    a lost job invisible to the user."""
    spec = RunSpec(pipeline_id=str(uuid4()), dataset="df")
    handle = registry.create(spec)

    assert isinstance(handle.run_id, UUID)
    assert handle.status is RunStatus.PENDING
    assert handle.spec_snapshot == {"pipeline_id": spec.pipeline_id, "dataset": "df"}


def test_get_returns_up_to_date_handle(registry: RunRegistry) -> None:
    """Regression: get() must read the current on-disk state, not a
    cached handle. Otherwise a second process polling status for a
    long-running job would forever see PENDING."""
    spec = RunSpec(pipeline_id=str(uuid4()))
    handle = registry.create(spec)

    registry.transition(handle.run_id, RunStatus.RUNNING)

    fresh = registry.get(handle.run_id)
    assert fresh.status is RunStatus.RUNNING
    assert fresh.run_id == handle.run_id


def test_get_unknown_run_id_returns_none(registry: RunRegistry) -> None:
    """Regression: callers (CLI, MCP) must distinguish "no such run"
    from "run exists". A KeyError forces callers into try/except noise;
    None is the clean sentinel."""
    assert registry.get(uuid4()) is None


# ── regression #2: durability across processes ────────────────────────


def test_create_survives_process_kill(tmp_path: Path) -> None:
    """Regression: the MCP server issues a run_id immediately, then a
    worker process picks it up. If the registry were not crash-durable
    across process boundaries, the worker could never resolve the run
    the server started. Mirror the ResponseCache crash test: hard
    ``os._exit`` proves the WAL commit survived."""
    db_path = tmp_path / "runs.db"
    spec = RunSpec(pipeline_id=str(uuid4()), dataset="df")
    spec_dict = spec.to_dict()

    worker = f"""
import os, sys
from ondine.orchestration.run_registry import RunRegistry, RunSpec
registry = RunRegistry({str(db_path)!r})
registry.create(RunSpec.from_dict({spec_dict!r}))
os._exit(9)
"""

    proc = subprocess.run(
        [sys.executable, "-c", worker],
        env={**os.environ, "PYTHONPATH": os.getcwd()},
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 9, (
        f"worker did not hard-exit: {proc.returncode}\\n{proc.stderr.decode()}"
    )

    registry = RunRegistry(db_path)
    runs = registry.list()
    assert len(runs) == 1
    assert runs[0].status is RunStatus.PENDING


def test_reopen_sees_previously_written_runs(tmp_path: Path) -> None:
    """Regression: status lookups happen in a new process (the CLI's
    ``ondine status <run_id>`` is a fresh invocation). Data must survive
    the close/reopen boundary."""
    db_path = tmp_path / "reopen.db"
    a = RunRegistry(db_path)
    spec = RunSpec(pipeline_id=str(uuid4()))
    handle = a.create(spec)
    a.close()

    b = RunRegistry(db_path)
    try:
        fresh = b.get(handle.run_id)
        assert fresh is not None
        assert fresh.status is RunStatus.PENDING
    finally:
        b.close()


# ── regression #3: transition validation & observer fan-out ─────────


def test_valid_transition_persists_and_returns_updated_handle(
    registry: RunRegistry,
) -> None:
    """Regression: transition() must both persist and return the new
    state. If it returned the stale handle, callers chaining off the
    return would race the on-disk write."""
    handle = registry.create(RunSpec(pipeline_id=str(uuid4())))

    updated = registry.transition(handle.run_id, RunStatus.RUNNING)

    assert updated.status is RunStatus.RUNNING
    assert registry.get(handle.run_id).status is RunStatus.RUNNING


def test_invalid_transition_raises_value_error(registry: RunRegistry) -> None:
    """Regression: PENDING → SUCCEEDED must be rejected — a run that
    never registered as RUNNING cannot be SUCCEEDED. Without this guard
    the registry would record impossible state histories that a later
    consumer (MCP /admin) would treat as valid."""
    handle = registry.create(RunSpec(pipeline_id=str(uuid4())))

    with pytest.raises(ValueError, match="transition"):
        registry.transition(handle.run_id, RunStatus.SUCCEEDED)


def test_transition_unknown_run_id_raises_key_error(registry: RunRegistry) -> None:
    """Regression: transitioning a non-existent run is a programmer error
    (the MCP tool only operates on known run_ids). A silent no-op would
    let a bug in the caller pass undetected."""
    with pytest.raises(KeyError):
        registry.transition(uuid4(), RunStatus.RUNNING)


def test_transition_invokes_observers_exactly_once(
    registry: RunRegistry,
) -> None:
    """Regression: observers drive the live progress feed in the MCP
    server. Firing twice duplicates every status event in the user's
    stream; firing zero times hides completion. Exactly once."""
    handle = registry.create(RunSpec(pipeline_id=str(uuid4())))
    seen: list[tuple[UUID, RunStatus, RunStatus]] = []

    class Recorder(RegistryObserver):
        def on_transition(self, run_id: UUID, old: RunStatus, new: RunStatus) -> None:
            seen.append((run_id, old, new))

    registry.add_observer(Recorder())

    registry.transition(handle.run_id, RunStatus.RUNNING)

    assert seen == [(handle.run_id, RunStatus.PENDING, RunStatus.RUNNING)]


def test_observer_exception_does_not_break_transition(
    registry: RunRegistry,
) -> None:
    """Regression: an observer (e.g. a metrics sink) must not be able to
    take down the registry — the transition must still persist even when
    a downstream listener raises. Matches the silent-observer-failure
    contract of ``ExecutionContext.notify_progress``."""
    handle = registry.create(RunSpec(pipeline_id=str(uuid4())))

    class Boom(RegistryObserver):
        def on_transition(self, run_id: UUID, old: RunStatus, new: RunStatus) -> None:
            raise RuntimeError("observer exploded")

    registry.add_observer(Boom())

    registry.transition(handle.run_id, RunStatus.RUNNING)

    assert registry.get(handle.run_id).status is RunStatus.RUNNING


# ── regression #4: metrics & provider_job_id fields ───────────────────


def test_transition_updates_metrics_and_provider_job_id(
    registry: RunRegistry,
) -> None:
    """Regression: the ProviderBatchBackend (§5) needs to persist the
    provider's batch_id once a job is submitted (SUBMITTED_REMOTE).
    Metrics (rows processed, cost so far) are updated on every
    transition by the live executor. If transition dropped these,
    the MCP ``ondine_status`` tool would show stale zeroes."""
    handle = registry.create(RunSpec(pipeline_id=str(uuid4())))
    registry.transition(handle.run_id, RunStatus.RUNNING)

    updated = registry.transition(
        handle.run_id,
        RunStatus.SUBMITTED_REMOTE,
        metrics={"rows": 1000, "cost": str(Decimal("0.42"))},
        provider_job_id="batch_abc123",
    )

    assert updated.provider_job_id == "batch_abc123"
    assert updated.metrics["rows"] == 1000
    fresh = registry.get(handle.run_id)
    assert fresh.provider_job_id == "batch_abc123"
    assert fresh.metrics["cost"] == str(Decimal("0.42"))


# ── regression #5: list filters & ordering ────────────────────────────


def test_list_filters_by_status(registry: RunRegistry) -> None:
    """Regression: the MCP ``ondine_status`` tool lists RUNNING jobs for
    the dashboard. If list(status=) ignored the filter, the dashboard
    would show terminal jobs forever and the user would never know a
    run finished until they refreshed."""
    h1 = registry.create(RunSpec(pipeline_id=str(uuid4())))
    h2 = registry.create(RunSpec(pipeline_id=str(uuid4())))
    h3 = registry.create(RunSpec(pipeline_id=str(uuid4())))

    registry.transition(h2.run_id, RunStatus.RUNNING)
    registry.transition(h3.run_id, RunStatus.RUNNING)
    registry.transition(h3.run_id, RunStatus.SUCCEEDED)

    assert {h.run_id for h in registry.list()} == {h1.run_id, h2.run_id, h3.run_id}
    assert [h.run_id for h in registry.list(RunStatus.RUNNING)] == [h2.run_id]
    assert [h.run_id for h in registry.list(RunStatus.PENDING)] == [h1.run_id]
    assert registry.list(RunStatus.FAILED) == []


def test_list_returns_newest_first(registry: RunRegistry) -> None:
    """Regression: the CLI ``ondine status`` shows the most recent run
    first. If list returned oldest-first, the user would scroll past
    stale jobs to find their latest one."""
    ids = [registry.create(RunSpec(pipeline_id=str(uuid4()))).run_id for _ in range(3)]
    listed = [h.run_id for h in registry.list()]
    assert listed == list(reversed(ids))


# ── regression #6: spec_snapshot round-trip ───────────────────────────


def test_spec_snapshot_roundtrips_arbitrary_dict(registry: RunRegistry) -> None:
    """Regression: the spec snapshot is what resume relies on — without
    it, a crashed ``provider_batch`` run could not be reattached by the
    CLI. Losing nested keys would silently drop the prompt/model."""
    snapshot = {
        "pipeline_id": str(uuid4()),
        "model": "gpt-4o-mini",
        "prompt": "Summarize: {text}",
        "nested": {"batch_size": 32, "columns": ["a", "b"]},
    }
    handle = registry.create(RunSpec.from_dict({"spec_snapshot": snapshot}))

    fresh = registry.get(handle.run_id)
    assert fresh.spec_snapshot == snapshot
    assert fresh.spec_snapshot["nested"]["columns"] == ["a", "b"]


# ── regression #7: Pipeline.execute(run_id=...) integration ──────────


def _make_pipeline(
    checkpoint_dir: Path,
    *,
    failing: bool = False,
) -> tuple[Any, PipelineSpecifications, Any]:
    """Build a minimal Pipeline with a mocked LLM client.

    The mock returns deterministic responses so execute() succeeds
    without network access — we only need the registry wire-up, not the
    LLM behaviour (which has its own test suite).

    When ``failing=True`` the mock client raises on every invoke and the
    processing spec uses ``ErrorPolicy.FAIL`` so the error propagates
    out of execute(). This exercises the real failure path instead of
    inventing a ``fail_fast`` knob that would just shadow ErrorPolicy.
    """
    from unittest.mock import patch

    from ondine.adapters.llm_client import LLMClient
    from ondine.api.pipeline import Pipeline
    from ondine.core.models import LLMResponse
    from ondine.core.specifications import ErrorPolicy, ProcessingSpec

    df = pd.DataFrame({"text": ["a", "b", "c"]})
    processing = ProcessingSpec(
        checkpoint_dir=checkpoint_dir,
        cleanup_on_success=False,
        progress_mode="none",
        max_retries=0,
        retry_delay=0.0,
    )
    if failing:
        processing = processing.model_copy(update={"error_policy": ErrorPolicy.FAIL})

    specs = PipelineSpecifications(
        dataset=DatasetSpec(
            source_type=DataSourceType.DATAFRAME,
            input_columns=["text"],
            output_columns=["result"],
        ),
        prompt=PromptSpec(template="{text}"),
        llm=LLMSpec(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            input_cost_per_1k_tokens=Decimal("0.00015"),
            output_cost_per_1k_tokens=Decimal("0.0006"),
        ),
        processing=processing,
    )

    if failing:

        class FailingClient(LLMClient):
            def invoke(self, prompt, **kwargs):
                raise RuntimeError("simulated provider failure")

            async def ainvoke(self, prompt, **kwargs):
                return self.invoke(prompt, **kwargs)

            def structured_invoke(self, prompt, output_cls, **kwargs):
                return self.invoke(prompt, **kwargs)

            async def structured_invoke_async(self, prompt, output_cls, **kwargs):
                return self.invoke(prompt, **kwargs)

            async def start(self):
                pass

            async def stop(self):
                pass

            def estimate_tokens(self, text):
                return len(text) // 4

        mock_client = FailingClient(specs.llm)
    else:

        class MockClient(LLMClient):
            def invoke(self, prompt, **kwargs):
                return LLMResponse(
                    text="ok",
                    tokens_in=1,
                    tokens_out=1,
                    model=self.model,
                    cost=Decimal("0.0001"),
                    latency_ms=1.0,
                )

            async def ainvoke(self, prompt, **kwargs):
                return self.invoke(prompt, **kwargs)

            def structured_invoke(self, prompt, output_cls, **kwargs):
                return self.invoke(prompt, **kwargs)

            async def structured_invoke_async(self, prompt, output_cls, **kwargs):
                return self.invoke(prompt, **kwargs)

            async def start(self):
                pass

            async def stop(self):
                pass

            def estimate_tokens(self, text):
                return len(text) // 4

        mock_client = MockClient(specs.llm)

    pipeline = Pipeline(specs, dataframe=df)
    patcher = patch(
        "ondine.api.pipeline.create_llm_client",
        return_value=mock_client,
    )
    patcher.start()
    return pipeline, specs, patcher


def test_pipeline_execute_with_run_id_registers_and_traces(
    tmp_path: Path,
) -> None:
    """Regression: ``Pipeline.execute(run_id=...)`` must register the run
    and trace its terminal state in the registry. Without the wire-up,
    the MCP ``ondine_status`` tool would return PENDING forever for a
    run that actually finished — a broken dashboard."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    registry = RunRegistry(checkpoint_dir)
    spec = RunSpec(
        pipeline_id=str(uuid4()),
        dataset="df",
        spec_snapshot={"model": "gpt-4o-mini"},
    )
    handle = registry.create(spec)

    pipeline, _, patcher = _make_pipeline(checkpoint_dir)
    try:
        pipeline.execute(run_id=handle.run_id, registry=registry)
    finally:
        patcher.stop()

    final = registry.get(handle.run_id)
    assert final is not None
    assert final.status is RunStatus.SUCCEEDED
    assert final.metrics["total_rows"] == 3


def test_pipeline_execute_without_run_id_leaves_registry_empty(
    tmp_path: Path,
) -> None:
    """Regression: the default path (run_id=None) must NOT touch the
    registry. Existing users who never opted in would otherwise find a
    ``runs.db`` file appearing in their checkpoint dir — a breaking
    side-effect on every execute() call."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    registry = RunRegistry(checkpoint_dir)

    pipeline, _, patcher = _make_pipeline(checkpoint_dir)
    try:
        pipeline.execute()
    finally:
        patcher.stop()

    registry2 = RunRegistry(checkpoint_dir)
    try:
        assert registry2.list() == []
    finally:
        registry.close()
        registry2.close()


def test_pipeline_execute_failure_records_failed_state(
    tmp_path: Path,
) -> None:
    """Regression: when the pipeline raises, the registry must record
    FAILED so the dashboard doesn't keep a dead run as RUNNING —
    otherwise operators never know it died.

    We exercise the real failure path: a mock LLM client that raises and
    an ``ErrorPolicy.FAIL`` processing spec, so the error propagates out
    of execute() exactly as a genuine provider outage would.
    """
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    registry = RunRegistry(checkpoint_dir)
    spec = RunSpec(pipeline_id=str(uuid4()), dataset="df")
    handle = registry.create(spec)

    pipeline, _, patcher = _make_pipeline(checkpoint_dir, failing=True)
    try:
        with pytest.raises(Exception):
            pipeline.execute(run_id=handle.run_id, registry=registry)
    finally:
        patcher.stop()

    final = registry.get(handle.run_id)
    assert final is not None
    assert final.status is RunStatus.FAILED


# ── regression: update_metrics writes live progress without a transition ──


def test_update_metrics_merges_without_changing_status(
    registry: RunRegistry,
) -> None:
    """Regression: the MCP ``ondine_status`` tool reports live rows/cost for a
    RUNNING job. If update_metrics transitioned status or replaced metrics
    wholesale, status polling would either break the FSM or lose the fields the
    terminal transition wrote. Live progress must merge onto the row in place.
    """
    handle = registry.create(RunSpec(pipeline_id=str(uuid4())))
    registry.transition(handle.run_id, RunStatus.RUNNING)

    registry.update_metrics(handle.run_id, {"processed_rows": 42, "cost": "1.25"})

    after = registry.get(handle.run_id)
    assert after is not None
    assert after.status is RunStatus.RUNNING  # unchanged
    assert after.metrics["processed_rows"] == 42
    assert after.metrics["cost"] == "1.25"


def test_update_metrics_is_visible_to_a_second_process(
    tmp_path: Path,
) -> None:
    """Regression: a status poller is a different process opening its own
    registry handle. If update_metrics were in-memory only, the MCP status tool
    would never see live progress from the worker thread/process that ran the
    job. The merge must hit the on-disk row.
    """
    registry_a = RunRegistry(tmp_path / "runs.db")
    handle = registry_a.create(RunSpec(pipeline_id=str(uuid4())))
    registry_a.transition(handle.run_id, RunStatus.RUNNING)
    registry_a.update_metrics(handle.run_id, {"processed_rows": 7})
    registry_a.close()

    registry_b = RunRegistry(tmp_path / "runs.db")
    try:
        seen = registry_b.get(handle.run_id)
        assert seen is not None
        assert seen.metrics.get("processed_rows") == 7
    finally:
        registry_b.close()


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def registry(tmp_path: Path) -> RunRegistry:
    db_path = tmp_path / "runs.db"
    r = RunRegistry(db_path)
    yield r
    r.close()
