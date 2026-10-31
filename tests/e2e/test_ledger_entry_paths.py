"""Every entry path leaves every row with exactly one ledger status.

Interfaces spec, section 6, test 2: whatever way a run is entered — sync,
async, streaming, the auto-retry pass, resume — each input row ends with one
status, and the counts per status sum to the row count. Streaming chunks and
the auto-retry pass run as sub-pipelines with ledgers of their own, merged
into the run's ledger by row mapping, so those are the paths where a row's
fate can be lost or doubled.

These runs reuse the parity net's scripted provider and builders
(``test_harness_parity.py``): the same frames whose *legacy* outcome that file
freezes are checked here through the *ledger*. Without a contract, a row with
an answer is ``unchecked`` and a lost row is ``failed`` with the provider's
own message as its reason.

Where the two views differ in migration step 1 (rows a run never reached,
rows the auto-retry pass recovered), the ledger states the truth and the
legacy views keep today's answer; the tests below say which is which.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import TYPE_CHECKING

import pytest

from ondine.harness.status import FINAL_STATUSES, RowStatus
from ondine.utils.budget_controller import BudgetExceededError
from tests.e2e.conformance_harness import (
    COST_PER_CALL,
    ProviderError,
    conformance_frame,
    token_for,
)
from tests.e2e.test_harness_parity import (
    PERMANENT,
    TRANSIENT,
    WHOLE_FRAME_OUTCOME,
    ScriptedLedgerClient,
    ScriptedStructuredClient,
    _auto_retry,
    _run_async,
    _run_concurrent,
    _run_sync,
    build,
    failure,
    hiccup,
    observe,
)

if TYPE_CHECKING:
    from ondine.harness.ledger import LedgerSnapshot

U = RowStatus.UNCHECKED
F = RowStatus.FAILED


def statuses(snapshot: LedgerSnapshot | None, rows: int) -> list[RowStatus]:
    """The ledger's statuses, after checking the section 6 invariant."""
    assert snapshot is not None, "the run left no ledger"
    assert_one_status_per_row(snapshot, rows)
    return [outcome.status for outcome in snapshot.rows]


def assert_one_status_per_row(snapshot: LedgerSnapshot, rows: int) -> None:
    assert [outcome.row for outcome in snapshot.rows] == list(range(rows))
    assert all(outcome.status in FINAL_STATUSES for outcome in snapshot.rows)
    assert sum(snapshot.counts.values()) == rows
    assert snapshot.violations == ()


def reasons(snapshot: LedgerSnapshot) -> dict[int, str]:
    """Reason per failed row."""
    return {
        outcome.row: outcome.reason for outcome in snapshot.rows if outcome.status is F
    }


def _shape(errors) -> list[tuple[int, str, str, str]]:
    return [(e.row_index, e.stage, e.error_type, e.message) for e in errors]


# ── execute() and execute_async() ─────────────────────────────────────────


@pytest.mark.parametrize(
    "run",
    [_run_sync, _run_concurrent, _run_async],
    ids=["execute", "execute-concurrent", "execute_async"],
)
def test_whole_frame_paths_give_one_status_per_row(run):
    client = ScriptedLedgerClient(permanent={PERMANENT}, transient={TRANSIENT: 1})

    result = run(client)

    assert statuses(result.row_ledger, 6) == [U, U, F, U, U, U]
    assert reasons(result.row_ledger) == {2: f"LLMInvocation: {failure(PERMANENT)}"}
    # With no stop and no sub-pipeline, the ledger's own error list is the
    # legacy one, entry for entry.
    snapshot = result.row_ledger
    assert _shape(snapshot.errors) == _shape(result.errors)
    assert snapshot.lost_row_indices == result.lost_row_indices
    assert snapshot.is_complete is result.is_complete
    assert observe(result) == WHOLE_FRAME_OUTCOME


def test_every_answered_row_records_its_answer_and_cost():
    client = ScriptedLedgerClient(permanent={PERMANENT})

    result = _run_sync(client)

    rows = result.row_ledger.rows
    assert [len(row.attempts) for row in rows] == [1, 1, 0, 1, 1, 1]
    assert sum(row.cost for row in rows) == result.costs.total_cost


def test_a_mega_batch_gives_one_status_per_row():
    client = ScriptedLedgerClient(permanent={PERMANENT}, transient={TRANSIENT: 1})

    result = build(client, conformance_frame(6), batch_size=3).build().execute()

    assert statuses(result.row_ledger, 6) == [F, F, F, U, U, U]
    rows = result.row_ledger.rows
    assert sum(row.cost for row in rows) == COST_PER_CALL


@pytest.mark.parametrize("batch_size", [1, 3], ids=["batch_size=1", "batch_size=3"])
def test_structured_output_gives_one_status_per_row(batch_size):
    from ondine import PipelineBuilder
    from tests.e2e.conformance_harness import StructuredAnswerBatch

    client = ScriptedStructuredClient(permanent={PERMANENT}, transient={TRANSIENT: 1})
    result = (
        PipelineBuilder.create()
        .from_dataframe(
            conformance_frame(6),
            input_columns=["marker"],
            output_columns=["token", "category"],
        )
        .with_prompt("Answer for {marker}")
        .with_custom_llm_client(client)
        .with_structured_output(StructuredAnswerBatch)
        .with_batch_size(batch_size)
        .build()
        .execute()
    )

    lost = {2} if batch_size == 1 else {0, 1, 2}
    assert statuses(result.row_ledger, 6) == [
        F if row in lost else U for row in range(6)
    ]


# ── streaming: one child ledger per chunk, merged by row mapping ──────────


def _streaming_client() -> ScriptedLedgerClient:
    return ScriptedLedgerClient(
        permanent={PERMANENT, token_for(5)}, transient={token_for(1): 1}
    )


def test_streaming_merges_each_chunk_at_its_input_rows():
    """Chunk results keep chunk-local numbering; the run's ledger does not."""
    pipeline = (
        build(_streaming_client(), conformance_frame(6))
        .with_streaming(chunk_size=4)
        .build()
    )

    chunks = list(pipeline.execute_stream())

    assert statuses(pipeline.row_ledger, 6) == [U, U, F, U, U, F]
    assert reasons(pipeline.row_ledger)[5] == (
        f"LLMInvocation: {failure(token_for(5))}"
    )
    assert [
        statuses(chunk.row_ledger, n) for chunk, n in zip(chunks, (4, 2), strict=True)
    ] == [
        [U, U, F, U],
        [U, F],
    ]


def test_pipelined_streaming_merges_each_chunk_at_its_input_rows():
    pipeline = build(_streaming_client(), conformance_frame(6)).build()

    list(pipeline.execute_stream_pipelined(chunk_size=4))

    assert statuses(pipeline.row_ledger, 6) == [U, U, F, U, U, F]


def test_a_stream_stopped_by_its_budget_fails_the_rows_it_never_reached():
    """Interfaces spec, section 6, test 8, on the streaming path."""
    client = ScriptedLedgerClient()
    pipeline = (
        build(client, conformance_frame(6))
        .with_streaming(chunk_size=2)
        .with_max_budget(float(COST_PER_CALL * 3))
        .build()
    )

    with pytest.raises(BudgetExceededError):
        list(pipeline.execute_stream())

    assert statuses(pipeline.row_ledger, 6) == [U, U, U, U, F, F]
    assert set(reasons(pipeline.row_ledger).values()) == {
        "run stopped before this row: budget cap"
    }


# ── the auto-retry pass: a child ledger merged as a later attempt ─────────


def test_auto_retry_merges_its_pass_as_a_later_attempt():
    """The re-asked empty row is delivered from the retry pass's attempt.

    The legacy views are today's: the retry pass never touched the error
    list, and the parity net freezes that. The ledger records the pass as
    attempt 2 of row 1 and keeps the two lost rows failed.
    """
    empty_row = token_for(1)
    client = ScriptedLedgerClient(
        permanent={PERMANENT}, transient={TRANSIENT: 1}, empty={empty_row: 1}
    )

    result = _auto_retry(
        build(client, conformance_frame(6), max_retries=1, concurrency=1),
        attempts=2,
    ).execute()

    assert statuses(result.row_ledger, 6) == [U, U, F, U, F, U]
    rows = result.row_ledger.rows
    assert rows[1].delivered_attempt == 2
    assert [a.attempt for a in rows[1].attempts] == [1, 2]
    assert reasons(result.row_ledger)[4] == f"LLMInvocation: {hiccup(TRANSIENT)}"
    assert sum(row.cost for row in rows) == result.costs.total_cost


def test_auto_retry_after_a_mega_batch_gives_one_status_per_row():
    client = ScriptedLedgerClient(permanent={PERMANENT}, empty={TRANSIENT: 1})

    result = _auto_retry(
        build(client, conformance_frame(6), batch_size=3), attempts=1
    ).execute()

    assert statuses(result.row_ledger, 6) == [F, F, F, U, U, U]
    assert result.row_ledger.rows[4].delivered_attempt == 2


# ── resume: the previous attempt's answers merged from the cache ──────────


def test_resume_gives_one_status_per_row_across_both_attempts(tmp_path):
    frame = conformance_frame(6)
    first = build(
        ScriptedLedgerClient(permanent={PERMANENT}),
        frame,
        checkpoint_dir=tmp_path,
        error_policy="fail",
        concurrency=1,
    ).build()

    with pytest.raises(ProviderError):
        first.execute()

    # The aborted run never reached rows 2-5: the fail policy raised on row 2
    # before anything was recorded for it.
    assert statuses(first.row_ledger, 6) == [U, U, F, F, F, F]
    assert set(reasons(first.row_ledger).values()) == {
        "run stopped before this row: crash"
    }

    second_client = ScriptedLedgerClient(
        permanent={PERMANENT}, transient={TRANSIENT: 1}
    )
    resumed = (
        build(second_client, frame, checkpoint_dir=tmp_path, concurrency=1)
        .build()
        .execute(resume_from=first.session_id)
    )

    assert statuses(resumed.row_ledger, 6) == [U, U, F, U, U, U]
    rows = resumed.row_ledger.rows
    # Rows 0 and 1 were answered by the first attempt, restored from its cache.
    assert [len(row.attempts) for row in rows] == [1, 1, 0, 1, 1, 1]
    assert observe(resumed) == WHOLE_FRAME_OUTCOME
    assert second_client.calls == Counter(
        {PERMANENT: 1, token_for(3): 1, TRANSIENT: 2, token_for(5): 1}
    )


def test_resume_reports_a_row_the_previous_attempt_skipped(tmp_path):
    """A skip the first attempt cached is a failure in the resumed ledger.

    The first attempt skips row 2, answers rows 0, 1 and 3, then stops on its
    budget. Resume restores all four from the cache, skip included, and asks
    only rows 4 and 5.

    The legacy views keep today's answer, which the ledger corrects later:
    the restored skip is not in the resumed run's error list and its cell
    keeps the ``[SKIPPED]`` text, while the checkpoint's counter still counts
    it, so the run reads incomplete with no lost row.
    """
    frame = conformance_frame(6)
    first = (
        build(
            ScriptedLedgerClient(permanent={PERMANENT}),
            frame,
            checkpoint_dir=tmp_path,
            concurrency=1,
        )
        .with_max_budget(float(COST_PER_CALL) * 2.5)
        .build()
    )
    with pytest.raises(BudgetExceededError):
        first.execute()
    assert statuses(first.row_ledger, 6) == [U, U, F, U, F, F]

    second_client = ScriptedLedgerClient()
    resumed = (
        build(second_client, frame, checkpoint_dir=tmp_path, concurrency=1)
        .build()
        .execute(resume_from=first.session_id)
    )

    assert statuses(resumed.row_ledger, 6) == [U, U, F, U, U, U]
    assert reasons(resumed.row_ledger) == {2: f"LLMInvocation: {failure(PERMANENT)}"}
    assert second_client.calls == Counter({token_for(4): 1, token_for(5): 1})
    outcome = observe(resumed)
    assert outcome.rows[2] == (PERMANENT, "[SKIPPED]")
    assert (outcome.errors, outcome.lost, outcome.complete) == ([], [], False)
    assert (outcome.skipped, outcome.failed) == (1, 0)


# ── a budget stop on the whole-frame path ─────────────────────────────────


def test_a_budget_stop_fails_every_row_the_run_never_reached():
    """Interfaces spec, section 6, test 8: rows after the stop are failed."""
    client = ScriptedLedgerClient()
    pipeline = (
        build(client, conformance_frame(6), concurrency=1)
        .with_max_budget(float(COST_PER_CALL) * 2.5)
        .build()
    )

    with pytest.raises(BudgetExceededError):
        pipeline.execute()

    assert statuses(pipeline.row_ledger, 6) == [U, U, U, F, F, F]
    assert set(reasons(pipeline.row_ledger).values()) == {
        "run stopped before this row: budget cap"
    }


def test_an_async_run_leaves_its_ledger_on_the_pipeline_too():
    client = ScriptedLedgerClient(permanent={PERMANENT})
    pipeline = (
        build(client, conformance_frame(6)).with_async_execution(max_concurrency=4)
    ).build()

    result = asyncio.run(pipeline.execute_async())

    assert pipeline.row_ledger == result.row_ledger
