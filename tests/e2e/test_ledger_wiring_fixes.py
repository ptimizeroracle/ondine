"""Row-ledger wiring, step 1 review fixes: each test pins one finding.

The step 1 review (findings 1, 2, 3, 5, 6, 9, 10 and the grounding "skip"
gap) found places where the ledger was told the wrong rows, told nothing, or
was exposed live. These tests run real pipelines over the parity net's
scripted provider (``test_harness_parity.py``) and read the result the way a
caller does.

Where a fix changes only the ledger's own view, the legacy views (``errors``,
``lost_row_indices``, ``is_complete``, the counters) are asserted unchanged
next to it. The one fix that changes ``errors`` on purpose (finding 2: a
skipped batch after a resume gap blamed the wrong rows) says so.
"""

from __future__ import annotations

import asyncio
import gc
from collections import Counter

import pytest

from ondine.adapters.response_cache import SqliteResponseCache
from ondine.context.memory_store import InMemoryContextStore
from ondine.harness.ledger import LedgerSnapshot, RowLedger, StopReason
from ondine.harness.status import RowStatus
from ondine.utils.budget_controller import BudgetExceededError
from tests.e2e.conformance_harness import (
    COST_PER_CALL,
    answer_for,
    conformance_frame,
    token_for,
)
from tests.e2e.test_harness_parity import (
    PERMANENT,
    ScriptedLedgerClient,
    _auto_retry,
    build,
    failure,
    observe,
    skipped,
)

U = RowStatus.UNCHECKED
F = RowStatus.FAILED


def statuses(snapshot: LedgerSnapshot | None) -> list[RowStatus]:
    assert isinstance(snapshot, LedgerSnapshot), f"not a snapshot: {snapshot!r}"
    assert sum(snapshot.counts.values()) == len(snapshot.rows)
    return [outcome.status for outcome in snapshot.rows]


def reasons(snapshot: LedgerSnapshot) -> dict[int, str]:
    return {o.row: o.reason for o in snapshot.rows if o.status is F}


# ── finding 2: a skipped batch blames the rows it held ────────────────────


def _jinja_prompt_failing_on(*rows: int) -> str:
    """A prompt whose rendering raises for ``rows`` only: those rows are never asked."""
    bad = ", ".join(f"'{token_for(row)}'" for row in rows)
    return (
        "Answer for {{ marker }}{% if marker in [" + bad + "] %}{{ 1 // 0 }}{% endif %}"
    )


def test_a_skipped_batch_after_a_resume_gap_blames_its_own_rows(tmp_path):
    """The batch after a gap holds rows 1 and 3; its skip must name 1 and 3.

    The first attempt never asks rows 1 and 3 (their prompt cannot be built)
    and answers the others, so the cache has a gap; its budget then stops it,
    which leaves a checkpoint. The resumed attempt sends rows 1 and 3 as one
    batch of two, which the provider refuses.

    Before the fix, the skip was reported as two consecutive rows from the
    batch's first row: rows 1 and 2. Row 2 was answered in the first attempt
    and row 3 was never named. This changes ``result.errors`` on purpose.
    """
    frame = conformance_frame(6)
    first = (
        build(
            ScriptedLedgerClient(),
            frame,
            checkpoint_dir=tmp_path,
            concurrency=1,
        )
        .with_prompt(_jinja_prompt_failing_on(1, 3))
        .with_jinja2(True)
        .with_max_budget(float(COST_PER_CALL) * 3.5)
        .build()
    )
    with pytest.raises(BudgetExceededError):
        first.execute()
    assert statuses(first.row_ledger) == [U, F, U, F, U, U]  # the gap

    second_client = ScriptedLedgerClient(permanent={token_for(1)})
    resumed = (
        build(
            second_client,
            frame,
            checkpoint_dir=tmp_path,
            concurrency=1,
            batch_size=2,
        )
        .build()
        .execute(resume_from=first.session_id)
    )

    assert second_client.calls == Counter({token_for(1): 1})  # one batch: 1 + 3
    message = failure(token_for(1))
    assert observe(resumed).errors == [skipped(1, message), skipped(3, message)]
    assert resumed.lost_row_indices == [1, 3]
    assert statuses(resumed.row_ledger) == [U, F, U, F, U, U]


# ── finding 9: a substituted default is a provider failure, not an answer ──


@pytest.mark.parametrize("batch_size", [1, 3], ids=["batch_size=1", "batch_size=3"])
def test_a_use_default_answer_is_a_provider_failure_in_the_ledger(batch_size):
    """USE_DEFAULT delivers its default value; the ledger knows it is not an answer.

    The legacy views are today's, unchanged: one prompt per row delivers the
    empty default and lists nothing; a batch's empty default fails to split
    and is listed under the splitter, as it always was.
    """
    client = ScriptedLedgerClient(permanent={PERMANENT})
    result = (
        build(
            client,
            conformance_frame(6),
            concurrency=1,
            error_policy="use_default",
            batch_size=batch_size,
        )
        .build()
        .execute()
    )

    lost = {2} if batch_size == 1 else {0, 1, 2}
    assert statuses(result.row_ledger) == [F if r in lost else U for r in range(6)]
    for row, reason in reasons(result.row_ledger).items():
        assert reason.startswith(f"LLMInvocation: {failure(PERMANENT)}"), row
        assert "default" in reason
    assert all(not result.row_ledger.rows[r].attempts for r in lost)

    outcome = observe(result)
    if batch_size == 1:
        assert outcome.rows[2] == (PERMANENT, "")
        assert (outcome.errors, outcome.lost, outcome.complete) == ([], [], True)
        assert (outcome.skipped, outcome.failed) == (0, 0)
    else:
        assert [e[:3] for e in outcome.errors] == [
            (r, "BatchDisaggregator", "parse_failure") for r in (0, 1, 2)
        ]
        assert (outcome.skipped, outcome.failed) == (0, 3)


# ── grounding "skip": a blanked row is never "delivered" ──────────────────


class _RefusesToGround(InMemoryContextStore):
    """Grounds every answer except the ones naming ``ungrounded`` tokens."""

    def __init__(self, *ungrounded: str) -> None:
        super().__init__()
        self._ungrounded = set(ungrounded)

    def ground(self, response_text, source_sentences, threshold=0.3, embed_fn=None):
        if any(token in response_text for token in self._ungrounded):
            return []
        return super().ground(response_text, source_sentences, 0.0, embed_fn)


def test_a_row_blanked_by_grounding_skip_is_failed_in_the_ledger():
    """The ledger names the verification stage; the legacy views are today's.

    Today's error list never named a row grounding blanked, so it still does
    not, and the run still reads complete.
    """
    result = (
        build(ScriptedLedgerClient(), conformance_frame(4), concurrency=1)
        .with_context_store(_RefusesToGround(answer_for(token_for(1))))
        .with_grounding(threshold=0.5, action="skip")
        .build()
        .execute()
    )

    outcome = observe(result)
    assert outcome.rows[1] == (token_for(1), None)
    assert (outcome.errors, outcome.lost, outcome.complete) == ([], [], True)

    assert statuses(result.row_ledger) == [U, F, U, U]
    assert reasons(result.row_ledger)[1].startswith("ContextVerification: ")


# ── finding 5: the result and the pipeline expose a frozen snapshot ───────


def test_the_result_and_the_pipeline_expose_a_snapshot_taken_after_auto_retry():
    empty_row = token_for(1)
    client = ScriptedLedgerClient(empty={empty_row: 1})
    pipeline = _auto_retry(
        build(client, conformance_frame(3), concurrency=1), attempts=1
    )

    result = pipeline.execute()

    assert isinstance(result.row_ledger, LedgerSnapshot)
    assert isinstance(pipeline.row_ledger, LedgerSnapshot)
    assert result.row_ledger == pipeline.row_ledger
    assert result.row_ledger is result.row_ledger  # frozen once, not per read
    # The retry pass recovered row 1 before the snapshot was taken.
    assert result.row_ledger.rows[1].delivered_attempt == 2


# ── finding 6: resume reads the cache's completed rows once ───────────────


def test_resume_reads_the_completed_rows_once(tmp_path, monkeypatch):
    frame = conformance_frame(6)
    first = (
        build(ScriptedLedgerClient(), frame, checkpoint_dir=tmp_path, concurrency=1)
        .with_max_budget(float(COST_PER_CALL) * 2.5)
        .build()
    )
    with pytest.raises(BudgetExceededError):
        first.execute()

    reads = Counter()
    original = SqliteResponseCache.completed_row_indices

    def counted(self, session_id):
        reads["completed_row_indices"] += 1
        return original(self, session_id)

    monkeypatch.setattr(SqliteResponseCache, "completed_row_indices", counted)
    resumed = (
        build(ScriptedLedgerClient(), frame, checkpoint_dir=tmp_path, concurrency=1)
        .build()
        .execute(resume_from=first.session_id)
    )

    assert reads["completed_row_indices"] == 1
    assert statuses(resumed.row_ledger) == [U] * 6


# ── finding 1: a stream keeps no chunk's live ledger ──────────────────────


def _live_ledgers() -> int:
    gc.collect()
    return sum(isinstance(o, RowLedger) for o in gc.get_objects())


def test_a_stream_keeps_no_chunk_ledger_alive_beyond_its_result():
    """A chunk's ledger lives exactly as long as the chunk result holding it.

    The stream itself keeps only the run's ledger, so its memory stays
    proportional to the chunk size. A chunk result keeps its compact live
    ledger (cheaper than a snapshot) and freezes it only when read; once the
    consumer drops the results, none of them is left.
    """
    before = _live_ledgers()
    pipeline = (
        build(ScriptedLedgerClient(permanent={PERMANENT}), conformance_frame(8))
        .with_streaming(chunk_size=2)
        .build()
    )

    chunks = list(pipeline.execute_stream())

    assert len(chunks) == 4
    assert isinstance(chunks[1].row_ledger, LedgerSnapshot)
    assert statuses(chunks[1].row_ledger) == [F, U]
    assert statuses(pipeline.row_ledger) == [U, U, F, U, U, U, U, U]

    # Reading pipeline.row_ledger froze the run's ledger too, so once the
    # consumer drops the chunk results no live ledger is left at all.
    del chunks
    assert _live_ledgers() - before == 0


# ── finding 10: a stream its consumer closes records a stop ───────────────


def test_a_stream_closed_by_its_consumer_records_a_stop():
    pipeline = (
        build(ScriptedLedgerClient(), conformance_frame(6))
        .with_streaming(chunk_size=2)
        .build()
    )

    async def take_first_chunk() -> None:
        stream = pipeline.execute_stream_async(chunk_size=2)
        async for _ in stream:
            break
        await stream.aclose()

    asyncio.run(take_first_chunk())

    snapshot = pipeline.row_ledger
    assert statuses(snapshot) == [U, U, F, F, F, F]
    assert snapshot.stop_reason is StopReason.CRASH
    assert set(reasons(snapshot).values()) == {"run stopped before this row: crash"}
