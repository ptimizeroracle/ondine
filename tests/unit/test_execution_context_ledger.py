"""ExecutionContext keeps its loss surface, read from the row ledger.

Migration step 1 makes the ledger the only record of row fate. The context
used to keep its own lost-row set, error list and counters; those are now
views of its ledger, with the same names and the same answers, so the stages
and the pipeline that read them do not change. Each test here pins one piece
of that surface against the ledger behind it.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from ondine.harness.contract import OutputField, build_contract
from ondine.harness.ledger import RowLedger
from ondine.harness.status import RowStatus
from ondine.orchestration.execution_context import ExecutionContext


def _shape(errors) -> list[tuple[int, str, str, str]]:
    return [(e.row_index, e.stage, e.error_type, e.message) for e in errors]


def test_losses_are_ledger_events_and_the_old_surface_reads_them():
    context = ExecutionContext(total_rows=4)
    context.open_ledger(build_contract(["answer"]))

    context.record_answers([0, 3], model="m", tokens=6, cost=Decimal("0.02"))
    context.record_skipped_rows(1, 1, stage="LLMInvocation", message="down")
    context.record_failed_rows([2], stage="ResponseParser", message="not JSON")

    assert _shape(context.row_errors) == [
        (1, "LLMInvocation", "skipped", "down"),
        (2, "ResponseParser", "parse_failure", "not JSON"),
    ]
    assert (context.skipped_rows, context.failed_rows) == (1, 1)
    snapshot = context.ledger.snapshot()
    assert [row.status for row in snapshot.rows] == [
        RowStatus.UNCHECKED,
        RowStatus.FAILED,
        RowStatus.FAILED,
        RowStatus.UNCHECKED,
    ]
    assert context.get_stats().skipped_rows == 1


def test_a_batch_answer_shares_its_cost_and_tokens_among_its_rows():
    context = ExecutionContext(total_rows=3)

    context.record_answers([0, 1, 2], model="m", tokens=10, cost=Decimal("0.03"))

    attempts = [row.attempts[0] for row in context.ledger.snapshot().rows]
    assert [a.tokens for a in attempts] == [4, 3, 3]
    assert sum(a.cost for a in attempts) == Decimal("0.03")
    assert {a.model for a in attempts} == {"m"}


def test_a_batch_cost_that_does_not_divide_still_sums_exactly():
    context = ExecutionContext(total_rows=3)

    context.record_answers([0, 1, 2], model="m", tokens=3, cost=Decimal("0.0001"))

    attempts = [row.attempts[0] for row in context.ledger.snapshot().rows]
    assert sum(a.cost for a in attempts) == Decimal("0.0001")


def test_the_error_list_respects_the_contexts_own_recording_limit(monkeypatch):
    monkeypatch.setattr(ExecutionContext, "MAX_RECORDED_ROW_ERRORS", 2)
    context = ExecutionContext(total_rows=5)

    context.record_skipped_rows(0, 5, stage="LLMInvocation", message="down")

    assert [e.row_index for e in context.row_errors] == [0, 1]
    assert context.skipped_rows == 5


def test_a_context_never_told_its_size_widens_its_ledger():
    """Stages are driven directly in tests and by backends with no loader."""
    context = ExecutionContext()

    context.record_failed_rows([3], stage="ResponseParser", message="not JSON")

    assert context.ledger.row_count == 4
    assert [e.row_index for e in context.row_errors] == [3]
    assert context.failed_rows == 1


def test_reading_the_surface_before_any_report_creates_nothing():
    context = ExecutionContext()

    assert context.row_errors == []
    assert (context.failed_rows, context.skipped_rows) == (0, 0)


def test_the_ledger_is_opened_over_the_known_rows_with_the_contracts_policy():
    context = ExecutionContext(total_rows=3)
    contract = build_contract(["answer"])

    ledger = context.open_ledger(contract)

    assert ledger is context.ledger
    assert ledger.row_count == 3
    assert ledger.policy.checked is False


def test_opening_keeps_rows_reported_before_the_size_was_known():
    context = ExecutionContext()
    context.record_failed_rows([0], stage="PromptFormatter", message="bad row")
    context.total_rows = 2

    context.open_ledger(build_contract(["answer"]))

    assert context.ledger.row_count == 2
    assert [e.row_index for e in context.row_errors] == [0]


def test_opening_under_a_different_policy_after_reports_is_refused():
    context = ExecutionContext(total_rows=1)
    context.record_failed_rows([0], stage="PromptFormatter", message="bad row")

    with pytest.raises(RuntimeError, match="policy"):
        context.open_ledger(build_contract(["answer"], fields=[OutputField("answer")]))


# ---------------------------------------------------------------------------
# Counts carried over from a checkpoint
# ---------------------------------------------------------------------------


def test_checkpoint_counts_are_carried_and_new_losses_add_to_them():
    """Resume restores the counters but not the rows behind them, as before."""
    data = ExecutionContext(total_rows=3).to_checkpoint()
    data.update(failed_rows=2, skipped_rows=1)
    context = ExecutionContext.from_checkpoint(data)

    context.record_skipped_rows(0, 1, stage="LLMInvocation", message="down")

    assert (context.failed_rows, context.skipped_rows) == (2, 2)
    assert [e.row_index for e in context.row_errors] == [0]


def test_a_checkpoint_round_trip_keeps_the_counts():
    context = ExecutionContext(total_rows=3)
    context.record_failed_rows([1], stage="ResponseParser", message="not JSON")
    context.record_skipped_rows(2, 1, stage="LLMInvocation", message="down")

    data = context.to_checkpoint()
    restored = ExecutionContext.from_checkpoint(data)

    assert (data["failed_rows"], data["skipped_rows"]) == (1, 1)
    assert (restored.failed_rows, restored.skipped_rows) == (1, 1)
    assert restored.row_errors == []
    assert restored.to_checkpoint()["failed_rows"] == 1


def test_a_checkpoint_without_counts_carries_none():
    data = ExecutionContext(total_rows=1).to_checkpoint()
    del data["failed_rows"], data["skipped_rows"]

    restored = ExecutionContext.from_checkpoint(data)

    assert (restored.failed_rows, restored.skipped_rows) == (0, 0)


@pytest.mark.parametrize("counter", ["failed_rows", "skipped_rows"])
def test_assigning_a_count_still_works_but_is_deprecated(counter):
    """Custom stages that did `context.failed_rows += n` keep working.

    ExecutionContext is public, so the old write path stays for now; it warns,
    and the value read back is the value assigned, on top of what the ledger
    has counted.
    """
    context = ExecutionContext(total_rows=4)
    context.record_failed_rows([0], stage="s", message="m")
    before = getattr(context, counter)

    with pytest.warns(DeprecationWarning, match=counter):
        setattr(context, counter, before + 2)

    assert getattr(context, counter) == before + 2


def test_record_answers_is_one_batch_answer_on_the_ledger():
    """The context reports to the ledger; the ledger shares the call's cost."""
    context = ExecutionContext(total_rows=3)
    book = RowLedger(3, context.ledger.policy)

    context.record_answers([2, 0], model="m", tokens=5, cost=Decimal("0.0003"))
    book.record_batch_answer([2, 0], 1, model="m", tokens=5, cost=Decimal("0.0003"))

    assert context.ledger.snapshot().rows == book.snapshot().rows
