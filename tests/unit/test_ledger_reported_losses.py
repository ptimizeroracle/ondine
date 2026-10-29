"""The ledger's reading of today's loss bookkeeping, for migration step 1.

Until the harness wiring is complete, ``ExecutionContext.row_errors``,
``failed_rows`` and ``skipped_rows`` must say exactly what they said before
the ledger existed: the parity net (``tests/e2e/test_harness_parity.py``)
freezes them. The ledger's own snapshot view decides differently on purpose
(row order, the provider outranking a stage, recoveries, rows never reached),
so these tests pin the *other* view, :meth:`RowLedger.reported_losses`, which
reads the same events the way ``ExecutionContext`` used to keep them:

* a row is lost the first time a failure is reported for it, and that first
  report is its cause;
* the error list is in report order, bounded, while the counts stay exact;
* only failures reported to this ledger count — a sub-pipeline's, arriving by
  merge, never reached today's context either;
* a row nobody reported on is not in it.
"""

from __future__ import annotations

from decimal import Decimal

from ondine.harness.contract import build_contract
from ondine.harness.judge import judge
from ondine.harness.ledger import LedgerPolicy, RowLedger, StopReason
from ondine.harness.status import RowStatus

UNCHECKED = LedgerPolicy.from_contract(build_contract(["answer"]))


def _shape(errors) -> list[tuple[int, str, str, str]]:
    return [(e.row_index, e.stage, e.error_type, e.message) for e in errors]


def _answer(book: RowLedger, row: int) -> None:
    book.record_answer(row, 1, model="m", tokens=3, cost=Decimal("0.01"))


def test_errors_keep_report_order_not_row_order():
    """Today's list is append-ordered: a later stage's loss comes after."""
    book = RowLedger(6, UNCHECKED, strict=True)
    for row in (1, 3, 4):
        _answer(book, row)
    book.record_stage_failure([5], stage="PromptFormatter", message="bad row")
    book.record_provider_failure([2, 0], 1, stage="LLMInvocation", message="down")

    losses = book.reported_losses()

    assert _shape(losses.errors) == [
        (5, "PromptFormatter", "parse_failure", "bad row"),
        (2, "LLMInvocation", "skipped", "down"),
        (0, "LLMInvocation", "skipped", "down"),
    ]
    assert (losses.skipped_rows, losses.failed_rows) == (2, 1)
    # The snapshot's own view stays row-ordered.
    assert [e.row_index for e in book.snapshot().errors] == [0, 2, 5]


def test_the_first_report_is_the_cause_even_when_the_snapshot_disagrees():
    """A stage failure reported before a provider failure keeps the row.

    ``ExecutionContext`` counted a row lost once, under its first report. The
    ledger's status rule ranks the provider above the stage instead; the two
    views are allowed to differ, and this one must keep today's answer.
    """
    book = RowLedger(4, UNCHECKED, strict=True)
    book.record_stage_failure([2], stage="ResponseParser", message="not JSON")
    book.record_provider_failure(range(4), 1, stage="LLM", message="batch failed")

    losses = book.reported_losses()

    assert (losses.skipped_rows, losses.failed_rows) == (3, 1)
    assert _shape(losses.errors) == [
        (2, "ResponseParser", "parse_failure", "not JSON"),
        (0, "LLM", "skipped", "batch failed"),
        (1, "LLM", "skipped", "batch failed"),
        (3, "LLM", "skipped", "batch failed"),
    ]
    assert book.snapshot().skipped_rows == 4


def test_a_second_report_for_a_lost_row_is_not_counted_again():
    book = RowLedger(2, UNCHECKED, strict=True)
    book.record_provider_failure([1], 1, stage="LLM", message="provider down")
    book.record_stage_failure([1], stage="ResponseParser", message="not JSON")

    losses = book.reported_losses()

    assert (losses.skipped_rows, losses.failed_rows) == (1, 0)
    assert [e.message for e in losses.errors] == ["provider down"]


def test_the_error_list_is_bounded_in_report_order_and_the_counts_stay_exact():
    book = RowLedger(1005, UNCHECKED, strict=True)
    book.record_stage_failure([1004], stage="PromptFormatter", message="first")
    book.record_provider_failure(range(1004), 1, stage="LLM", message="down")

    losses = book.reported_losses(limit=3)

    assert [e.row_index for e in losses.errors] == [1004, 0, 1]
    assert (losses.skipped_rows, losses.failed_rows) == (1004, 1)
    assert len(book.reported_losses().errors) == RowLedger.MAX_RECORDED_ROW_ERRORS


def test_a_limit_of_zero_reads_the_counts_alone():
    book = RowLedger(3, UNCHECKED, strict=True)
    book.record_provider_failure([0, 1], 1, stage="LLM", message="down")

    losses = book.reported_losses(limit=0)

    assert losses.errors == ()
    assert losses.skipped_rows == 2


def test_errors_carry_the_report_time_and_are_not_retryable():
    book = RowLedger(1, UNCHECKED, strict=True)
    book.record_provider_failure([0], 1, stage="LLM", message="down")

    first = book.reported_losses().errors[0]
    again = book.reported_losses().errors[0]

    assert first.retryable is False
    assert first.timestamp == again.timestamp


def test_failures_arriving_by_merge_are_not_reported_losses():
    """Today a sub-pipeline's losses never reached the parent's context."""
    child = RowLedger(2, UNCHECKED, strict=True)
    child.record_provider_failure([0], 1, stage="LLM", message="chunk down")
    _answer(child, 1)
    parent = RowLedger(4, UNCHECKED, strict=True)
    _answer(parent, 1)
    parent.record_provider_failure([0], 1, stage="LLM", message="own loss")

    parent.merge(child, [2, 3])

    losses = parent.reported_losses()
    assert _shape(losses.errors) == [(0, "LLM", "skipped", "own loss")]
    assert parent.snapshot().lost_row_indices == [0, 2]


def test_a_row_recovered_by_merge_stays_a_reported_loss():
    """Today the auto-retry pass never edited the error list it recovered."""
    retry = RowLedger(1, UNCHECKED, strict=True)
    _answer(retry, 0)
    parent = RowLedger(2, UNCHECKED, strict=True)
    _answer(parent, 0)
    parent.record_stage_failure([1], stage="PromptFormatter", message="bad row")

    parent.merge(retry, [1], retry=True)

    assert [e.row_index for e in parent.reported_losses().errors] == [1]
    assert parent.snapshot().rows[1].status is RowStatus.UNCHECKED


def test_rows_never_reached_are_not_reported_losses():
    """The snapshot blames them on the stop; today nothing recorded them."""
    book = RowLedger(3, UNCHECKED, strict=True)
    _answer(book, 0)
    book.record_run_stopped(StopReason.BUDGET_CAP)

    assert book.reported_losses().errors == ()
    snapshot = book.snapshot()
    assert snapshot.lost_row_indices == [1, 2]
    assert snapshot.rows[1].reason == "run stopped before this row: budget cap"


def test_an_unparseable_answer_reported_as_a_verdict_reads_as_today():
    """Step 2 moves the parse step's failures into verdicts; same reading."""
    contract = build_contract(["intent", "amount"], response_format="json")
    book = RowLedger(1, LedgerPolicy.from_contract(contract), strict=True)
    _answer(book, 0)

    book.record_verdict(0, 1, judge(contract, "not json"), stage="ResponseParser")

    losses = book.reported_losses()
    snapshot = book.snapshot()
    assert _shape(losses.errors) == _shape(snapshot.errors)
    assert (losses.skipped_rows, losses.failed_rows) == (0, 1)


def test_a_passing_verdict_is_not_a_reported_loss():
    contract = build_contract(["answer"])
    book = RowLedger(1, LedgerPolicy.from_contract(contract), strict=True)

    book.record_verdict(0, 1, judge(contract, "fine"), stage="ResponseParser")

    assert book.reported_losses().errors == ()


# ---------------------------------------------------------------------------
# widen: a context driven outside a pipeline learns its size late
# ---------------------------------------------------------------------------


def test_widen_adds_pending_rows_and_never_shrinks():
    book = RowLedger(1, UNCHECKED, strict=True)
    _answer(book, 0)

    book.widen(3)
    book.widen(2)

    assert book.row_count == 3
    book.record_provider_failure([2], 1, stage="LLM", message="down")
    snapshot = book.snapshot()
    assert snapshot.lost_row_indices == [1, 2]
    assert snapshot.rows[1].reason == "run stopped before this row"


def test_the_policy_is_readable():
    assert RowLedger(0, UNCHECKED).policy == UNCHECKED
