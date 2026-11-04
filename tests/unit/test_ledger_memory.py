"""What the row ledger keeps in memory per answered row, and that it keeps it exactly.

The ledger is opened on every run, so its cost per row is paid by every client,
on every row: at 5M rows, each byte kept per row is 5 MB. The common row is
answered once and judged once (or, without a contract, not judged at all), and
the ledger must keep that row in a few bytes, not in a record of objects. Rows
that need more (several attempts, a failure, a merge conflict) may cost more:
they are the exceptions, and the ledger must keep everything they need.

The byte figures are measured with ``tracemalloc`` over every allocation the
ledger makes while the rows are reported, divided by the rows answered.
"""

from __future__ import annotations

import gc
import tracemalloc
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from ondine.harness.contract import OutputField, build_contract
from ondine.harness.judge import judge
from ondine.harness.ledger import LedgerPolicy, RowLedger
from ondine.harness.status import RowStatus
from ondine.orchestration.execution_context import ExecutionContext

if TYPE_CHECKING:
    from collections.abc import Callable

ROWS = 20_000
UNCHECKED = build_contract(["answer"])
INTENT = OutputField("intent", allowed=["refund", "cancel"], normalise=["casefold"])
CHECKED = build_contract(["intent"], fields=[INTENT])


def _bytes_per_row(build: Callable[[], object]) -> float:
    """Bytes still allocated after ``build`` returns, per row of ``ROWS``."""
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        kept = build()
        gc.collect()
        after = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    assert kept is not None
    return (after - before) / ROWS


def _costs() -> list[Decimal]:
    """A different cost on every row, as token-priced calls give."""
    return [Decimal(f"0.000{100 + i % 900}{i % 7}") for i in range(ROWS)]


def test_an_unchecked_answered_row_costs_under_50_bytes():
    costs = _costs()

    def run() -> ExecutionContext:
        context = ExecutionContext(total_rows=ROWS)
        context.open_ledger(UNCHECKED)
        for row in range(ROWS):
            context.record_answers(
                [row], model="gpt-4o-mini", tokens=100 + row % 50, cost=costs[row]
            )
        return context

    assert _bytes_per_row(run) < 50


def test_a_batch_answered_row_costs_under_50_bytes():
    """A mega-batch shares one call's cost; the shares must stay compact too."""
    costs = _costs()

    def run() -> ExecutionContext:
        context = ExecutionContext(total_rows=ROWS)
        context.open_ledger(UNCHECKED)
        for first in range(0, ROWS, 10):
            context.record_answers(
                range(first, first + 10),
                model="gpt-4o-mini",
                tokens=1000 + first % 13,
                cost=costs[first] * 7,
            )
        return context

    assert _bytes_per_row(run) < 50


def test_a_checked_answered_and_judged_row_costs_under_50_bytes():
    """Accepted and repaired rows: one answer, one verdict, nothing else."""
    costs = _costs()
    verdicts = [judge(CHECKED, text) for text in ("refund", "They want to Cancel.")]

    def run() -> RowLedger:
        book = RowLedger(ROWS, LedgerPolicy.from_contract(CHECKED))
        for row in range(ROWS):
            book.record_answer(row, 1, model="m", tokens=100, cost=costs[row])
            book.record_verdict(row, 1, verdicts[row % 2], stage="ResponseParser")
        return book

    assert _bytes_per_row(run) < 50


def test_lost_rows_cost_little_too():
    """A provider down for the whole run loses every row; the counts stay exact."""

    def run() -> RowLedger:
        book = RowLedger(ROWS, LedgerPolicy.from_contract(UNCHECKED))
        for first in range(0, ROWS, 100):
            book.record_provider_failure(
                range(first, first + 100), 1, stage="LLM", message="503"
            )
        return book

    assert _bytes_per_row(run) < 50


# ---------------------------------------------------------------------------
# Kept compactly, read back exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cost",
    [
        Decimal("0.01"),
        Decimal("0.0100"),
        Decimal("0"),
        Decimal("0E-9"),
        Decimal("-0"),
        Decimal("-0.25"),
        Decimal("12345678901234567.89"),
        Decimal("0.003333333333333333333333333333"),
        Decimal("1E+300"),
        Decimal("1E-300"),
        Decimal("NaN"),
    ],
    ids=str,
)
def test_an_answers_cost_reads_back_exactly(cost):
    book = RowLedger(1, LedgerPolicy.from_contract(UNCHECKED), strict=True)

    book.record_answer(0, 1, model="m", tokens=1, cost=cost)

    kept = book.snapshot().rows[0].attempts[0].cost
    assert str(kept) == str(cost)


@pytest.mark.parametrize("tokens", [0, 7, 2**63 - 1, 2**63, -(2**70)])
def test_an_answers_tokens_read_back_exactly(tokens):
    book = RowLedger(1, LedgerPolicy.from_contract(UNCHECKED), strict=True)

    book.record_answer(0, 1, model="m", tokens=tokens, cost=Decimal("0.01"))

    assert book.snapshot().rows[0].attempts[0].tokens == tokens


def test_an_answer_on_a_very_high_attempt_reads_back_exactly():
    book = RowLedger(1, LedgerPolicy.from_contract(UNCHECKED), strict=True)

    book.record_answer(0, 70_000, model="m", tokens=1, cost=Decimal("0.01"))

    row = book.snapshot().rows[0]
    assert row.delivered_attempt == 70_000
    assert [a.attempt for a in row.attempts] == [70_000]


def test_many_models_each_read_back_as_the_model_that_answered():
    rows = 70_000
    book = RowLedger(rows, LedgerPolicy.from_contract(UNCHECKED))

    for row in range(rows):
        book.record_answer(row, 1, model=f"model-{row}", tokens=1, cost=Decimal(0))

    snapshot = book.snapshot()
    assert snapshot.rows[5].delivered_by == "model-5"
    assert snapshot.rows[rows - 1].delivered_by == f"model-{rows - 1}"


def test_a_raw_answer_is_kept_for_a_row_that_later_fails():
    """The answer arrived first, compactly; the failure arrives after it."""
    book = RowLedger(1, LedgerPolicy.from_contract(CHECKED), strict=True)
    book.record_answer(0, 1, model="m", tokens=1, cost=Decimal(0), raw_answer="key-0")

    book.record_verdict(0, 1, judge(CHECKED, "maybe"), stage="ResponseParser")

    row = book.snapshot().rows[0]
    assert row.status is RowStatus.REJECTED
    assert row.attempts[0].raw_answer == "key-0"
