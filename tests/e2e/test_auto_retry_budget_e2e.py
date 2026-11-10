"""End-to-end: auto-retry spends within ``with_max_budget()``, never past it.

Each auto-retry pass is a sub-pipeline. It used to copy the run's specs with
``max_budget`` unchanged, so every pass got the whole budget again and a run
with auto-retry could spend up to ``(1 + max_retry_attempts)`` times its cap.

A pass now asks only the rows the remaining budget (``max_budget`` minus what
the run has spent) can pay for, estimated from the run's cost per row so far,
and the pass itself runs uncapped, so it never stops halfway and loses
answers already paid for. When nothing is affordable, auto-retry stops.

Every test runs the public API against the conformance provider: each call
costs exactly ``COST_PER_CALL`` ($0.0001), so every spend below is exact.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from ondine import PipelineBuilder
from tests.e2e.conformance_harness import (
    COST_PER_CALL,
    LedgerClient,
    answer_for,
    conformance_frame,
    expected_answers,
    token_for,
)

PROMPT = "Answer for {marker}"


class BlankFirstClient(LedgerClient):
    """Answers blank on each row's first ``blank_times`` calls, then for real.

    A blank answer is paid for like any other call, and auto-retry re-runs
    the row because its output cell is empty.
    """

    def __init__(self, *, blank_times: int = 1) -> None:
        super().__init__()
        self._blank_times = blank_times

    def _answer_token(self, token: str) -> str:
        if self._record(token) <= self._blank_times:
            return ""
        return answer_for(token)


def build(
    client: LedgerClient,
    frame: pd.DataFrame,
    *,
    max_budget: float | None,
    attempts: int = 1,
    checked: bool = False,
):
    """A pipeline over ``frame`` with auto-retry on and ``attempts`` passes.

    ``checked`` declares a contract whose allowed values are the rows' own
    answers: the ledger, not the empty cells, then picks the rows to retry.
    """
    builder = (
        PipelineBuilder.create()
        .from_dataframe(frame, input_columns=["marker"], output_columns=["answer"])
        .with_prompt(PROMPT)
        .with_custom_llm_client(client)
    )
    if checked:
        builder = builder.with_output_field(
            "answer", allowed=expected_answers(len(frame))
        )
    if max_budget is not None:
        builder = builder.with_max_budget(max_budget)
    pipeline = builder.build()
    pipeline.specifications.processing.auto_retry_failed = True
    pipeline.specifications.processing.max_retry_attempts = attempts
    return pipeline


def calls(client: LedgerClient, rows: int) -> list[int]:
    """How many calls each row of a ``rows``-row frame received, in order."""
    return [client.calls[token_for(row)] for row in range(rows)]


def answers(result) -> list:
    frame = result.to_pandas()
    return [None if pd.isna(value) else value for value in frame["answer"]]


def test_auto_retry_spends_within_the_budget():
    """10 blank rows cost $0.0010; a $0.0015 cap leaves room for 4 retries.

    Before the fix the retry pass got a fresh $0.0015 and retried all 10
    rows: $0.0020 spent against a $0.0015 cap.
    """
    client = BlankFirstClient()

    result = build(client, conformance_frame(10), max_budget=0.0015).execute()

    assert result.costs.total_cost <= Decimal("0.0015")
    # The pass still runs: it retries the leading rows the rest can pay for
    # (estimated at $0.0001 per row, with a 25% margin: 4 rows).
    assert result.costs.total_cost == 14 * COST_PER_CALL
    assert calls(client, 10) == [2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
    assert answers(result)[:4] == [answer_for(token_for(i)) for i in range(4)]
    # A retried row's answer was paid for and kept: nothing lost halfway.
    assert all(value in (None, "") for value in answers(result)[4:])


def test_budget_spent_before_any_retry_retries_nothing():
    """The first pass spends the whole cap: auto-retry stops, cleanly.

    No exception, no extra call, and the result is the first pass's.
    """
    client = BlankFirstClient()

    result = build(client, conformance_frame(5), max_budget=0.0005).execute()

    assert result.costs.total_cost == 5 * COST_PER_CALL
    assert calls(client, 5) == [1] * 5
    assert all(value in (None, "") for value in answers(result))


def test_each_retry_attempt_runs_within_what_is_left():
    """Three attempts share one cap; each asks what the rest can pay for.

    4 rows blank twice, cap $0.0011:
    first pass $0.0004; attempt 1 retries all 4 ($0.0008, still blank);
    attempt 2 can pay for 2 ($0.0010, rows 0 and 1 answered);
    attempt 3 has $0.0001 left, under one row's estimate, so it stops.
    Before the fix every attempt got the whole cap: $0.0012 spent.
    """
    client = BlankFirstClient(blank_times=2)

    result = build(
        client, conformance_frame(4), max_budget=0.0011, attempts=3
    ).execute()

    assert result.costs.total_cost <= Decimal("0.0011")
    assert result.costs.total_cost == 10 * COST_PER_CALL
    assert calls(client, 4) == [3, 3, 2, 2]
    assert answers(result)[:2] == [answer_for(token_for(i)) for i in range(2)]
    assert all(value in (None, "") for value in answers(result)[2:])


def test_without_a_budget_auto_retry_is_unchanged():
    """No cap: every failed row is retried, every attempt, as before."""
    client = BlankFirstClient(blank_times=2)

    result = build(client, conformance_frame(6), max_budget=None, attempts=2).execute()

    assert calls(client, 6) == [3] * 6
    assert result.costs.total_cost == 18 * COST_PER_CALL
    assert answers(result) == [answer_for(token_for(i)) for i in range(6)]


def test_free_provider_failures_do_not_shrink_the_estimate():
    """A provider error costs nothing; the estimate is per paid row.

    Under a contract, where the ledger picks the failed rows. (Without one,
    auto-retry reads a provider-skipped row's ``[SKIPPED]`` cell as an answer
    and never retries it: a separate gap.)

    6 rows, rows 3-5 raise on their first call: the first pass pays for 3
    rows ($0.0003). Per paid row that is $0.0001; averaged over all 6 rows
    it would be $0.00005, and retrying all 3 would cost $0.0006 against a
    $0.00055 cap. The $0.00025 left pays for 2 retries.
    """
    failing = {token_for(row) for row in (3, 4, 5)}
    client = LedgerClient(fail_tokens=failing, fail_times=1)

    result = build(
        client, conformance_frame(6), max_budget=0.00055, checked=True
    ).execute()

    assert result.costs.total_cost <= Decimal("0.00055")
    assert result.costs.total_cost == 5 * COST_PER_CALL
    assert calls(client, 6) == [1, 1, 1, 2, 2, 1]
    assert answers(result)[:5] == [answer_for(token_for(i)) for i in range(5)]
    assert answers(result)[5] is None
