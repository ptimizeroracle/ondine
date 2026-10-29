"""The row ledger: one status per input row, whatever order the events arrive in.

The ledger is the single owner of every row's fate. Stages report events to it
(an answer arrived, the judge's verdict, a provider or stage failure, the run
stopping, a sub-pipeline's ledger to merge) and it alone turns them into one
status per row. Each test here pins one of the rules in the interface spec
(harness-F1F2-interfaces.md, section 3), because a rule broken here is a row
delivered, blanked or counted wrongly in every client's dataframe.

The verdicts below are the judge's own types (``Verdict``, ``FieldVerdict``,
``Reason``), built by hand so each test states exactly the verdict it means.
There is one vocabulary: the ledger reads what the judge returns, and
``test_judge_ledger_seam.py`` runs the real judge into the real ledger.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from ondine.harness.contract import OutputField, RunPolicy, build_contract
from ondine.harness.judge import FieldVerdict, Reason, Verdict
from ondine.harness.ledger import (
    LedgerInvariantError,
    LedgerPolicy,
    RowLedger,
    StopReason,
)
from ondine.harness.status import RowStatus

if TYPE_CHECKING:
    from collections.abc import Callable

#: The stage that reports verdicts in these tests: today's parse step.
PARSER = "ResponseParser"

# ---------------------------------------------------------------------------
# Verdicts, in the judge's own types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """One field of a hand-built verdict, before it is given its name."""

    outcome: str = "value"
    reasons: tuple[str, ...] = ()
    format_fixed: bool = False
    required: bool = True


PASS = FieldSpec()
ABSENT = FieldSpec(outcome="absent", required=False)


def failing(reason: str, *, required: bool = True) -> FieldSpec:
    return FieldSpec(outcome="failed", reasons=(reason,), required=required)


def fixed(reason: str = "") -> FieldSpec:
    return FieldSpec(format_fixed=True, reasons=(reason,) if reason else ())


def verdict(*, checked: bool = True, **fields: FieldSpec) -> Verdict:
    """A judge ``Verdict`` with one ``FieldVerdict`` per keyword, in order."""
    return Verdict(
        checked=checked,
        fields=tuple(
            FieldVerdict(
                name=name,
                outcome=spec.outcome,
                value=None if spec.outcome == "failed" else f"{name}-value",
                reasons=tuple(Reason(name, "allowed_values", r) for r in spec.reasons),
                format_fixed=spec.format_fixed,
                required=spec.required,
            )
            for name, spec in fields.items()
        ),
    )


def unchecked_verdict(**fields: FieldSpec) -> Verdict:
    """What the judge returns under the unchecked contract."""
    return verdict(checked=False, **fields)


def policy(**run: object) -> LedgerPolicy:
    """The ledger policy of a checked contract whose run policy is ``run``."""
    contract = build_contract(
        ["intent"], fields=[OutputField("intent")], policy=RunPolicy(**run)
    )
    return LedgerPolicy.from_contract(contract)


CHECKED = policy()
FLAGGED = policy(on_failure="deliver_flagged")
UNCHECKED = LedgerPolicy.from_contract(build_contract(["answer"]))


def ledger(rows: int, policy: LedgerPolicy = CHECKED) -> RowLedger:
    """A strict ledger: a broken invariant raises here instead of being logged."""
    return RowLedger(rows, policy, strict=True)


def answer(book: RowLedger, row: int, attempt: int = 1, model: str = "cheap") -> None:
    book.record_answer(
        row,
        attempt,
        model=model,
        tokens=10,
        cost=Decimal("0.01"),
        raw_answer=f"raw-{row}-{attempt}",
    )


def judged(book: RowLedger, row: int, attempt: int, result: Verdict) -> None:
    """An answer and its verdict for one attempt: the common case."""
    answer(book, row, attempt)
    book.record_verdict(row, attempt, result, stage=PARSER)


def status(book: RowLedger, row: int) -> RowStatus:
    return book.snapshot().rows[row].status


# ---------------------------------------------------------------------------
# Creation and the pending -> failed rule
# ---------------------------------------------------------------------------


def test_every_row_starts_pending_and_the_counts_say_so():
    book = ledger(4)

    assert [book.outcome(row).status for row in range(4)] == [RowStatus.PENDING] * 4
    assert book.counts()[RowStatus.PENDING] == 4


def test_a_row_never_reached_is_failed_at_snapshot_not_left_pending():
    """No code path can leave a row without a final status (define it away)."""
    book = ledger(3)
    judged(book, 0, 1, verdict(intent=PASS))

    snapshot = book.snapshot()

    assert [row.status for row in snapshot.rows] == [
        RowStatus.ACCEPTED,
        RowStatus.FAILED,
        RowStatus.FAILED,
    ]
    assert "run stopped before this row" in snapshot.rows[1].reason
    assert RowStatus.PENDING not in snapshot.counts


def test_a_budget_stop_at_row_120_of_200_fails_the_rest_and_totals_still_add_up():
    """Acceptance test 8: the rows after the stop are failed, with the cause."""
    book = ledger(200)
    for row in range(120):
        judged(book, row, 1, verdict(intent=PASS))
    book.record_run_stopped(StopReason.BUDGET_CAP)

    snapshot = book.snapshot()

    assert snapshot.counts[RowStatus.ACCEPTED] == 120
    assert snapshot.counts[RowStatus.FAILED] == 80
    assert sum(snapshot.counts.values()) == 200
    stopped = snapshot.rows[120]
    assert stopped.status is RowStatus.FAILED
    assert "run stopped before this row" in stopped.reason
    assert "budget cap" in stopped.reason
    assert snapshot.stop_reason is StopReason.BUDGET_CAP


def test_counts_cover_every_final_status_and_always_sum_to_the_row_count():
    book = ledger(6)
    judged(book, 0, 1, verdict(intent=PASS))
    judged(book, 1, 1, verdict(intent=fixed()))
    judged(book, 2, 1, verdict(intent=failing("intent: not allowed")))
    book.record_provider_failure([3], 1, stage="LLMInvocation", message="503")

    counts = book.snapshot().counts

    assert set(counts) == {
        RowStatus.ACCEPTED,
        RowStatus.REPAIRED,
        RowStatus.REJECTED,
        RowStatus.UNVERIFIED,
        RowStatus.FAILED,
        RowStatus.UNCHECKED,
    }
    assert counts[RowStatus.FAILED] == 3  # row 3, plus rows 4 and 5 never reached
    assert sum(counts.values()) == 6


# ---------------------------------------------------------------------------
# Status rules
# ---------------------------------------------------------------------------


def test_a_deliverable_first_answer_is_accepted_and_names_its_model():
    book = ledger(1)
    answer(book, 0, 1, model="haiku")
    book.record_verdict(0, 1, verdict(intent=PASS), stage=PARSER)

    row = book.snapshot().rows[0]

    assert row.status is RowStatus.ACCEPTED
    assert row.reason == ""
    assert row.delivered_attempt == 1
    assert row.delivered_by == "haiku"


def test_a_format_fix_on_a_required_field_makes_the_row_repaired():
    book = ledger(1)
    judged(book, 0, 1, verdict(intent=fixed("intent: wordy answer")))

    row = book.snapshot().rows[0]

    assert row.status is RowStatus.REPAIRED
    assert "format fix" in row.reason


def test_a_later_attempt_that_passes_after_a_failed_check_is_repaired():
    book = ledger(1)
    judged(book, 0, 1, verdict(intent=failing("intent: 'maybe' is not allowed")))
    judged(book, 0, 2, verdict(intent=PASS))

    row = book.snapshot().rows[0]

    assert row.status is RowStatus.REPAIRED
    assert row.delivered_attempt == 2
    assert "intent: 'maybe' is not allowed" in row.reason


def test_a_failed_check_rejects_the_row_with_the_check_as_reason():
    book = ledger(1)
    judged(book, 0, 1, verdict(intent=failing("intent: 'maybe' is not allowed")))

    row = book.snapshot().rows[0]

    assert row.status is RowStatus.REJECTED
    assert "intent: 'maybe' is not allowed" in row.reason
    assert row.failed_checks == ("intent: 'maybe' is not allowed",)


def test_deliver_flagged_turns_the_same_failure_into_unverified():
    book = ledger(1, FLAGGED)
    judged(book, 0, 1, verdict(intent=failing("intent: 'maybe' is not allowed")))

    row = book.snapshot().rows[0]

    assert row.status is RowStatus.UNVERIFIED
    assert row.delivered_attempt == 1
    assert "intent: 'maybe' is not allowed" in row.reason


def test_the_row_status_is_its_worst_required_field():
    book = ledger(2)
    judged(book, 0, 1, verdict(intent=PASS, amount=fixed()))
    judged(book, 1, 1, verdict(intent=fixed(), amount=failing("amount: below 0")))

    rows = book.snapshot().rows

    assert rows[0].status is RowStatus.REPAIRED
    assert dict(rows[0].field_statuses) == {
        "intent": RowStatus.ACCEPTED,
        "amount": RowStatus.REPAIRED,
    }
    assert rows[1].status is RowStatus.REJECTED
    assert dict(rows[1].field_statuses)["amount"] is RowStatus.REJECTED


def test_a_failed_optional_field_is_named_but_does_not_decide_the_status():
    book = ledger(1)
    note = failing("note: longer than 200 characters", required=False)
    judged(book, 0, 1, verdict(intent=PASS, note=note))

    row = book.snapshot().rows[0]

    assert row.status is RowStatus.ACCEPTED
    assert "note: longer than 200 characters" in row.reason


def test_a_provider_failure_with_no_verdict_is_failed():
    book = ledger(1)
    book.record_provider_failure(
        [0], 1, stage="LLMInvocation", message="401 invalid key"
    )

    row = book.snapshot().rows[0]

    assert row.status is RowStatus.FAILED
    assert "401 invalid key" in row.reason


def test_a_stage_failure_with_no_verdict_is_failed():
    book = ledger(2)
    book.record_stage_failure(
        [0, 1], stage="PromptFormatter", message="prompt could not be built"
    )

    snapshot = book.snapshot()

    assert [row.status for row in snapshot.rows] == [RowStatus.FAILED] * 2
    assert "prompt could not be built" in snapshot.rows[1].reason


def test_a_repair_cut_short_by_the_provider_stays_rejected_with_a_provider_sub_reason():
    """The answer existed and failed a check: that is a rejection, not a loss."""
    book = ledger(1)
    judged(book, 0, 1, verdict(intent=failing("intent: 'maybe' is not allowed")))
    book.record_provider_failure([0], 2, stage="LLMInvocation", message="timeout")

    row = book.snapshot().rows[0]

    assert row.status is RowStatus.REJECTED
    assert row.sub_reason == "provider"
    assert "timeout" in row.reason
    assert "intent: 'maybe' is not allowed" in row.reason


def test_a_provider_failure_older_than_the_failed_verdict_is_not_a_sub_reason():
    book = ledger(1)
    book.record_provider_failure([0], 1, stage="LLMInvocation", message="timeout")
    judged(book, 0, 2, verdict(intent=failing("intent: 'maybe' is not allowed")))

    row = book.snapshot().rows[0]

    assert row.status is RowStatus.REJECTED
    assert row.sub_reason is None


@pytest.mark.parametrize("success_first", [True, False])
def test_an_older_failure_never_overwrites_a_newer_success(success_first):
    """Ordering is by attempt number, never by which event landed last."""
    book = ledger(1)

    def success() -> None:
        judged(book, 0, 2, verdict(intent=PASS))

    def failure() -> None:
        book.record_provider_failure([0], 1, stage="LLMInvocation", message="503")

    for event in (success, failure) if success_first else (failure, success):
        event()

    row = book.snapshot().rows[0]
    assert row.status is RowStatus.ACCEPTED
    assert row.delivered_attempt == 2


@pytest.mark.parametrize("newest_first", [True, False])
def test_the_latest_attempt_decides_whatever_order_the_verdicts_arrive_in(
    newest_first,
):
    book = ledger(1)
    older = verdict(intent=PASS)
    newer = verdict(intent=failing("intent: 'maybe' is not allowed"))
    order = [(2, newer), (1, older)] if newest_first else [(1, older), (2, newer)]

    for attempt, result in order:
        judged(book, 0, attempt, result)

    assert status(book, 0) is RowStatus.REJECTED


def test_unverified_delivers_the_attempt_with_the_fewest_failed_checks():
    book = ledger(2, FLAGGED)
    two_failures = verdict(intent=failing("intent: a"), amount=failing("amount: b"))
    one_failure = verdict(intent=failing("intent: a"), amount=PASS)
    judged(book, 0, 1, one_failure)
    judged(book, 0, 2, two_failures)
    # A tie goes to the highest attempt.
    judged(book, 1, 1, one_failure)
    judged(book, 1, 2, one_failure)

    rows = book.snapshot().rows

    assert rows[0].status is RowStatus.UNVERIFIED
    assert rows[0].delivered_attempt == 1
    assert rows[1].delivered_attempt == 2


def test_an_answer_never_judged_under_a_contract_is_failed():
    """A required check that never ran cannot vouch for the value."""
    book = ledger(1)
    answer(book, 0, 1)

    row = book.snapshot().rows[0]

    assert row.status is RowStatus.FAILED
    assert "never judged" in row.reason


# ---------------------------------------------------------------------------
# No contract: unchecked rows and today's views
# ---------------------------------------------------------------------------


def test_without_a_contract_an_answered_row_is_unchecked():
    book = ledger(2, UNCHECKED)
    answer(book, 0, 1)
    judged(book, 1, 1, unchecked_verdict(answer=PASS))

    assert [row.status for row in book.snapshot().rows] == [
        RowStatus.UNCHECKED,
        RowStatus.UNCHECKED,
    ]


def test_without_a_contract_an_unparseable_answer_is_failed_not_rejected():
    """No check was declared, so nothing can be rejected; it is a lost row."""
    book = ledger(1, UNCHECKED)
    judged(
        book, 0, 1, unchecked_verdict(answer=failing("response could not be parsed"))
    )

    snapshot = book.snapshot()

    assert snapshot.rows[0].status is RowStatus.FAILED
    assert [(e.row_index, e.error_type) for e in snapshot.errors] == [
        (0, "parse_failure")
    ]


def test_views_without_a_contract_match_what_the_context_reports_today():
    """The #262/#312 guard: same errors, lost rows, counts and completeness.

    Mirrors ExecutionContext: a provider failure under the skip policy is a
    ``skipped`` error, a stage failure a ``parse_failure``, and a row lost to
    the provider and then tripped over by the parser is lost once, to the
    provider, whichever report arrives first.
    """
    book = ledger(5, UNCHECKED)
    for row in (0, 2):
        answer(book, row, 1)
    book.record_stage_failure([4], stage="ResponseParser", message="not JSON")
    book.record_provider_failure(
        [3], 1, stage="LLMInvocationStage", message="rate limited"
    )
    book.record_stage_failure([3], stage="ResponseParser", message="not JSON")
    book.record_provider_failure(
        [1], 1, stage="LLMInvocationStage", message="401 bad key"
    )

    snapshot = book.snapshot()

    assert [
        (e.row_index, e.stage, e.error_type, e.message) for e in snapshot.errors
    ] == [
        (1, "LLMInvocationStage", "skipped", "401 bad key"),
        (3, "LLMInvocationStage", "skipped", "rate limited"),
        (4, "ResponseParser", "parse_failure", "not JSON"),
    ]
    assert snapshot.lost_row_indices == [1, 3, 4]
    assert snapshot.skipped_rows == 2
    assert snapshot.failed_rows == 1
    assert snapshot.is_complete is False


def test_a_clean_run_without_a_contract_is_complete_with_no_errors():
    book = ledger(3, UNCHECKED)
    for row in range(3):
        answer(book, row, 1)

    snapshot = book.snapshot()

    assert snapshot.errors == ()
    assert snapshot.lost_row_indices == []
    assert snapshot.is_complete is True


def test_the_error_list_is_bounded_but_the_counts_stay_exact():
    """Same bound as ExecutionContext.MAX_RECORDED_ROW_ERRORS, same reason."""
    assert RowLedger.MAX_RECORDED_ROW_ERRORS == 1000
    book = ledger(1005, UNCHECKED)
    book.record_provider_failure(range(1005), 1, stage="LLM", message="down")

    snapshot = book.snapshot()

    assert len(snapshot.errors) == 1000
    assert snapshot.errors[0].row_index == 0
    assert snapshot.skipped_rows == 1005
    assert len(snapshot.lost_row_indices) == 1005


def test_rejected_rows_get_their_own_list_and_break_completeness():
    """#312: a rejected row is not a lost row, and not a complete one either."""
    book = ledger(3)
    judged(book, 0, 1, verdict(intent=PASS))
    judged(book, 1, 1, verdict(intent=failing("intent: no")))
    book.record_provider_failure([2], 1, stage="LLM", message="503")

    snapshot = book.snapshot()

    assert snapshot.lost_row_indices == [2]
    assert snapshot.rejected_row_indices == [1]
    assert [e.row_index for e in snapshot.errors] == [2]
    assert snapshot.is_complete is False


def test_with_a_contract_complete_means_every_row_accepted_or_repaired():
    book = ledger(2)
    judged(book, 0, 1, verdict(intent=PASS))
    judged(book, 1, 1, verdict(intent=fixed()))

    assert book.snapshot().is_complete is True

    flagged = ledger(1, FLAGGED)
    judged(flagged, 0, 1, verdict(intent=failing("intent: no")))
    assert flagged.snapshot().is_complete is False


# ---------------------------------------------------------------------------
# Arrival order
# ---------------------------------------------------------------------------


def _scripted_events() -> list[Callable[[RowLedger], None]]:
    """Every kind of event across 12 rows, each a closure over the ledger.

    Covers: clean pass, format fix, re-ask repair, rejection, provider repair
    interruption, provider loss then recovery, stage loss, provider + stage on
    the same row, an older failure after a newer success, an optional failure,
    a row never reached, and the run stopping.
    """

    def a(row: int, attempt: int, model: str = "cheap"):
        return lambda b: answer(b, row, attempt, model)

    def v(row: int, attempt: int, result: Verdict):
        return lambda b: b.record_verdict(row, attempt, result, stage=PARSER)

    def p(row: int, attempt: int, message: str):
        return lambda b: b.record_provider_failure(
            [row], attempt, stage="LLM", message=message
        )

    def s(row: int, message: str):
        return lambda b: b.record_stage_failure([row], stage="Parser", message=message)

    bad = verdict(intent=failing("intent: 'maybe' is not allowed"))
    return [
        a(0, 1), v(0, 1, verdict(intent=PASS)),
        a(1, 1), v(1, 1, verdict(intent=fixed("intent: wordy"))),
        a(2, 1), v(2, 1, bad), a(2, 2, "strong"), v(2, 2, verdict(intent=PASS)),
        a(3, 1), v(3, 1, bad),
        a(4, 1), v(4, 1, bad), p(4, 2, "timeout"),
        p(5, 1, "503"), a(5, 2), v(5, 2, verdict(intent=PASS)),
        s(6, "batch could not be split"),
        p(7, 1, "rate limited"), s(7, "not JSON"),
        p(8, 1, "older failure"), a(8, 2), v(8, 2, verdict(intent=PASS)),
        a(9, 1),
        v(9, 1, verdict(intent=PASS, note=failing("note: too long", required=False))),
        # Row 10 is answered but its check fails on two fields at once.
        a(10, 1),
        v(10, 1, verdict(intent=failing("intent: x"), amount=failing("amount: y"))),
        # Row 11 is never reached.
        lambda b: b.record_run_stopped(StopReason.EARLY_STOP),
    ]  # fmt: skip


def _fingerprint(book: RowLedger) -> list[tuple]:
    return [
        (r.status, r.reason, r.sub_reason, r.delivered_attempt, r.failed_checks)
        for r in book.snapshot().rows
    ]


@pytest.mark.parametrize("seed", range(8))
def test_shuffled_events_give_the_same_statuses_and_reasons(seed):
    """Acceptance test 3: async calls finish in any order; the ledger must not care."""
    in_order = ledger(12)
    for event in _scripted_events():
        event(in_order)

    events = _scripted_events()
    random.Random(seed).shuffle(events)  # noqa: S311 - a seeded test shuffle
    shuffled = ledger(12)
    for event in events:
        event(shuffled)

    assert _fingerprint(shuffled) == _fingerprint(in_order)
    assert [r.status for r in in_order.snapshot().rows] == [
        RowStatus.ACCEPTED,
        RowStatus.REPAIRED,
        RowStatus.REPAIRED,
        RowStatus.REJECTED,
        RowStatus.REJECTED,
        RowStatus.ACCEPTED,
        RowStatus.FAILED,
        RowStatus.FAILED,
        RowStatus.ACCEPTED,
        RowStatus.ACCEPTED,
        RowStatus.REJECTED,
        RowStatus.FAILED,
    ]


# ---------------------------------------------------------------------------
# Early stop
# ---------------------------------------------------------------------------


def _first_rows(book: RowLedger, rejected: int, total: int = 50) -> None:
    for row in range(total):
        result = (
            verdict(intent=failing("intent: no"))
            if row < rejected
            else verdict(intent=PASS)
        )
        judged(book, row, 1, result)


def test_early_stop_fires_when_more_than_half_of_the_first_50_are_rejected():
    """Acceptance test 9: 26 of 50 rejected stops the run; 25 does not."""
    over = ledger(200)
    _first_rows(over, rejected=26)
    at_limit = ledger(200)
    _first_rows(at_limit, rejected=25)

    assert over.early_stop_reached() is True
    assert at_limit.early_stop_reached() is False


def test_early_stop_waits_until_every_one_of_the_first_50_rows_is_final():
    book = ledger(200)
    for row in range(1, 50):
        judged(book, row, 1, verdict(intent=failing("intent: no")))

    assert book.early_stop_reached() is False  # row 0 still in flight

    book.record_provider_failure([0], 1, stage="LLM", message="503")
    assert book.early_stop_reached() is True


def test_early_stop_counts_the_first_rows_by_input_order_not_by_arrival():
    """Rows 50+ finishing first must not stand in for rows 0-49."""
    book = ledger(200)
    for row in range(50, 100):
        judged(book, row, 1, verdict(intent=failing("intent: no")))
    for row in range(50):
        judged(book, row, 1, verdict(intent=PASS))

    assert book.early_stop_reached() is False


def test_early_stop_threshold_and_window_come_from_the_policy():
    strict_policy = policy(early_stop_fraction=0.1, early_stop_window=10)
    book = ledger(20, strict_policy)
    _first_rows(book, rejected=2, total=10)

    assert book.early_stop_reached() is True

    off = ledger(20, policy(early_stop_fraction=None))
    _first_rows(off, rejected=10, total=10)
    assert off.early_stop_reached() is False


def test_early_stop_on_a_run_shorter_than_the_window_uses_every_row():
    book = ledger(4)
    _first_rows(book, rejected=3, total=4)

    assert book.early_stop_reached() is True


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def test_streaming_chunks_merge_into_the_parent_rows_they_map_to():
    parent = ledger(5)
    first = ledger(3)
    for row in range(3):
        judged(first, row, 1, verdict(intent=PASS))
    second = ledger(2)
    judged(second, 0, 1, verdict(intent=failing("intent: no")))
    second.record_provider_failure([1], 1, stage="LLM", message="503")

    parent.merge(first, [0, 1, 2])
    parent.merge(second, [3, 4])

    assert [row.status for row in parent.snapshot().rows] == [
        RowStatus.ACCEPTED,
        RowStatus.ACCEPTED,
        RowStatus.ACCEPTED,
        RowStatus.REJECTED,
        RowStatus.FAILED,
    ]


def test_a_row_seen_by_two_streaming_chunks_keeps_the_higher_attempt():
    parent = ledger(1)
    newer = ledger(1)
    judged(newer, 0, 2, verdict(intent=PASS))
    older = ledger(1)
    judged(older, 0, 1, verdict(intent=failing("intent: no")))

    parent.merge(newer, [0])
    parent.merge(older, [0])

    row = parent.snapshot().rows[0]
    assert row.status is RowStatus.REPAIRED
    assert row.delivered_attempt == 2


def test_a_row_recovered_by_auto_retry_ends_with_the_recovery():
    """Acceptance test 7: the retry pass's success replaces the parent's failure.

    The retry pipeline numbers its own attempts from 1 again; the merge places
    them after the parent's, so the recovery is the latest attempt.
    """
    parent = ledger(4, UNCHECKED)
    answer(parent, 0, 1)
    parent.record_provider_failure([1], 1, stage="LLM", message="503")
    answer(parent, 2, 1)
    parent.record_stage_failure([3], stage="Parser", message="not JSON")
    assert parent.snapshot().lost_row_indices == [1, 3]

    retry = ledger(2, UNCHECKED)  # re-runs parent rows 1 and 3
    answer(retry, 0, 1, model="retry-model")
    retry.record_provider_failure([1], 1, stage="LLM", message="still down")
    parent.merge(retry, [1, 3], retry=True)

    snapshot = parent.snapshot()
    assert snapshot.rows[1].status is RowStatus.UNCHECKED
    assert snapshot.rows[1].delivered_attempt == 2
    assert snapshot.rows[1].delivered_by == "retry-model"
    assert snapshot.rows[3].status is RowStatus.FAILED
    assert "still down" in snapshot.rows[3].reason
    assert snapshot.lost_row_indices == [3]
    assert [e.message for e in snapshot.errors] == ["still down"]
    assert sum(snapshot.counts.values()) == 4


def test_a_recovered_row_under_a_contract_is_accepted_after_a_provider_loss():
    """No answer failed a check, so the first judged answer passing is accepted."""
    parent = ledger(1)
    parent.record_provider_failure([0], 1, stage="LLM", message="503")
    retry = ledger(1)
    judged(retry, 0, 1, verdict(intent=PASS))

    parent.merge(retry, [0], retry=True)

    row = parent.snapshot().rows[0]
    assert row.status is RowStatus.ACCEPTED
    assert row.delivered_attempt == 2
    assert parent.snapshot().is_complete is True


def test_a_stop_in_a_merged_chunk_reaches_the_parent():
    parent = ledger(4)
    chunk = ledger(2)
    judged(chunk, 0, 1, verdict(intent=PASS))
    chunk.record_run_stopped(StopReason.BUDGET_CAP)

    parent.merge(chunk, [0, 1])

    snapshot = parent.snapshot()
    assert snapshot.stop_reason is StopReason.BUDGET_CAP
    assert "budget cap" in snapshot.rows[3].reason


# ---------------------------------------------------------------------------
# Snapshot, cost and raw answers
# ---------------------------------------------------------------------------


def test_a_snapshot_is_frozen_against_later_events():
    book = ledger(2)
    judged(book, 0, 1, verdict(intent=PASS))
    snapshot = book.snapshot()

    judged(book, 1, 1, verdict(intent=PASS))

    assert snapshot.rows[1].status is RowStatus.FAILED
    assert snapshot.counts[RowStatus.ACCEPTED] == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.rows[0].status = RowStatus.REJECTED  # type: ignore[misc]
    with pytest.raises(TypeError):
        snapshot.counts[RowStatus.ACCEPTED] = 99  # type: ignore[index]


def test_a_row_records_every_attempt_and_its_total_cost():
    book = ledger(1)
    judged(book, 0, 1, verdict(intent=failing("intent: no")))
    judged(book, 0, 2, verdict(intent=PASS))

    row = book.snapshot().rows[0]

    assert [a.attempt for a in row.attempts] == [1, 2]
    assert row.cost == Decimal("0.02")


def test_raw_answers_are_kept_for_the_first_quarantined_rows_only():
    """D3: statuses and counts are exact; raw answers stop at the cap."""
    contract = build_contract(["intent"], fields=[OutputField("intent")])
    book = ledger(4, LedgerPolicy.from_contract(contract, raw_answer_cap=2))
    judged(book, 0, 1, verdict(intent=PASS))
    for row in (1, 2, 3):
        judged(book, row, 1, verdict(intent=failing("intent: no")))

    rows = book.snapshot().rows

    assert rows[0].attempts[0].raw_answer is None  # accepted: not quarantined
    assert rows[1].attempts[0].raw_answer == "raw-1-1"
    assert rows[2].attempts[0].raw_answer == "raw-2-1"
    assert rows[3].attempts[0].raw_answer is None
    assert rows[3].status is RowStatus.REJECTED


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_an_event_for_a_row_outside_the_input_is_a_broken_invariant():
    book = ledger(2)

    with pytest.raises(LedgerInvariantError):
        answer(book, 2, 1)


def test_outside_tests_a_broken_invariant_is_reported_not_raised():
    book = RowLedger(2, CHECKED)  # not strict

    answer(book, 7, 1)
    judged(book, 0, 1, verdict(intent=PASS))

    snapshot = book.snapshot()
    assert len(snapshot.violations) == 1
    assert "7" in snapshot.violations[0]
    assert snapshot.rows[0].status is RowStatus.ACCEPTED


@pytest.mark.parametrize("safe_first", [True, False])
def test_two_different_verdicts_for_one_attempt_keep_the_safer_one(safe_first):
    """A conflict is a defect upstream; never deliver the value it puts in doubt."""
    book = RowLedger(1, CHECKED)
    answer(book, 0, 1)
    passing, rejecting = verdict(intent=PASS), verdict(intent=failing("intent: no"))
    for result in (rejecting, passing) if safe_first else (passing, rejecting):
        book.record_verdict(0, 1, result, stage=PARSER)

    snapshot = book.snapshot()
    assert snapshot.rows[0].status is RowStatus.REJECTED
    assert len(snapshot.violations) == 1


def test_two_different_verdicts_for_one_attempt_raise_in_a_strict_ledger():
    strict = ledger(1)
    judged(strict, 0, 1, verdict(intent=PASS))

    with pytest.raises(LedgerInvariantError):
        strict.record_verdict(0, 1, verdict(intent=failing("intent: no")), stage=PARSER)


def test_the_same_verdict_reported_twice_is_not_a_conflict():
    """Resume re-judges cached answers; a deterministic judge repeats itself."""
    book = ledger(1)
    judged(book, 0, 1, verdict(intent=PASS))
    judged(book, 0, 1, verdict(intent=PASS))

    row = book.snapshot().rows[0]
    assert row.status is RowStatus.ACCEPTED
    assert len(row.attempts) == 1


@dataclass(frozen=True)
class _ContradictoryVerdict(Verdict):
    """A judge verdict whose row flag disagrees with its own fields."""

    @property
    def passed(self) -> bool:
        return True


def test_a_verdict_that_contradicts_its_own_fields_is_a_broken_invariant():
    book = ledger(1)
    contradiction = _ContradictoryVerdict(
        checked=True, fields=verdict(intent=failing("intent: no")).fields
    )

    with pytest.raises(LedgerInvariantError):
        book.record_verdict(0, 1, contradiction, stage=PARSER)


def test_a_contradictory_verdict_is_never_delivered():
    book = RowLedger(1, CHECKED)  # not strict
    answer(book, 0, 1)
    contradiction = _ContradictoryVerdict(
        checked=True, fields=verdict(intent=failing("intent: no")).fields
    )
    book.record_verdict(0, 1, contradiction, stage=PARSER)

    snapshot = book.snapshot()
    assert snapshot.rows[0].status is RowStatus.REJECTED
    assert len(snapshot.violations) == 1


def test_a_merge_mapping_must_cover_every_child_row():
    parent = ledger(3)
    child = ledger(2)

    with pytest.raises(LedgerInvariantError):
        parent.merge(child, [0])


def test_attempt_numbers_start_at_one():
    book = ledger(1)

    with pytest.raises(LedgerInvariantError):
        book.record_provider_failure([0], 0, stage="LLM", message="503")


def test_an_impossible_ledger_is_refused_at_construction():
    with pytest.raises(ValueError):
        RowLedger(-1, CHECKED)
    contract = build_contract(["intent"], fields=[OutputField("intent")])
    with pytest.raises(ValueError):
        LedgerPolicy.from_contract(contract, raw_answer_cap=-1)


# ---------------------------------------------------------------------------
# One vocabulary with the judge
# ---------------------------------------------------------------------------


def test_an_absent_optional_field_passes_and_a_failed_one_does_not():
    """A field passes unless its outcome is "failed"; "absent" is a valid answer."""
    book = ledger(2)
    judged(book, 0, 1, verdict(intent=PASS, note=ABSENT))
    judged(book, 1, 1, verdict(intent=failing("intent: no"), note=ABSENT))

    rows = book.snapshot().rows

    assert rows[0].status is RowStatus.ACCEPTED
    assert dict(rows[0].field_statuses) == {
        "intent": RowStatus.ACCEPTED,
        "note": RowStatus.ACCEPTED,
    }
    assert rows[1].status is RowStatus.REJECTED
    assert rows[1].failed_checks == ("intent: no",)


@pytest.mark.parametrize(
    ("book_policy", "result"),
    [
        (CHECKED, unchecked_verdict(intent=PASS)),
        (UNCHECKED, verdict(answer=PASS)),
    ],
    ids=["unchecked verdict in a checked run", "checked verdict in an unchecked run"],
)
def test_a_verdict_judged_against_another_contract_is_a_broken_invariant(
    book_policy, result
):
    """ "Checked" has one source, the contract; a verdict disagreeing is a defect."""
    book = ledger(1, book_policy)

    with pytest.raises(LedgerInvariantError):
        book.record_verdict(0, 1, result, stage=PARSER)


# ---------------------------------------------------------------------------
# No contract: an unparseable answer reads exactly as today
# ---------------------------------------------------------------------------

_NOT_AN_OBJECT = (
    "response is not a JSON object with any of the output columns "
    "['intent', 'amount', 'note']: 'the intent is refund'"
)


def test_an_unparseable_multi_column_answer_gives_one_message_not_one_per_column():
    """The judge puts today's one parse message on every column; it counts once."""
    book = ledger(1, UNCHECKED)
    every_column = {
        name: failing(_NOT_AN_OBJECT) for name in ("intent", "amount", "note")
    }
    judged(book, 0, 1, unchecked_verdict(**every_column))

    snapshot = book.snapshot()

    assert [e.message for e in snapshot.errors] == [_NOT_AN_OBJECT]
    assert snapshot.rows[0].reason == _NOT_AN_OBJECT
    assert snapshot.rows[0].failed_checks == (_NOT_AN_OBJECT,)


def test_a_checked_verdict_counts_one_reason_shared_by_two_fields_once():
    book = ledger(1, FLAGGED)
    shared = "intent and amount: the answer is not a JSON object"
    judged(book, 0, 1, verdict(intent=failing(shared), amount=failing(shared)))

    row = book.snapshot().rows[0]

    assert row.failed_checks == (shared,)
    assert row.reason == shared


def test_an_unparseable_answer_is_reported_under_the_stage_that_judged_it():
    """Today's error list names the parser stage, so the ledger's must too."""
    book = ledger(1, UNCHECKED)
    answer(book, 0, 1)
    book.record_verdict(
        0,
        1,
        unchecked_verdict(answer=failing("response could not be parsed: boom")),
        stage="ResponseParser",
    )

    (error,) = book.snapshot().errors

    assert (error.stage, error.error_type, error.message) == (
        "ResponseParser",
        "parse_failure",
        "response could not be parsed: boom",
    )


# ---------------------------------------------------------------------------
# Ties on one attempt break on content, never on arrival or merge order
# ---------------------------------------------------------------------------


def _chunk_answering(model: str) -> RowLedger:
    chunk = RowLedger(1, CHECKED)
    answer(chunk, 0, 1, model=model)
    chunk.record_verdict(0, 1, verdict(intent=PASS), stage=PARSER)
    return chunk


@pytest.mark.parametrize("a_first", [True, False])
def test_two_chunks_answering_one_row_on_one_attempt_is_a_violation(a_first):
    """Two pipelines claimed the same answer; the delivered model must not
    depend on which chunk merged first."""
    parent = RowLedger(1, CHECKED)
    chunks = [_chunk_answering("model-a"), _chunk_answering("model-b")]
    for chunk in chunks if a_first else reversed(chunks):
        parent.merge(chunk, [0])

    snapshot = parent.snapshot()

    assert len(snapshot.violations) == 1
    assert "attempt 1" in snapshot.violations[0]
    assert snapshot.rows[0].delivered_by == "model-b"


def test_a_duplicate_merge_raises_in_a_strict_ledger():
    parent = ledger(1)
    parent.merge(_chunk_answering("model-a"), [0])

    with pytest.raises(LedgerInvariantError):
        parent.merge(_chunk_answering("model-b"), [0])


def test_merging_the_same_answer_twice_is_not_a_violation():
    parent = ledger(1)
    parent.merge(_chunk_answering("model-a"), [0])
    parent.merge(_chunk_answering("model-a"), [0])

    assert parent.snapshot().rows[0].delivered_by == "model-a"


@pytest.mark.parametrize("first", ["503 overloaded", "429 rate limited"])
def test_two_provider_failures_on_one_attempt_pick_the_same_cause_in_any_order(
    first,
):
    book = ledger(1, UNCHECKED)
    messages = ["503 overloaded", "429 rate limited"]
    messages.remove(first)
    for message in (first, *messages):
        book.record_provider_failure([0], 1, stage="LLM", message=message)

    snapshot = book.snapshot()

    assert snapshot.rows[0].reason == "LLM: 429 rate limited"
    assert [e.message for e in snapshot.errors] == ["429 rate limited"]


@pytest.mark.parametrize("first", ["timeout", "connection reset"])
def test_a_repair_cut_short_twice_names_the_same_provider_failure_in_any_order(
    first,
):
    book = ledger(1)
    judged(book, 0, 1, verdict(intent=failing("intent: no")))
    messages = ["timeout", "connection reset"]
    messages.remove(first)
    for message in (first, *messages):
        book.record_provider_failure([0], 2, stage="LLM", message=message)

    row = book.snapshot().rows[0]

    assert row.reason.endswith(
        "(repair cut short by provider failure: LLM: connection reset)"
    )


# ---------------------------------------------------------------------------
# One policy: the contract's
# ---------------------------------------------------------------------------


def test_the_ledger_policy_comes_from_the_contract():
    contract = build_contract(
        ["intent"],
        fields=[OutputField("intent")],
        policy=RunPolicy(
            on_failure="deliver_flagged", early_stop_fraction=0.2, early_stop_window=5
        ),
    )

    book = ledger(10, LedgerPolicy.from_contract(contract, raw_answer_cap=3))
    for row in range(5):
        result = verdict(intent=failing("intent: no") if row < 2 else PASS)
        judged(book, row, 1, result)

    assert book.snapshot().rows[0].status is RowStatus.UNVERIFIED
    assert book.early_stop_reached() is False  # unverified rows are not rejected


def test_early_stop_window_and_threshold_follow_the_contract():
    book = ledger(10, policy(early_stop_fraction=0.2, early_stop_window=5))
    _first_rows(book, rejected=2, total=5)

    assert book.early_stop_reached() is True


def test_an_unchecked_contract_gives_an_unchecked_ledger_with_no_early_stop():
    """Its run policy keeps the defaults, but nothing can be rejected."""
    contract = build_contract(["answer"])
    assert contract.checked is False
    book = ledger(50, LedgerPolicy.from_contract(contract))
    for row in range(50):
        judged(book, row, 1, unchecked_verdict(answer=failing("not JSON")))

    assert book.early_stop_reached() is False
    assert book.snapshot().counts[RowStatus.FAILED] == 50


def test_the_early_stop_fraction_range_is_the_contracts():
    """1.0 would never stop a run; the contract refuses it, so the ledger must."""
    from ondine.harness.contract import ContractError

    with pytest.raises(ContractError):
        policy(early_stop_fraction=1.0)
    with pytest.raises(ValueError):
        LedgerPolicy(
            checked=True, run=RunPolicy(early_stop_fraction=1.0), raw_answer_cap=1
        )


# ---------------------------------------------------------------------------
# Deliver flagged: the status agrees with the value delivered
# ---------------------------------------------------------------------------


def test_deliver_flagged_never_delivers_an_earlier_passing_attempt_as_unverified():
    """The latest answer failed; an earlier one passed with zero failed checks.

    Delivering the passing one while calling the row unverified would make the
    row's status disagree with every field status. Only answers that failed a
    check are candidates for an unverified row.
    """
    book = ledger(1, FLAGGED)
    judged(book, 0, 1, verdict(intent=PASS))
    judged(book, 0, 2, verdict(intent=failing("intent: 'maybe' is not allowed")))

    row = book.snapshot().rows[0]

    assert row.status is RowStatus.UNVERIFIED
    assert row.delivered_attempt == 2
    assert dict(row.field_statuses) == {"intent": RowStatus.UNVERIFIED}
    assert "intent: 'maybe' is not allowed" in row.reason


# ---------------------------------------------------------------------------
# Polling cost
# ---------------------------------------------------------------------------


def test_polling_early_stop_and_counts_only_re_decides_rows_that_changed(monkeypatch):
    """The model-call stage polls early stop per batch; a poll must not re-derive
    every row each time, only those with new events since the last poll."""
    from ondine.harness import ledger as ledger_module

    book = ledger(200)
    _first_rows(book, rejected=10)
    book.early_stop_reached()
    book.counts()

    calls = []
    real = ledger_module._decide
    monkeypatch.setattr(
        ledger_module, "_decide", lambda *a, **k: calls.append(a[0]) or real(*a, **k)
    )
    assert book.early_stop_reached() is False
    book.counts()
    assert calls == []

    judged(book, 3, 2, verdict(intent=PASS))
    judged(book, 120, 1, verdict(intent=PASS))
    assert book.early_stop_reached() is False
    assert book.counts()[RowStatus.ACCEPTED] == 41  # row 3 is repaired
    assert sorted(set(calls)) == [3, 120]
