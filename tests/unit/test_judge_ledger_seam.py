"""The seam between the judge and the ledger, run for real end to end.

``test_row_ledger.py`` hand-builds verdicts to pin each status rule, and
``test_judge.py`` pins what the judge returns. Neither proves the two agree on
one vocabulary: a ledger that reads a field the judge does not have would pass
both and crash on the first real row. Here every verdict comes from the real
``judge()`` against a real ``build_contract()``, and goes straight into a real
``RowLedger``, with no adapter in between.

Under the unchecked contract the ledger's error list is also compared with
what today's ``ResponseParserStage`` records in a real ``ExecutionContext``:
same stage name, same error type, same single message.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from ondine.core.models import LLMResponse, ResponseBatch, RowMetadata
from ondine.core.specifications import PromptSpec
from ondine.harness.contract import (
    OutputContract,
    OutputField,
    RunPolicy,
    build_contract,
)
from ondine.harness.judge import judge
from ondine.harness.ledger import LedgerPolicy, LedgerSnapshot, RowLedger
from ondine.harness.status import RowStatus
from ondine.orchestration.execution_context import ExecutionContext
from ondine.stages.parser_factory import create_response_parser
from ondine.stages.response_parser_stage import ResponseParserStage

INTENT = OutputField("intent", allowed=["refund", "cancel"], normalise=["casefold"])


def _run(contract: OutputContract, *answers: str) -> LedgerSnapshot:
    """Judge each answer as one attempt of row 0, report it, snapshot."""
    parser = ResponseParserStage(output_columns=list(contract.output_columns))
    book = RowLedger(1, LedgerPolicy.from_contract(contract), strict=True)
    for attempt, text in enumerate(answers, start=1):
        book.record_answer(
            0, attempt, model=f"model-{attempt}", tokens=5, cost=Decimal("0.01")
        )
        book.record_verdict(0, attempt, judge(contract, text), stage=parser.name)
    return book.snapshot()


def _today_errors(contract: OutputContract, text: str, response_format: str):
    """The row errors today's parse step records for ``text``."""
    columns = list(contract.output_columns)
    spec = PromptSpec(template="Classify {text}", response_format=response_format)
    stage = ResponseParserStage(
        parser=create_response_parser(spec, columns), output_columns=columns
    )
    context = ExecutionContext()
    response = LLMResponse(
        text=text,
        tokens_in=1,
        tokens_out=1,
        model="test",
        cost=Decimal("0"),
        latency_ms=0.0,
    )
    batch = ResponseBatch(responses=[response], metadata=[RowMetadata(row_index=0)])
    stage.process(([batch], columns), context)
    return context.row_errors


def _shape(errors) -> list[tuple]:
    return [(e.row_index, e.stage, e.error_type, e.message) for e in errors]


# ---------------------------------------------------------------------------
# Unchecked contract: today's behaviour
# ---------------------------------------------------------------------------


def test_unchecked_single_column_answer_is_delivered_unchecked():
    contract = build_contract(["answer"])

    snapshot = _run(contract, "  assistant: Positive  ")

    row = snapshot.rows[0]
    assert row.status is RowStatus.UNCHECKED
    assert row.delivered_attempt == 1
    assert row.delivered_by == "model-1"
    assert snapshot.errors == ()
    assert snapshot.is_complete is True


@pytest.mark.parametrize(
    ("text", "response_format"),
    [
        ("the intent is refund", "raw"),
        ("```json\n{oops\n```", "json"),
        ('{"label": "x"}', "raw"),
    ],
    ids=["not json", "broken fence", "no output column"],
)
def test_unchecked_multi_column_parse_failure_reads_as_today(text, response_format):
    """One message under the parser's stage name, not one per column under "judge"."""
    contract = build_contract(
        ["intent", "amount", "note"], response_format=response_format
    )

    snapshot = _run(contract, text)

    today = _today_errors(contract, text, response_format)
    assert len(today) == 1
    assert _shape(snapshot.errors) == _shape(today)
    assert snapshot.rows[0].status is RowStatus.FAILED
    assert snapshot.rows[0].reason == today[0].message
    assert snapshot.failed_rows == 1
    assert snapshot.lost_row_indices == [0]


# ---------------------------------------------------------------------------
# Checked contract: every status the judge can lead to
# ---------------------------------------------------------------------------


def test_checked_answer_that_passes_is_accepted():
    contract = build_contract(["intent"], fields=[INTENT])

    row = _run(contract, "Refund").rows[0]

    assert row.status is RowStatus.ACCEPTED
    assert row.reason == ""
    assert dict(row.field_statuses) == {"intent": RowStatus.ACCEPTED}
    assert row.delivered_attempt == 1


def test_checked_wordy_answer_fixed_by_format_is_repaired():
    contract = build_contract(["intent"], fields=[INTENT])

    row = _run(contract, "The customer clearly wants a refund.").rows[0]

    assert row.status is RowStatus.REPAIRED
    assert "intent: format fix" in row.reason
    assert dict(row.field_statuses) == {"intent": RowStatus.REPAIRED}


def test_checked_answer_outside_the_allowed_values_is_rejected():
    contract = build_contract(["intent"], fields=[INTENT])

    snapshot = _run(contract, "maybe")

    row = snapshot.rows[0]
    assert row.status is RowStatus.REJECTED
    assert row.reason == "intent: 'maybe' is not one of the 2 allowed values"
    assert row.failed_checks == (row.reason,)
    assert row.delivered_attempt is None
    assert snapshot.rejected_row_indices == [0]
    assert snapshot.errors == ()  # a rejection is not a lost row (#312)


def test_checked_answer_that_passes_on_a_later_attempt_is_repaired():
    contract = build_contract(["intent"], fields=[INTENT])

    row = _run(contract, "maybe", "cancel").rows[0]

    assert row.status is RowStatus.REPAIRED
    assert row.delivered_attempt == 2
    assert row.delivered_by == "model-2"
    assert "-> passed on attempt 2" in row.reason


def test_checked_failure_under_deliver_flagged_is_unverified():
    contract = build_contract(
        ["intent"],
        fields=[INTENT],
        policy=RunPolicy(on_failure="deliver_flagged"),
    )

    snapshot = _run(contract, "maybe")

    row = snapshot.rows[0]
    assert row.status is RowStatus.UNVERIFIED
    assert row.delivered_attempt == 1
    assert dict(row.field_statuses) == {"intent": RowStatus.UNVERIFIED}
    assert "is not one of the 2 allowed values" in row.reason
    assert snapshot.is_complete is False


def test_checked_multi_field_answer_with_an_absent_optional_field_is_accepted():
    contract = build_contract(
        ["intent", "note"],
        fields=[INTENT, OutputField("note", required=False, absent="n/a")],
    )

    row = _run(contract, '{"intent": "cancel", "note": "n/a"}').rows[0]

    assert row.status is RowStatus.ACCEPTED
    assert dict(row.field_statuses) == {
        "intent": RowStatus.ACCEPTED,
        "note": RowStatus.ACCEPTED,
    }


def test_checked_answer_missing_a_required_field_is_rejected_with_one_reason_per_field():
    contract = build_contract(
        ["intent", "amount"],
        fields=[INTENT, OutputField("amount", type="integer", minimum=0)],
    )

    row = _run(contract, "not an object").rows[0]

    assert row.status is RowStatus.REJECTED
    assert row.failed_checks == (
        "intent: missing, the answer is not a JSON object",
        "amount: missing, the answer is not a JSON object",
    )
