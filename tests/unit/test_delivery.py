"""Delivery: what the client receives, read from one ledger snapshot.

Each test pins a guarantee from the interface spec (harness-F1F2-interfaces.md,
section 4): every input row comes out, in input order; values are kept or
blanked by status; the five harness columns appear only under a checked
contract; the ``reason`` column names the check that blocked delivery last,
plus the path taken; the quarantine and the run report are readings of the
same snapshot; and the same snapshot always gives the same output.

The snapshots come from the real ``build_contract`` -> ``judge`` ->
``RowLedger`` chain, so delivery is tested against what the ledger and the
judge really produce, not against hand-built look-alikes.
"""

from __future__ import annotations

import random
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from ondine import PipelineBuilder
from ondine.harness.contract import (
    OutputContract,
    OutputField,
    RunPolicy,
    build_contract,
)
from ondine.harness.delivery import (
    HARNESS_COLUMNS,
    CheckFailure,
    Delivery,
    JudgedAnswer,
    deliver,
    harness_column_conflicts,
)
from ondine.harness.judge import judge
from ondine.harness.ledger import (
    PROVIDER_SUB_REASON,
    LedgerPolicy,
    RowLedger,
    StopReason,
)
from ondine.harness.status import RowStatus
from tests.e2e.conformance_harness import LedgerClient, conformance_frame, token_for

PARSER = "ResponseParser"

INTENT = OutputField(
    "intent", allowed=["refund", "cancel", "other"], normalise=["casefold"]
)
AMOUNT = OutputField("amount", type="number", minimum=0)
NOTE = OutputField("note", required=False, max_length=5)


def _contract(*fields: OutputField, **policy: Any) -> OutputContract:
    return build_contract(
        fields=fields, policy=RunPolicy(**policy), response_format="json"
    )


class Run:
    """A real ledger fed by the real judge, plus the judged answers delivery reads."""

    def __init__(
        self, contract: OutputContract, rows: int, *, raw_answer_cap: int = 1000
    ) -> None:
        self.contract = contract
        policy = LedgerPolicy.from_contract(contract, raw_answer_cap=raw_answer_cap)
        self.ledger = RowLedger(rows, policy, strict=True)
        self.judged: dict[tuple[int, int], JudgedAnswer] = {}
        self.inputs = [{"text": f"message {i}", "id": i} for i in range(rows)]

    def answer(
        self,
        row: int,
        attempt: int,
        text: str,
        *,
        model: str = "small",
        cost: str = "0.01",
    ) -> None:
        self.ledger.record_answer(
            row,
            attempt,
            model=model,
            tokens=10,
            cost=Decimal(cost),
            raw_answer=text,
        )
        verdict = judge(self.contract, text, self.inputs[row])
        self.ledger.record_verdict(row, attempt, verdict, stage=PARSER)
        self.judged[(row, attempt)] = JudgedAnswer.from_verdict(verdict)

    def provider_failure(self, row: int, attempt: int) -> None:
        self.ledger.record_provider_failure(
            [row], attempt, stage="LLMInvocation", message="timeout"
        )

    def deliver(self, **options: Any) -> Delivery:
        return deliver(
            self.ledger.snapshot(),
            self.judged,
            self.inputs,
            self.contract,
            **options,
        )


def _json(**values: Any) -> str:
    import json

    return json.dumps(values)


# ---------------------------------------------------------------------------
# Unchecked contract: today's frame, nothing added
# ---------------------------------------------------------------------------


class TestUnchecked:
    def test_frame_equals_plain_values_frame_with_no_added_columns(self):
        contract = build_contract(["label"])
        run = Run(contract, 3)
        run.answer(0, 1, "positive")
        run.answer(2, 1, "negative")
        run.provider_failure(1, 1)

        out = run.deliver()

        plain = pd.DataFrame(
            [
                {"text": "message 0", "id": 0, "label": "positive"},
                {"text": "message 1", "id": 1, "label": None},
                {"text": "message 2", "id": 2, "label": "negative"},
            ]
        )
        pd.testing.assert_frame_equal(out.to_pandas(), plain)
        assert not set(HARNESS_COLUMNS) & set(out.columns)

    def test_report_has_counts_but_no_coverage_or_rules(self):
        run = Run(build_contract(["label"]), 2)
        run.answer(0, 1, "positive")

        report = run.deliver().report

        assert report.counts[RowStatus.UNCHECKED] == 1
        assert report.counts[RowStatus.FAILED] == 1
        assert report.coverage is None
        assert report.rule_pass_rates == ()
        assert report.invariant_holds

    def test_failed_rows_are_quarantined_even_without_a_contract(self):
        run = Run(build_contract(["label"]), 2)
        run.answer(0, 1, "positive")

        quarantine = run.deliver().quarantine

        assert [entry.row for entry in quarantine] == [1]
        assert quarantine[0].status is RowStatus.FAILED


# ---------------------------------------------------------------------------
# Rows: count, order, values kept or blanked by status
# ---------------------------------------------------------------------------


class TestRows:
    def test_row_count_and_order_match_the_input_even_when_the_run_stopped(self):
        run = Run(_contract(INTENT), 5)
        run.answer(3, 1, _json(intent="refund"))
        run.answer(0, 1, _json(intent="cancel"))
        run.ledger.record_run_stopped(StopReason.BUDGET_CAP)

        out = run.deliver()
        frame = out.to_pandas()

        assert len(frame) == 5
        assert list(frame["id"]) == [0, 1, 2, 3, 4]
        assert list(frame["status"]) == [
            "accepted",
            "failed",
            "failed",
            "accepted",
            "failed",
        ]
        assert frame.loc[1, "reason"] == "run stopped before this row: budget cap"

    def test_values_are_kept_or_blanked_by_status(self):
        run = Run(_contract(INTENT), 4)
        run.answer(0, 1, _json(intent="refund"))  # accepted
        run.answer(1, 1, _json(intent="I think they want a refund"))  # repaired
        run.answer(2, 1, _json(intent="upgrade"))  # rejected
        run.provider_failure(3, 1)  # failed

        frame = run.deliver().to_pandas()

        assert list(frame["status"]) == ["accepted", "repaired", "rejected", "failed"]
        assert frame.loc[0, "intent"] == "refund"
        assert frame.loc[1, "intent"] == "refund"
        assert pd.isna(frame.loc[2, "intent"])
        assert pd.isna(frame.loc[3, "intent"])

    def test_unverified_rows_keep_the_failed_value(self):
        run = Run(_contract(INTENT, on_failure="deliver_flagged"), 1)
        run.answer(0, 1, _json(intent="upgrade"))

        frame = run.deliver().to_pandas()

        assert frame.loc[0, "status"] == "unverified"
        assert frame.loc[0, "intent"] == "upgrade"
        assert frame.loc[0, "delivered_by"] == "small"

    def test_a_failed_optional_field_is_blanked_on_an_accepted_row(self):
        run = Run(_contract(INTENT, NOTE), 1)
        run.answer(0, 1, _json(intent="refund", note="far too long"))

        frame = run.deliver().to_pandas()

        assert frame.loc[0, "status"] == "accepted"
        assert frame.loc[0, "intent"] == "refund"
        assert pd.isna(frame.loc[0, "note"])

    def test_keep_valid_fields_delivers_the_passing_fields_of_a_rejected_row(self):
        run = Run(_contract(INTENT, AMOUNT, partial_rows="keep_valid_fields"), 1)
        run.answer(0, 1, _json(intent="refund", amount=-3))

        frame = run.deliver().to_pandas()

        assert frame.loc[0, "status"] == "rejected"
        assert frame.loc[0, "intent"] == "refund"
        assert pd.isna(frame.loc[0, "amount"])

    def test_whole_row_policy_blanks_every_field_of_a_rejected_row(self):
        run = Run(_contract(INTENT, AMOUNT), 1)
        run.answer(0, 1, _json(intent="refund", amount=-3))

        frame = run.deliver().to_pandas()

        assert pd.isna(frame.loc[0, "intent"])
        assert pd.isna(frame.loc[0, "amount"])

    def test_columns_are_inputs_then_outputs_then_the_harness_columns(self):
        run = Run(_contract(INTENT, AMOUNT), 1)
        run.answer(0, 1, _json(intent="refund", amount=3))

        out = run.deliver()

        assert out.columns == (
            "text",
            "id",
            "intent",
            "amount",
            "status",
            "reason",
            "checks",
            "repair_method",
            "delivered_by",
        )
        assert list(out.to_pandas().columns) == list(out.columns)

    def test_records_are_fresh_copies(self):
        run = Run(_contract(INTENT), 1)
        run.answer(0, 1, _json(intent="refund"))
        out = run.deliver()

        out.records()[0]["intent"] = "tampered"

        assert out.records()[0]["intent"] == "refund"

    def test_inputs_are_not_mutated(self):
        run = Run(_contract(INTENT), 1)
        run.answer(0, 1, _json(intent="refund"))

        run.deliver()

        assert run.inputs[0] == {"text": "message 0", "id": 0}


# ---------------------------------------------------------------------------
# The five harness columns
# ---------------------------------------------------------------------------


class TestHarnessColumns:
    def test_accepted_row(self):
        run = Run(_contract(INTENT, AMOUNT), 1)
        run.answer(0, 1, _json(intent="refund", amount=3), model="haiku")

        row = run.deliver().records()[0]

        assert row["status"] == "accepted"
        assert row["reason"] == ""
        assert row["checks"] == "intent: accepted; amount: accepted"
        assert row["repair_method"] == "none"
        assert row["delivered_by"] == "haiku"

    def test_format_fixed_row_names_what_the_fix_repaired(self):
        """The judge keeps the pre-fix check in ``fixed_reasons``; the ledger's
        note reads it, and delivery shows the ledger's reason as it is."""
        run = Run(_contract(INTENT), 1)
        run.answer(0, 1, _json(intent="They want to cancel."))
        fixed = "intent: 'They want to cancel.' is not one of the 3 allowed values"

        row = run.deliver().records()[0]

        assert row["status"] == "repaired"
        assert row["repair_method"] == "format"
        assert row["checks"] == "intent: repaired (format)"
        assert row["reason"].startswith("intent: format fix")
        assert row["reason"].count(fixed) == 1
        assert row["reason"].endswith("[path: attempt 1 format fix]")
        assert row["reason"] == (
            f"{run.ledger.snapshot().rows[0].reason} [path: attempt 1 format fix]"
        )

    def test_row_repaired_by_a_later_attempt(self):
        run = Run(_contract(INTENT), 1)
        run.answer(0, 1, _json(intent="upgrade"), model="small")
        run.answer(0, 2, _json(intent="cancel"), model="large")

        row = run.deliver().records()[0]

        assert row["status"] == "repaired"
        assert row["repair_method"] == "re-ask"
        assert row["delivered_by"] == "large"
        assert row["checks"] == "intent: repaired (re-ask)"
        assert "'upgrade' is not one of the 3 allowed values" in row["reason"]
        assert row["reason"].endswith(
            "[path: attempt 1 failed check -> attempt 2 passed]"
        )

    def test_reason_is_the_last_blocking_check_plus_the_path(self):
        run = Run(_contract(INTENT), 1)
        run.answer(0, 1, _json(intent="upgrade"))
        run.answer(0, 2, _json(intent="downgrade"))

        row = run.deliver().records()[0]

        assert row["status"] == "rejected"
        assert "'downgrade'" in row["reason"]
        assert "'upgrade'" not in row["reason"]
        assert row["reason"].endswith(
            "[path: attempt 1 failed check -> attempt 2 failed check]"
        )
        assert row["repair_method"] == "none"
        assert row["delivered_by"] is None
        assert row["checks"] == "intent: rejected"

    def test_failed_row_has_the_failure_and_no_checks(self):
        run = Run(_contract(INTENT), 1)
        run.provider_failure(0, 1)

        row = run.deliver().records()[0]

        assert row["status"] == "failed"
        assert row["reason"] == "LLMInvocation: timeout"
        assert row["checks"] == ""
        assert row["delivered_by"] is None

    def test_absent_optional_field_reads_absent(self):
        run = Run(_contract(INTENT, NOTE), 1)
        run.answer(0, 1, _json(intent="refund"))

        assert run.deliver().records()[0]["checks"] == (
            "intent: accepted; note: absent"
        )

    def test_conflicting_input_column_is_refused(self):
        run = Run(_contract(INTENT), 1)
        run.inputs[0]["status"] = "vip"
        run.answer(0, 1, _json(intent="refund"))

        with pytest.raises(ValueError, match="status"):
            run.deliver()

    def test_conflicts_are_reported_for_build_time_checks(self):
        assert harness_column_conflicts(["text", "status", "reason"]) == [
            "status",
            "reason",
        ]
        assert harness_column_conflicts(["text"]) == []


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------


class TestQuarantine:
    def test_one_entry_per_rejected_unverified_or_failed_row_in_input_order(self):
        run = Run(_contract(INTENT), 4)
        run.provider_failure(3, 1)
        run.answer(0, 1, _json(intent="refund"))
        run.answer(1, 1, _json(intent="upgrade"))

        quarantine = run.deliver().quarantine

        assert [(e.row, e.status) for e in quarantine] == [
            (1, RowStatus.REJECTED),
            (2, RowStatus.FAILED),
            (3, RowStatus.FAILED),
        ]

    def test_entry_holds_inputs_answers_checks_models_and_cost(self):
        run = Run(_contract(INTENT, AMOUNT), 1)
        first = _json(intent="upgrade", amount=-1)
        second = _json(intent="refund", amount=-2)
        run.answer(0, 1, first, model="small", cost="0.01")
        run.answer(0, 2, second, model="large", cost="0.10")

        (entry,) = run.deliver().quarantine

        assert entry.inputs == {"text": "message 0", "id": 0}
        assert [(a.attempt, a.model, a.raw_answer) for a in entry.attempts] == [
            (1, "small", first),
            (2, "large", second),
        ]
        assert [(c.attempt, c.field, c.rule) for c in entry.failed_checks] == [
            (1, "intent", "allowed_values"),
            (1, "amount", "minimum"),
            (2, "amount", "minimum"),
        ]
        assert entry.models == ("small", "large")
        assert entry.cost == Decimal("0.11")
        assert "amount: -2" in entry.reason

    def test_raw_answers_follow_the_snapshot_cap(self):
        run = Run(_contract(INTENT), 2, raw_answer_cap=1)
        run.answer(0, 1, _json(intent="upgrade"))
        run.answer(1, 1, _json(intent="upsell"))

        first, second = run.deliver().quarantine

        assert first.attempts[0].raw_answer is not None
        assert second.attempts[0].raw_answer is None
        assert second.failed_checks  # checks are exact whatever the cap

    def test_checks_fall_back_to_the_ledger_when_no_judged_answer_was_given(self):
        run = Run(_contract(INTENT), 1)
        run.answer(0, 1, _json(intent="upgrade"))
        run.judged.clear()

        (entry,) = run.deliver().quarantine

        assert [(c.attempt, c.field, c.rule) for c in entry.failed_checks] == [
            (None, None, None)
        ]
        assert "'upgrade'" in entry.failed_checks[0].text

    def test_quarantine_frame_has_one_row_per_entry(self):
        run = Run(_contract(INTENT), 2)
        run.answer(0, 1, _json(intent="upgrade"))

        frame = run.deliver().quarantine_to_pandas()

        assert list(frame["row"]) == [0, 1]
        assert list(frame["status"]) == ["rejected", "failed"]
        assert list(frame.columns) == [
            "row",
            "status",
            "reason",
            "sub_reason",
            "inputs",
            "raw_answers",
            "failed_checks",
            "models",
            "cost",
        ]

    def test_record_keeps_each_checks_attempt_field_and_rule(self):
        run = Run(_contract(INTENT, AMOUNT), 1)
        run.answer(0, 1, _json(intent="upgrade", amount=1))
        run.answer(0, 2, _json(intent="refund", amount=-2))

        (record,) = [e.to_record() for e in run.deliver().quarantine]

        assert record["failed_checks"] == [
            {
                "attempt": 1,
                "field": "intent",
                "rule": "allowed_values",
                "reason": "intent: 'upgrade' is not one of the 3 allowed values",
            },
            {
                "attempt": 2,
                "field": "amount",
                "rule": "minimum",
                "reason": "amount: -2 is below the minimum 0",
            },
        ]

    def test_record_keeps_the_provider_sub_reason(self):
        run = Run(_contract(INTENT), 1)
        run.answer(0, 1, _json(intent="upgrade"))
        run.provider_failure(0, 2)

        (entry,) = run.deliver().quarantine
        record = entry.to_record()

        assert entry.status is RowStatus.REJECTED
        assert record["sub_reason"] == PROVIDER_SUB_REASON
        assert "repair cut short by provider failure" in record["reason"]

    def test_record_sub_reason_is_none_when_the_ledger_gives_none(self):
        run = Run(_contract(INTENT), 1)
        run.answer(0, 1, _json(intent="upgrade"))

        (entry,) = run.deliver().quarantine

        assert entry.to_record()["sub_reason"] is None


# ---------------------------------------------------------------------------
# Run report
# ---------------------------------------------------------------------------


class TestReport:
    def _mixed_run(self) -> Run:
        run = Run(_contract(INTENT, AMOUNT), 6)
        run.answer(0, 1, _json(intent="refund", amount=1), cost="0.02")
        run.answer(1, 1, _json(intent="I'd say cancel", amount=1), cost="0.02")
        run.answer(2, 1, _json(intent="upgrade", amount=1), cost="0.03")
        run.answer(3, 1, _json(intent="upsell", amount=-5), cost="0.03")
        run.answer(4, 1, _json(intent="refund", amount=-1), cost="0.04")
        run.provider_failure(5, 1)
        return run

    def test_totals_coverage_and_invariant(self):
        report = self._mixed_run().deliver().report

        assert report.total == 6
        assert dict(report.counts) == {
            RowStatus.ACCEPTED: 1,
            RowStatus.REPAIRED: 1,
            RowStatus.REJECTED: 3,
            RowStatus.UNVERIFIED: 0,
            RowStatus.FAILED: 1,
            RowStatus.UNCHECKED: 0,
        }
        assert report.coverage == pytest.approx(2 / 6)
        assert report.invariant_holds
        assert "6 rows" in report.invariant_line
        assert "holds" in report.invariant_line

    def test_top_rejection_reasons_are_grouped_by_field_and_rule(self):
        report = self._mixed_run().deliver().report

        assert [(r.field, r.rule, r.count) for r in report.top_rejection_reasons] == [
            ("amount", "minimum", 2),
            ("intent", "allowed_values", 2),
        ]
        assert report.top_rejection_reasons[0].example.startswith("amount: ")

    def test_top_rejection_reasons_are_capped(self):
        report = self._mixed_run().deliver(top_reasons=1).report

        assert len(report.top_rejection_reasons) == 1

    def test_pass_rate_per_rule_over_each_rows_final_answer(self):
        report = self._mixed_run().deliver().report
        rates = {
            (r.field, r.rule): (r.passed, r.judged) for r in report.rule_pass_rates
        }

        assert rates[("intent", "allowed_values")] == (3, 5)
        assert rates[("amount", "minimum")] == (3, 5)
        assert rates[("intent", "required")] == (5, 5)
        assert rates[("amount", "type")] == (5, 5)

    def test_costs(self):
        report = self._mixed_run().deliver().report

        assert report.cost_total == Decimal("0.14")
        assert report.cost_per_delivered_row == Decimal("0.07")
        assert report.cost_on_rejected == Decimal("0.10")

    def test_warns_when_a_rule_fails_on_most_rows_and_rejection_is_high(self):
        run = Run(_contract(INTENT), 3)
        for row in range(3):
            run.answer(row, 1, _json(intent="upgrade"))

        warnings = run.deliver(rejection_warning_threshold=0.5).report.warnings

        assert any("intent.allowed_values failed on 3 of 3" in w for w in warnings)
        assert any("rejected 3 of 3 rows" in w for w in warnings)

    def test_a_clean_run_has_no_warnings(self):
        run = Run(_contract(INTENT), 2)
        run.answer(0, 1, _json(intent="refund"))
        run.answer(1, 1, _json(intent="cancel"))

        report = run.deliver().report

        assert report.warnings == ()
        assert report.coverage == 1.0
        assert report.cost_on_rejected == Decimal("0")

    def test_rendered_report_states_the_stop_and_the_invariant(self):
        run = Run(_contract(INTENT), 2)
        run.answer(0, 1, _json(intent="refund"))
        run.ledger.record_run_stopped(StopReason.EARLY_STOP)

        text = run.deliver().report.render()

        assert "early stop" in text
        assert "Totals:" in text
        assert "Coverage: 50.0%" in text

    def test_empty_input(self):
        run = Run(_contract(INTENT), 0)

        out = run.deliver()

        assert out.to_pandas().empty
        assert out.report.total == 0
        assert out.report.coverage is None
        assert out.report.cost_per_delivered_row is None
        assert out.report.invariant_holds

    def test_missing_judged_values_blank_the_row_and_are_reported(self):
        run = Run(_contract(INTENT), 1)
        run.answer(0, 1, _json(intent="refund"))
        run.judged.clear()

        out = run.deliver()

        assert pd.isna(out.to_pandas().loc[0, "intent"])
        assert any("no judged answer" in v for v in out.report.violations)
        assert "no judged answer" in out.report.render()

    @pytest.mark.parametrize("stray_row", [2, -1])
    def test_judged_answers_for_rows_outside_the_input_are_reported(self, stray_row):
        run = Run(_contract(INTENT), 2)
        run.answer(0, 1, _json(intent="refund"))
        run.answer(1, 1, _json(intent="cancel"))
        stray = JudgedAnswer.from_verdict(judge(run.contract, _json(intent="refund")))
        run.judged[(stray_row, 1)] = stray

        out = run.deliver()

        assert list(out.to_pandas()["intent"]) == ["refund", "cancel"]
        (violation,) = out.report.violations
        assert f"row {stray_row}, attempt 1" in violation
        assert "outside the 2 input rows" in violation


# ---------------------------------------------------------------------------
# Inputs that cannot be delivered, and determinism
# ---------------------------------------------------------------------------


class TestGuarantees:
    def test_input_row_count_must_equal_the_ledgers(self):
        run = Run(_contract(INTENT), 2)
        run.inputs.pop()

        with pytest.raises(ValueError, match="2 rows"):
            run.deliver()

    def test_same_snapshot_same_output(self):
        run = TestReport()._mixed_run()
        snapshot = run.ledger.snapshot()

        first = deliver(snapshot, run.judged, run.inputs, run.contract)
        second = deliver(snapshot, dict(run.judged), run.inputs, run.contract)

        assert first == second
        pd.testing.assert_frame_equal(first.to_pandas(), second.to_pandas())
        assert first.report.render() == second.report.render()

    def test_judged_mapping_order_does_not_matter(self):
        run = TestReport()._mixed_run()
        snapshot = run.ledger.snapshot()
        items = list(run.judged.items())
        random.Random(7).shuffle(items)  # noqa: S311 - a seeded test shuffle

        assert deliver(snapshot, dict(items), run.inputs, run.contract) == deliver(
            snapshot, run.judged, run.inputs, run.contract
        )


class TestOverRealPipelineOutput:
    """Delivery over a real run, not a hand-built frame.

    The spec's guarantee for the unchecked contract is that the frame is
    identical to today's. Comparing with a frame built by hand cannot prove
    that; comparing with what ``Pipeline.execute()`` returns can.
    """

    @pytest.mark.xfail(
        reason="needs step 2 verdict recording", raises=AssertionError, strict=True
    )
    def test_unchecked_frame_equals_the_pipelines_own_frame(self):
        failing = token_for(2)
        frame = conformance_frame(5)
        result = (
            PipelineBuilder.create()
            .from_dataframe(frame, input_columns=["marker"], output_columns=["answer"])
            .with_prompt("Answer for {marker}")
            .with_custom_llm_client(LedgerClient(fail_tokens={failing}))
            .with_max_retries(1)
            .with_batch_size(1)
            .with_concurrency(1)
            .build()
            .execute()
        )
        assert result.row_ledger is not None
        snapshot = result.row_ledger
        contract = build_contract(["answer"])
        inputs = frame.to_dict("records")
        # The judge is deterministic, so re-judging the recorded raw answers
        # gives the verdicts the run itself reached.
        judged = {
            (outcome.row, attempt.attempt): JudgedAnswer.from_verdict(
                judge(contract, attempt.raw_answer, inputs[outcome.row])
            )
            for outcome in snapshot.rows
            for attempt in outcome.attempts
            if attempt.raw_answer is not None
        }

        out = deliver(snapshot, judged, inputs, contract)

        today = result.to_pandas()[["marker", "answer"]]
        pd.testing.assert_frame_equal(out.to_pandas(), today)
        assert [o.status for o in snapshot.rows] == [
            RowStatus.UNCHECKED,
            RowStatus.UNCHECKED,
            RowStatus.FAILED,
            RowStatus.UNCHECKED,
            RowStatus.UNCHECKED,
        ]
        assert out.report.violations == ()


class TestJudgedAnswer:
    def test_from_verdict_keeps_values_outcomes_and_rules(self):
        contract = _contract(INTENT, AMOUNT, NOTE)
        verdict = judge(contract, _json(intent="REFUND", amount="x"))

        answer = JudgedAnswer.from_verdict(verdict)

        assert [(f.name, f.outcome) for f in answer.fields] == [
            ("intent", "value"),
            ("amount", "failed"),
            ("note", "absent"),
        ]
        assert answer.fields[0].value == "refund"
        assert answer.fields[1].failures == (
            CheckFailure(
                field="amount", rule="type", text="amount: 'x' is not a number"
            ),
        )
        assert not answer.passed
