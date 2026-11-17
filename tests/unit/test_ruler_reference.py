"""The Ruler's reference: how a cell is read, how a source becomes one, what it claims.

``read_reference`` reads each reference cell by the field's contract rules,
apart from comparing it. ``Reference`` is the one input of the comparison;
its kind, not which source made it, decides the claim.

Design: ``.claude/designs/certified-columns.md`` section 13.0.2.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from ondine.harness.contract import OutputField, build_contract
from ondine.harness.status import RowStatus
from ondine.ruler import CertificationError, ReferenceAnswers
from ondine.ruler.delivered import DeliveredValues
from ondine.ruler.ruler import certify, read_reference
from ondine.ruler.source import Reference

LABELS = ["refund", "billing"]
CONTRACT = build_contract(fields=[OutputField("intent", allowed=LABELS)])


def _read(raw, row=3):
    return read_reference(CONTRACT, "intent", raw, row)


# ---------------------------------------------------------------------------
# read_reference
# ---------------------------------------------------------------------------


def test_a_list_of_acceptable_answers_is_read_value_by_value():
    accepted, problems = _read(["refund", "billing"])

    assert accepted == ("refund", "billing")
    assert problems == []


def test_a_value_that_breaks_the_rules_is_collected_and_the_others_still_read():
    accepted, problems = _read(["refund", "complaint"], row=7)

    assert accepted == ("refund",)
    assert len(problems) == 1
    # A problem is data (what the sheet needs to locate the cell), and a line.
    (problem,) = problems
    assert (problem.field, problem.row, problem.raw) == ("intent", 7, "complaint")
    assert problem.reasons
    assert problem.line().startswith("row 7, intent: 'complaint' — ")


def test_a_value_that_needs_a_format_fix_is_refused_not_repaired():
    # A model's answer "I think it is billing." is repaired to billing; a
    # label is not: the caller must give the exact value.
    accepted, problems = _read("I think it is billing.")

    assert accepted == ()
    assert len(problems) == 1
    assert problems[0].raw == "I think it is billing."
    assert problems[0].reasons == ()
    assert problems[0].line().endswith("needs a format fix; give the exact value")


def test_an_empty_answer_is_skipped_and_leaves_the_cell_without_a_value():
    for empty in (None, "", "  ", [], [None, ""]):
        assert _read(empty) == ((), [])


# ---------------------------------------------------------------------------
# Reference: a model's answers, per field
# ---------------------------------------------------------------------------


def _answers(**overrides) -> ReferenceAnswers:
    fields = {
        "model": "reference-1",
        "prompt_fingerprint": "p",
        "run_id": "r",
        "fingerprint": "f",
        "rows": (0, 1, 2, 3),
        "values": {"a": {0: "x", 1: "y"}, "b": {0: 1}},
        "unresolved": {1: "provider", 2: "over the cost cap"},
        "cost": Decimal("0"),
    }
    return ReferenceAnswers(**{**fields, **overrides})


def test_a_models_per_row_causes_apply_to_every_field_the_row_lacks():
    reference = _answers().as_reference()

    assert reference.kind == "model"
    assert reference.model == "reference-1"
    assert reference.answers == {"a": {0: "x", 1: "y"}, "b": {0: 1}}
    # Row 1 answered "a" but not "b"; row 2 and 3 answered neither; row 3 has
    # no recorded cause.
    assert reference.causes == {
        "a": {2: "over the cost cap", 3: "no answer"},
        "b": {1: "provider", 2: "over the cost cap", 3: "no answer"},
    }


# ---------------------------------------------------------------------------
# The claim comes from the kind, not from which source made the reference
# ---------------------------------------------------------------------------


def _delivered() -> DeliveredValues:
    return DeliveredValues(
        run_id="run",
        row_count=2,
        statuses=(RowStatus.ACCEPTED, RowStatus.ACCEPTED),
        models=("m", "m"),
        values={"intent": ("refund", "billing")},
        fingerprint="f",
        checked=True,
        contract_fingerprint=CONTRACT.fingerprint,
    )


def _certify(reference: Reference):
    return certify(_delivered(), CONTRACT, ["intent"], [], reference)


ANSWERS = {"intent": {0: "refund", 1: "refund"}}


def test_the_kind_model_makes_an_audit_and_labels_make_a_certificate():
    audit = _certify(Reference("model", ANSWERS, model="reference-1"))
    labels = _certify(Reference("labels", ANSWERS))

    assert (audit.claim, labels.claim) == ("audit", "certificate")
    assert "agreement with reference-1" in audit.claim_text
    assert "your labels" in labels.claim_text
    # Same answers, same comparison: only the claim differs.
    assert audit["intent"].estimate == labels["intent"].estimate


def test_labels_report_no_unresolved_cause_even_when_the_reference_has_some():
    causes = {"intent": {1: "cannot tell"}}
    answers = {"intent": {0: "refund", 1: []}}  # row 1 is covered, unanswered

    labels = _certify(Reference("labels", answers, causes))
    model = _certify(Reference("model", answers, causes, model="m"))

    assert labels["intent"].unresolved_causes == {}
    assert model["intent"].unresolved_causes == {"cannot tell": 1}


def test_labels_from_people_are_refused_until_they_have_a_producer():
    with pytest.raises(CertificationError, match="people"):
        _certify(Reference("people", ANSWERS))
