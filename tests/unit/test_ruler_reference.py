"""The Ruler's reference: how a cell is read, how a source becomes one, what it claims.

``read_reference`` reads each reference cell by the field's contract rules,
apart from comparing it. ``Reference`` is the one input of the comparison;
its kind, not which source made it, decides the claim.

Design: ``.claude/designs/certified-columns.md`` section 13.0.2.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pandas as pd

from ondine.harness.contract import OutputField, build_contract
from ondine.harness.status import RowStatus
from ondine.ruler import ReferenceAnswers
from ondine.ruler.delivered import DeliveredValues
from ondine.ruler.ruler import certify, read_reference
from ondine.ruler.sample import Sample
from ondine.ruler.source import ABSENT, Labellers, Reference

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


def test_labels_from_people_make_a_certificate_worded_for_blind_labels():
    people = _certify(Reference("people", ANSWERS, sheet="a91e3c"))

    assert people.claim == "certificate"
    assert people.claim_text == (
        "accuracy against blind human labels, on every delivered row"
    )
    assert people.sheet == "a91e3c"
    labels = _certify(Reference("labels", ANSWERS))
    assert people["intent"].estimate == labels["intent"].estimate


def test_a_people_reference_reports_causes_per_field_and_labels_stay_silent():
    answers = {"intent": {0: "refund", 1: []}}
    causes = {"intent": {1: "not labelled"}}

    people = _certify(Reference("people", answers, causes))

    assert people["intent"].unresolved_causes == {"not labelled": 1}


# ---------------------------------------------------------------------------
# Two labellers on a row: a second answer set, and a cause only comparing can know
# ---------------------------------------------------------------------------


def _doubled(first, second, delivered=("refund", "billing")):
    """Two rows both labellers were given: ``first`` is A's answers, ``second`` B's."""
    view = replace(_delivered(), values={"intent": delivered})
    reference = Reference(
        "people",
        {"intent": first},
        {"intent": {0: "cannot tell", 1: "cannot tell"}},
        labellers=Labellers(2, frozenset({0, 1}), {"intent": second}),
    )
    sample = Sample("run", "f", 1, ("intent",), 2, (0, 1), "census", pd.DataFrame())
    return certify(view, CONTRACT, ["intent"], [], reference, sample=sample)["intent"]


def test_each_labellers_answers_are_judged_against_the_delivered_value_on_their_own():
    # Row 0 delivered refund: both right. Row 1 delivered billing: A right, B wrong.
    field = _doubled({0: "refund", 1: "billing"}, {0: "refund", 1: "refund"})

    assert (field.estimate.right, field.estimate.unresolved) == (1, 1)
    assert field.unresolved_causes == {"labellers disagree": 1}
    assert field.labellers.split == 1


def test_both_wrong_is_wrong_naming_the_first_labellers_answers_then_the_seconds_others():
    # Delivered refund and billing; row 0 both say billing, row 1 A refund, B billing/refund.
    field = _doubled(
        {0: "billing", 1: "refund"},
        {0: "billing", 1: ["refund", "billing"]},
        delivered=("refund", "complaint"),
    )

    assert field.estimate.wrong == 2
    assert [(w.row, w.reference) for w in field.wrong_rows] == [
        (0, ("billing",)),
        (1, ("refund", "billing")),
    ]


def test_an_answer_from_one_labeller_counts_and_none_uses_the_references_cause():
    one = _doubled({0: "refund"}, {1: "billing"})
    nobody = _doubled({}, {})

    assert (one.estimate.right, one.estimate.unresolved) == (2, 0)
    assert nobody.estimate.unresolved == 2
    assert nobody.unresolved_causes == {"cannot tell": 2}
    assert nobody.labellers.both_answered == 0


def test_no_value_is_read_by_an_optional_field_and_refused_by_a_required_one():
    optional = build_contract(
        fields=[OutputField("note", allowed=["x"], required=False)]
    )
    required = build_contract(fields=[OutputField("note", allowed=["x"])])

    assert read_reference(optional, "note", ABSENT, 3) == ((None,), [])
    assert read_reference(optional, "note", ["x", ABSENT], 3) == (("x", None), [])
    accepted, problems = read_reference(required, "note", ABSENT, 3)
    assert accepted == ()
    assert problems[0].raw == "-"
    assert "required" in problems[0].reasons[0]
