"""Certified columns, end to end through a running pipeline (Ruler phase 2).

Every run uses the conformance provider with scripted answers, so each row's
status and delivered value are known in advance, and so is what a
certificate must say about them: which rows it compares, how many are right,
and the exact range.

Design: ``.claude/designs/certified-columns.md`` sections 11.1-11.4.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ondine import PipelineBuilder
from ondine.harness.contract import OutputField
from ondine.ruler import CertificationError
from ondine.ruler.intervals import hypergeom_bounds
from tests.e2e.conformance_harness import ProviderError, conformance_frame
from tests.e2e.test_contract_delivery_e2e import ScriptedClient

LABELS = ["refund", "billing"]

#: Five rows, one of each kind: accepted, repaired, rejected, lost, accepted.
ANSWERS = {0: "refund", 1: "I think it is billing.", 2: "a complaint", 4: "billing"}
LOST = 3
#: Gold for the five rows: row 1 is delivered "billing" but is a refund.
GOLD = ["refund", "refund", "billing", "refund", "billing"]


def _frame(rows: int, gold=None) -> pd.DataFrame:
    frame = conformance_frame(rows)
    if gold is not None:
        frame["true_intent"] = gold
    return frame


def _builder(client, frame: pd.DataFrame, *, columns=("intent",)):
    return (
        PipelineBuilder.create()
        .from_dataframe(frame, input_columns=["marker"], output_columns=list(columns))
        .with_prompt("Classify {marker}")
        .with_custom_llm_client(client)
        .with_batch_size(1)
        .with_concurrency(1)
        .with_max_retries(1)
    )


def _labelled(client, frame):
    return _builder(client, frame).with_output_field("intent", allowed=LABELS)


def _five_row_run():
    pipeline = _labelled(
        ScriptedClient(ANSWERS, fail_rows={LOST}), _frame(5, GOLD)
    ).build()
    return pipeline, pipeline.execute()


def _big_run(rows: int, wrong_every: int = 7):
    """``rows`` accepted rows; the gold disagrees on every ``wrong_every``-th."""
    answers = {row: LABELS[row % 2] for row in range(rows)}
    gold = [
        LABELS[(row + 1) % 2] if row % wrong_every == 0 else LABELS[row % 2]
        for row in range(rows)
    ]
    pipeline = _labelled(ScriptedClient(answers), _frame(rows, gold)).build()
    return pipeline, pipeline.execute(), gold


# ---------------------------------------------------------------------------
# 4. Census: only delivered rows are certified, with exact status lines
# ---------------------------------------------------------------------------


def test_a_census_compares_only_the_rows_that_delivered_a_value():
    pipeline, result = _five_row_run()

    cert = pipeline.certify(result, gold="true_intent")

    intent = cert["intent"]
    assert cert.claim == "certificate"
    assert "every delivered row" in cert.claim_text
    assert (intent.estimate.population, intent.estimate.compared) == (3, 3)
    assert (intent.estimate.right, intent.estimate.wrong) == (2, 1)
    assert intent.estimate.exact
    assert intent.not_delivered == {"rejected": 1, "failed": 1}


def test_each_status_gets_its_own_line_and_the_wrong_row_is_named():
    pipeline, result = _five_row_run()

    intent = pipeline.certify(result, gold="true_intent")["intent"]

    lines = {g.name: (g.estimate.right, g.estimate.resolved) for g in intent.by_status}
    assert lines == {"accepted": (2, 2), "repaired": (0, 1)}
    assert [(w.row, w.delivered, w.reference) for w in intent.wrong_rows] == [
        (1, "billing", ("refund",))
    ]
    assert [(c.delivered, c.reference, c.count) for c in intent.confusions] == [
        ("billing", "refund", 1)
    ]


def test_an_unverified_row_is_delivered_so_it_is_certified():
    client = ScriptedClient(ANSWERS, fail_rows={LOST})
    pipeline = (
        _labelled(client, _frame(5, GOLD))
        .with_contract_policy(on_failure="deliver_flagged")
        .build()
    )
    result = pipeline.execute()

    intent = pipeline.certify(
        result, gold={0: "refund", 1: "refund", 2: "billing", 4: "billing"}
    )["intent"]

    # Row 2's "a complaint" was delivered flagged: it is compared, and its
    # value is not an allowed label, so it cannot equal any gold label.
    assert "unverified" in {g.name for g in intent.by_status}
    assert intent.estimate.population == 4
    assert intent.estimate.wrong == 2


# ---------------------------------------------------------------------------
# 5. The sample: Ondine draws it, from this run, once
# ---------------------------------------------------------------------------


def test_drawing_twice_on_one_result_gives_the_same_rows():
    pipeline, result, _ = _big_run(300)

    first = pipeline.draw_sample(result, size=50)
    again = pipeline.draw_sample(result, size=50)

    assert first.rows == again.rows
    assert first.seed == again.seed
    assert len(first.rows) == 50
    assert list(first.inputs.columns) == ["marker"]
    assert list(first.inputs.index) == list(first.rows)


def test_a_new_run_of_the_same_data_draws_its_own_rows():
    pipeline, result, _ = _big_run(300)
    rerun = pipeline.execute()

    assert (
        pipeline.draw_sample(result, size=50).rows
        != pipeline.draw_sample(rerun, size=50).rows
    )


def test_a_sample_certifies_its_rows_with_the_exact_range():
    pipeline, result, gold = _big_run(300)

    sample = pipeline.draw_sample(result, size=60)
    labels = {row: gold[row] for row in sample.rows}
    cert = pipeline.certify(result, gold=labels, sample=sample)
    est = cert["intent"].estimate

    # Whether the range holds the truth is a 95% property (benchmarks/ruler);
    # here: the right rows are counted exactly and the range is the exact one.
    assert "sample Ondine drew" in cert.claim_text
    assert (est.population, est.compared) == (300, 60)
    assert est.right == sum(LABELS[row % 2] == gold[row] for row in sample.rows)
    lo, hi = hypergeom_bounds([est.right], 60, 300)
    assert (est.low, est.high) == (lo[0] / 300, hi[0] / 300)


def test_sampled_rows_without_a_label_widen_the_range_both_ways():
    pipeline, result, gold = _big_run(300)
    sample = pipeline.draw_sample(result, size=60)
    unlabelled = set(sample.rows[:5])
    labels = {row: gold[row] for row in sample.rows if row not in unlabelled}

    est = pipeline.certify(result, gold=labels, sample=sample)["intent"].estimate

    assert est.unresolved == 5
    lo, _ = hypergeom_bounds([est.right], 60, 300)
    _, hi = hypergeom_bounds([est.right + 5], 60, 300)
    assert (est.low, est.high) == (lo[0] / 300, hi[0] / 300)


def test_a_sample_drawn_from_another_run_is_refused():
    pipeline, result, gold = _big_run(300)
    other = pipeline.execute()
    sample = pipeline.draw_sample(other, size=30)

    with pytest.raises(CertificationError, match="drawn from run"):
        pipeline.certify(result, gold="true_intent", sample=sample)


def test_labels_on_rows_you_picked_give_a_check_with_no_range():
    pipeline, result, gold = _big_run(300)

    cert = pipeline.certify(result, gold={row: gold[row] for row in range(40)})

    assert cert.claim == "check"
    assert "not a claim about the column" in cert.claim_text
    est = cert["intent"].estimate
    assert (est.compared, est.low, est.high) == (40, None, None)


def test_a_population_no_larger_than_the_size_is_taken_whole():
    pipeline, result, gold = _big_run(30)

    sample = pipeline.draw_sample(result, size=200)
    est = pipeline.certify(
        result, gold={row: gold[row] for row in sample.rows}, sample=sample
    )["intent"].estimate

    assert sample.census
    assert est.exact


def test_a_precision_target_sets_the_size_before_drawing():
    pipeline, result, _ = _big_run(1000)

    sample = pipeline.draw_sample(result, precision=0.1)

    lo, hi = hypergeom_bounds([len(sample.rows) // 2], len(sample.rows), 1000)
    assert (hi[0] - lo[0]) / 1000 / 2 <= 0.1
    smaller = len(sample.rows) - 1
    lo, hi = hypergeom_bounds([smaller // 2], smaller, 1000)
    assert (hi[0] - lo[0]) / 1000 / 2 > 0.1


# ---------------------------------------------------------------------------
# 6. Bad labels are all listed at once
# ---------------------------------------------------------------------------


def test_every_label_that_breaks_the_rules_is_listed_in_one_error():
    pipeline, result = _five_row_run()

    with pytest.raises(CertificationError) as raised:
        pipeline.certify(result, gold={0: "refnd", 1: "refund", 4: "a complaint"})

    message = str(raised.value)
    assert "2 reference value(s)" in message
    assert "row 0" in message
    assert "row 4" in message


def test_a_label_read_through_an_alias_counts_as_that_value():
    client = ScriptedClient({0: "refund", 1: "billing"})
    pipeline = (
        _builder(client, _frame(2))
        .with_output_field(
            "intent",
            allowed=LABELS,
            aliases={"money back": "refund"},
            normalise=["casefold"],
        )
        .build()
    )
    result = pipeline.execute()

    # The label goes through the field's cleanups and aliases, as an answer would.
    est = pipeline.certify(result, gold=["Money back", "billing"])["intent"].estimate

    assert est.right == 2


def test_a_label_is_read_by_the_same_rules_as_an_answer_and_no_looser():
    pipeline, result = _five_row_run()

    # No casefold declared: "Refund" breaks the rules for a label as it would
    # for an answer, instead of being matched loosely.
    with pytest.raises(CertificationError, match="row 0"):
        pipeline.certify(result, gold={0: "Refund", 1: "refund", 4: "billing"})


# ---------------------------------------------------------------------------
# 7. Several fields: one population each, and row accuracy
# ---------------------------------------------------------------------------


def _two_field_run():
    answers = {
        0: json.dumps({"intent": "refund", "urgency": 2}),
        1: json.dumps({"intent": "billing", "urgency": 9}),  # urgency fails
        2: json.dumps({"intent": "refund", "urgency": 4}),
    }
    pipeline = (
        _builder(ScriptedClient(answers), _frame(3), columns=("intent", "urgency"))
        .with_output_field("intent", allowed=LABELS)
        .with_output_field("urgency", type="integer", minimum=1, maximum=5)
        .with_contract_policy(partial_rows="keep_valid_fields")
        .build()
    )
    return pipeline, pipeline.execute()


def test_each_field_is_certified_on_the_rows_that_delivered_it():
    pipeline, result = _two_field_run()

    cert = pipeline.certify(
        result,
        gold={"intent": ["refund", "billing", "billing"], "urgency": [2, 3, 4]},
    )

    assert cert["intent"].estimate.population == 3  # row 1 still ships intent
    assert cert["urgency"].estimate.population == 2
    assert cert["intent"].estimate.right == 2
    assert cert["urgency"].estimate.right == 2
    # Rows delivering both fields: 0 and 2; row 2's intent is wrong.
    assert (cert.row_accuracy.population, cert.row_accuracy.right) == (2, 1)


def test_numbers_compare_within_the_tolerance_given():
    pipeline, result = _two_field_run()
    gold = {"urgency": [3, None, 4]}

    strict = pipeline.certify(result, gold=gold)["urgency"].estimate
    loose = pipeline.certify(result, gold=gold, tolerance={"urgency": 1})[
        "urgency"
    ].estimate

    assert (strict.right, loose.right) == (1, 2)


def test_a_list_of_acceptable_answers_matches_any_of_them():
    pipeline, result = _five_row_run()

    est = pipeline.certify(
        result, gold={0: "refund", 1: ["refund", "billing"], 4: "billing"}
    )["intent"].estimate

    assert est.right == 3


# ---------------------------------------------------------------------------
# 9. A run with no contract
# ---------------------------------------------------------------------------


def test_a_run_with_no_contract_needs_declarations_to_be_certified():
    pipeline = _builder(
        ScriptedClient(ANSWERS, fail_rows={LOST}), _frame(5, GOLD)
    ).build()
    result = pipeline.execute()

    with pytest.raises(CertificationError, match="no output contract"):
        pipeline.certify(result, gold="true_intent")

    cert = pipeline.certify(
        result, gold="true_intent", fields=[OutputField("intent", allowed=LABELS)]
    )
    est = cert["intent"].estimate
    # Without a contract every answer was delivered as text: "I think it is
    # billing." reads as billing (wrong, gold says refund), "a complaint"
    # cannot be read (wrong); rows 0 and 4 are right.
    assert (est.population, est.right) == (4, 2)


# ---------------------------------------------------------------------------
# 10-11. The run is untouched; a resumed run certifies the same
# ---------------------------------------------------------------------------


def test_certifying_changes_neither_the_frame_nor_the_run_report():
    pipeline, result = _five_row_run()
    frame = result.to_pandas().copy()
    report = result.report.render()

    sample = pipeline.draw_sample(result, size=2)
    pipeline.certify(result, gold="true_intent")
    pipeline.certify(
        result, gold={row: GOLD[row] for row in sample.rows}, sample=sample
    )

    pd.testing.assert_frame_equal(result.to_pandas(), frame)
    assert result.report.render() == report


def test_a_resumed_run_certifies_like_an_uninterrupted_one(tmp_path):
    answers = {**ANSWERS, LOST: "refund"}
    first = (
        _labelled(ScriptedClient(ANSWERS, fail_rows={LOST}), _frame(5, GOLD))
        .with_error_policy("fail")
        .with_checkpoint_dir(str(tmp_path))
        .with_checkpoint_interval(1)
        .build()
    )
    with pytest.raises(ProviderError):
        first.execute()
    resumed_pipeline = (
        _labelled(ScriptedClient(answers), _frame(5, GOLD))
        .with_checkpoint_dir(str(tmp_path))
        .build()
    )
    resumed = resumed_pipeline.execute(resume_from=first.session_id)
    straight_pipeline = _labelled(ScriptedClient(answers), _frame(5, GOLD)).build()
    straight = straight_pipeline.execute()

    a = resumed_pipeline.certify(resumed, gold="true_intent")["intent"].estimate
    b = straight_pipeline.certify(straight, gold="true_intent")["intent"].estimate

    assert (
        (a.population, a.right, a.wrong)
        == (b.population, b.right, b.wrong)
        == (4, 3, 1)
    )


def test_the_rendered_certificate_states_the_claim_and_its_basis():
    pipeline, result, gold = _big_run(300)
    sample = pipeline.draw_sample(result, size=60)

    text = pipeline.certify(
        result, gold={row: gold[row] for row in sample.rows}, sample=sample
    ).render()

    assert text.startswith(
        "Certificate: accuracy against your labels, on a sample Ondine drew"
    )
    assert "95% range" in text
    assert "sample of 60 of 300 delivered rows" in text
    assert f"seed {sample.seed}" in text


def test_a_slice_of_the_gold_column_labels_the_sample_by_row():
    pipeline, result, gold = _big_run(300)
    sample = pipeline.draw_sample(result, size=60)

    column = result.to_pandas()["true_intent"]
    by_slice = pipeline.certify(result, gold=column[list(sample.rows)], sample=sample)
    by_mapping = pipeline.certify(
        result, gold={row: gold[row] for row in sample.rows}, sample=sample
    )

    assert by_slice["intent"].estimate == by_mapping["intent"].estimate
