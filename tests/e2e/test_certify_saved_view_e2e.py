"""A run certified from its saved view, in a freshly built pipeline.

Labelling takes days, and an ``ExecutionResult`` neither pickles nor outlives
its process. The Ruler's own view of the run (``DeliveredValues``) does: it
goes out as JSON and certifies, in a pipeline built again from the same
code, exactly as the in-session result does.

Design: ``.claude/designs/certified-columns.md`` section 13.0.1 and 13.13.
"""

from __future__ import annotations

import json

import pytest

from ondine.harness.contract import OutputField
from ondine.ruler import CertificationError
from ondine.ruler.delivered import DeliveredValues
from tests.e2e.test_certify_e2e import (
    ANSWERS,
    GOLD,
    LABELS,
    LOST,
    _big_run,
    _builder,
    _frame,
    _labelled,
    _two_field_run,
)
from tests.e2e.test_contract_delivery_e2e import ScriptedClient


def _fresh_pipeline():
    """The five-row pipeline, built again from the same code, never executed."""
    return _labelled(ScriptedClient(ANSWERS, fail_rows={LOST}), _frame(5, GOLD)).build()


def _five_row():
    pipeline = _fresh_pipeline()
    return pipeline, pipeline.execute()


def _saved(pipeline, result, **options) -> DeliveredValues:
    """The view through JSON text, as a file written days ago and read now."""
    view = pipeline.save_view(result, **options)
    return DeliveredValues.from_dict(json.loads(json.dumps(view.to_dict())))


def test_a_run_certified_from_a_json_round_trip_matches_the_in_session_certificate():
    pipeline, result = _five_row()
    in_session = pipeline.certify(result, gold=GOLD)

    restored = _saved(pipeline, result)
    later = _fresh_pipeline().certify(restored, gold=GOLD)

    assert later.to_dict() == in_session.to_dict()
    assert later.render() == in_session.render()
    assert later["intent"].not_delivered == {"rejected": 1, "failed": 1}


def test_a_sample_drawn_on_the_result_certifies_against_the_restored_view():
    pipeline, result, gold = _big_run(300)
    sample = pipeline.draw_sample(result, size=60)
    labels = {row: gold[row] for row in sample.rows}
    in_session = pipeline.certify(result, gold=labels, sample=sample)

    from_view = pipeline.certify(_saved(pipeline, result), gold=labels, sample=sample)

    assert from_view.claim_text == in_session.claim_text
    assert from_view.to_dict() == in_session.to_dict()


def test_a_sample_from_another_run_is_still_refused_against_a_view():
    pipeline, result, gold = _big_run(300)
    other = pipeline.draw_sample(pipeline.execute(), size=30)

    with pytest.raises(CertificationError, match="drawn from run"):
        pipeline.certify(_saved(pipeline, result), gold=gold, sample=other)


def test_a_value_changed_in_the_saved_file_is_refused():
    pipeline, result = _five_row()
    data = json.loads(json.dumps(pipeline.save_view(result).to_dict()))
    data["values"]["intent"][0] = "billing"

    with pytest.raises(CertificationError, match="fingerprint"):
        DeliveredValues.from_dict(data)


def test_a_view_from_another_contract_is_refused():
    pipeline, result = _five_row()
    restored = _saved(pipeline, result)
    other = (
        _builder(ScriptedClient(ANSWERS), _frame(5, GOLD))
        .with_output_field("intent", allowed=[*LABELS, "complaint"])
        .build()
    )

    # The contract fingerprint covers the declarations and the policy.
    with pytest.raises(CertificationError, match="declarations and contract policy"):
        other.certify(restored, gold=GOLD)


def test_a_view_of_a_checked_run_is_refused_by_a_pipeline_with_no_contract():
    pipeline, result = _five_row()
    bare = _builder(ScriptedClient(ANSWERS), _frame(5, GOLD)).build()

    with pytest.raises(CertificationError, match="another output contract"):
        bare.certify(_saved(pipeline, result), gold=GOLD)


def test_a_column_name_as_gold_needs_the_result_frame_and_is_refused_with_a_view():
    pipeline, result = _five_row()

    with pytest.raises(CertificationError, match="no result frame"):
        pipeline.certify(_saved(pipeline, result), gold="true_intent")


def test_labels_given_as_a_sequence_or_a_mapping_both_work_with_a_view():
    pipeline, result = _five_row()
    restored = _saved(pipeline, result)
    mapping = {0: "refund", 1: ["refund", "billing"], 4: "billing"}

    assert (
        pipeline.certify(restored, gold=GOLD).to_dict()
        == pipeline.certify(result, gold=GOLD).to_dict()
    )
    assert (
        pipeline.certify(restored, gold=mapping).to_dict()
        == pipeline.certify(result, gold=mapping).to_dict()
    )
    with pytest.raises(CertificationError, match="5 rows"):
        pipeline.certify(restored, gold=GOLD[:4])


def test_two_fields_round_trip_and_a_sample_binds_to_the_same_values():
    pipeline, result = _two_field_run()
    gold = {"intent": ["refund", "billing", "billing"], "urgency": [2, 3, 4]}
    sample = pipeline.draw_sample(result, size=2)
    in_session = pipeline.certify(result, gold=gold, sample=sample)

    view = _saved(pipeline, result)

    # The one creation path hashes the pipeline's certifiable fields, as the
    # result does, so the sample drawn on the result is not refused.
    assert view.fingerprint == sample.fingerprint
    assert pipeline.certify(view, gold=gold, sample=sample).to_dict() == (
        in_session.to_dict()
    )


def test_a_view_saved_with_other_fields_than_the_pipelines_is_refused_both_ways():
    pipeline, result = _two_field_run()
    gold = {"intent": ["refund", "billing", "billing"], "urgency": [2, 3, 4]}
    missing = DeliveredValues.read(result, ["intent"])
    extra = DeliveredValues.read(result, ["intent", "urgency", "status"])

    with pytest.raises(CertificationError, match="missing 'urgency'"):
        pipeline.certify(missing, gold=gold)
    with pytest.raises(CertificationError, match="extra 'status'"):
        pipeline.certify(extra, gold=gold)


def test_a_run_with_no_contract_needs_the_same_declarations_with_a_view():
    pipeline = _builder(
        ScriptedClient(ANSWERS, fail_rows={LOST}), _frame(5, GOLD)
    ).build()
    result = pipeline.execute()
    declarations = [OutputField("intent", allowed=LABELS)]
    in_session = pipeline.certify(result, gold=GOLD, fields=declarations)

    with pytest.raises(CertificationError, match="no output contract"):
        pipeline.save_view(result)
    restored = _saved(pipeline, result, fields=declarations)

    assert not restored.checked
    assert restored.contract_fingerprint is None
    assert pipeline.certify(restored, gold=GOLD, fields=declarations).to_dict() == (
        in_session.to_dict()
    )
    with pytest.raises(CertificationError, match="no output contract"):
        pipeline.certify(restored, gold=GOLD)


def test_a_sample_cannot_be_drawn_from_a_view_which_holds_no_inputs():
    pipeline, result = _five_row()

    with pytest.raises(CertificationError, match="no input columns"):
        pipeline.draw_sample(_saved(pipeline, result))
