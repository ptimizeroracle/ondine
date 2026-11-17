"""The review sheet's pure parts: the cell syntax, the dealing, the item ids.

None of these runs a pipeline. The cell parser reads a person's typed cell by
the field's contract rules (the reading gold gets), the dealing decides who
labels which rows, and the hashes tell an edited input from a re-sorted one.

Design: ``.claude/designs/certified-columns.md`` sections 13.3-13.5.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from ondine.harness.contract import OutputField, build_contract
from ondine.harness.status import RowStatus
from ondine.ruler import CertificationError
from ondine.ruler.cells import read_cell
from ondine.ruler.delivered import DeliveredValues, _fingerprint
from ondine.ruler.ruler import read_reference
from ondine.ruler.sample import draw
from ondine.ruler.sheet import (
    deal,
    input_hash,
    item_id,
    sheet_id_of,
    sheet_seed,
)
from ondine.ruler.source import ABSENT

LABELS = ["refund", "billing"]


def _contract(**fields):
    declared = [
        OutputField("intent", allowed=LABELS, **fields.get("intent", {})),
        OutputField(
            "note", allowed=["x", "y"], required=False, **fields.get("note", {})
        ),
        OutputField("qty", type="integer", required=False),
        OutputField("price", type="number", minimum=0),
        OutputField("ok", type="boolean"),
    ]
    return build_contract(fields=declared)


CONTRACT = _contract()


def _cell(name, text, contract=CONTRACT):
    return read_cell(contract, name, text, 4)


# ---------------------------------------------------------------------------
# Cell syntax (13.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\t"])
def test_an_empty_cell_is_not_labelled(text):
    cell = _cell("intent", text)

    assert cell.state == "empty"
    assert cell.problems == ()


@pytest.mark.parametrize("text", ["?", " ? "])
def test_a_question_mark_alone_means_cannot_tell(text):
    assert _cell("intent", text).state == "unsure"


def test_one_answer_is_read_by_the_field_rules():
    cell = _cell("intent", " refund ")

    assert (cell.state, cell.raw, cell.problems) == ("answer", "refund", ())


def test_several_answers_split_on_the_bar_become_a_list():
    cell = _cell("intent", "refund | billing")

    assert (cell.state, cell.raw) == ("answer", ["refund", "billing"])


def test_an_allowed_value_with_a_bar_is_read_whole_before_any_split():
    contract = build_contract(fields=[OutputField("intent", allowed=["a | b", "c"])])

    whole = read_cell(contract, "intent", "a | b", 1)
    split = read_cell(contract, "intent", "a | b | c", 1)

    assert whole.raw == "a | b"
    # The whole cell is not an allowed value, so it is split: a, b, c, and a is not allowed.
    assert split.problems
    assert split.problems[0].raw == "a"


def test_a_bar_with_an_empty_part_or_a_question_mark_in_a_part_is_an_error():
    empty = _cell("intent", "refund | ")
    unsure = _cell("intent", "refund | ?")

    assert [p.reasons for p in empty.problems] == [("an empty answer after a '|'",)]
    assert "'?' must be the whole cell" in unsure.problems[0].reasons[0]


def test_every_bad_part_is_a_problem_and_the_good_ones_are_still_read():
    cell = _cell("intent", "refnd | billing | zzz")

    assert [p.raw for p in cell.problems] == ["refnd", "zzz"]


def test_a_dash_means_no_value_on_an_optional_field_only():
    optional = _cell("note", "-")
    required = _cell("intent", "-")

    assert (optional.state, optional.raw, optional.problems) == ("answer", ABSENT, ())
    assert required.problems
    assert "required" in required.problems[0].reasons[0]
    assert required.problems[0].raw == "-"


def test_a_dash_among_several_answers_is_one_of_them_on_an_optional_field():
    cell = _cell("note", "x | -")

    assert cell.raw == ["x", ABSENT]
    assert cell.problems == ()
    accepted, found = read_reference(CONTRACT, "note", cell.raw, 4)
    assert accepted == ("x", None)
    assert found == []


def test_the_fields_own_absent_token_also_means_no_value():
    contract = build_contract(
        fields=[OutputField("note", allowed=["x"], required=False, absent="n/a")]
    )
    cell = read_cell(contract, "note", "n/a", 2)
    accepted, found = read_reference(contract, "note", cell.raw, 2)

    assert accepted == (None,)
    assert found == []


@pytest.mark.parametrize("text", ["12", "3.5", "1e3", " 7 "])
def test_numbers_use_a_dot_and_read_as_typed(text):
    cell = _cell("price", text)

    assert (cell.state, cell.raw, cell.problems) == ("answer", text.strip(), ())
    accepted, found = read_reference(CONTRACT, "price", cell.raw, 4)
    assert accepted == (float(text),)
    assert found == []


@pytest.mark.parametrize("text", ["3,5", "12,25"])
def test_a_comma_decimal_is_an_error_with_the_hint_to_use_a_dot(text):
    cell = _cell("price", text)

    assert len(cell.problems) == 1
    assert cell.hint == "use a dot, not a comma, for decimals"


def test_a_number_that_breaks_the_fields_bounds_is_an_error():
    assert _cell("price", "-2").problems


def test_an_integer_field_refuses_a_fraction():
    assert _cell("qty", "3.5").problems


@pytest.mark.parametrize("text", ["true", "TRUE", "False", "false"])
def test_booleans_read_in_any_case(text):
    cell = _cell("ok", text)

    assert cell.problems == ()
    assert cell.state == "answer"


def test_a_boolean_that_is_not_one_is_an_error():
    assert _cell("ok", "maybe").problems


def test_labels_are_read_no_looser_than_gold_case_included():
    strict = _cell("intent", "Refund")
    loose = read_cell(
        _contract(intent={"normalise": ["casefold"]}), "intent", "Refund", 1
    )

    assert strict.problems
    assert loose.problems == ()


def test_a_label_near_miss_gets_the_closest_allowed_value_as_a_suggestion():
    near = _cell("intent", "refnd")
    case = _cell("intent", "Refund")
    far = _cell("intent", "zzzzzz")

    assert near.suggestions == {"refnd": "refund"}
    assert case.suggestions == {"Refund": "refund"}
    assert far.suggestions == {}


def test_a_suggestion_can_be_an_alias_spelling():
    contract = build_contract(
        fields=[OutputField("intent", allowed=LABELS, aliases={"money back": "refund"})]
    )

    cell = read_cell(contract, "intent", "money bak", 1)

    assert cell.suggestions == {"money bak": "money back"}


def test_a_cell_is_never_read_into_a_value_the_gold_reading_would_refuse():
    for text in ["I think it is billing.", "refund!"]:
        _, gold_problems = read_reference(CONTRACT, "intent", text, 4)
        assert gold_problems
        assert _cell("intent", text).problems


# ---------------------------------------------------------------------------
# Dealing (13.3)
# ---------------------------------------------------------------------------


def _deal(n=50, labellers=2, double=0.2, seed=123):
    return deal(list(range(1000, 1000 + n)), labellers, double, seed)


def test_fifty_rows_double_ten_and_deal_thirty_to_each_sheet():
    dealt = _deal(50)

    assert len(dealt.doubled) == 10
    assert [len(s) for s in dealt.sheets] == [30, 30]
    assert set(dealt.sheets[0]) & set(dealt.sheets[1]) == set(dealt.doubled)
    assert set(dealt.sheets[0]) | set(dealt.sheets[1]) == set(range(1000, 1050))


@pytest.mark.parametrize(
    ("n", "double", "expected"),
    [
        (15, 0.2, 3),
        (25, 0.28, 7),  # 0.28 * 25 is 7.000000000000001 in floats: not 8
        (100, 0.28, 28),  # and 28.000000000000004: not 29
        (12, 0.2, 3),  # 2.4 rounds up
        (50, 0.2, 10),
        (3, 0.2, 1),
        (1, 0.2, 1),
        (7, 1.0, 7),
        (10, 0.35, 4),
        (100, 0.3, 30),
    ],
)
def test_the_doubled_count_is_the_ceiling_without_float_noise(n, double, expected):
    assert len(_deal(n, double=double).doubled) == expected


def test_with_double_one_every_row_is_on_both_sheets():
    dealt = _deal(20, double=1.0)

    assert set(dealt.sheets[0]) == set(dealt.sheets[1]) == set(range(1000, 1020))


def test_one_labeller_gets_every_row_and_nothing_is_doubled():
    dealt = _deal(30, labellers=1)

    assert dealt.doubled == ()
    assert len(dealt.sheets) == 1
    assert sorted(dealt.sheets[0]) == list(range(1000, 1030))


def test_the_same_seed_deals_the_same_sheets_and_another_seed_does_not():
    assert _deal(seed=5) == _deal(seed=5)
    assert _deal(seed=5) != _deal(seed=6)


def test_each_sheet_is_shuffled_not_in_input_order():
    dealt = _deal(60)

    for rows in dealt.sheets:
        assert rows != sorted(rows)


def test_the_unshared_rows_alternate_between_the_labellers_in_input_order():
    dealt = _deal(50)
    shared = set(dealt.doubled)
    only_a = sorted(set(dealt.sheets[0]) - shared)
    only_b = sorted(set(dealt.sheets[1]) - shared)
    rest = sorted(set(range(1000, 1050)) - shared)

    assert only_a == rest[0::2]
    assert only_b == rest[1::2]


def test_the_sheet_seed_is_derived_and_a_whole_double_is_the_same_as_a_float():
    assert sheet_seed(77, 2, 1) == sheet_seed(77, 2, 1.0)
    assert sheet_seed(77, 2, 0.2) != sheet_seed(77, 2, 0.3)
    assert sheet_seed(77, 2, 0.2) != sheet_seed(77, 1, 0.2)
    assert sheet_seed(77, 2, 0.2) != sheet_seed(78, 2, 0.2)


def test_the_sheet_id_follows_the_rows_so_two_samples_of_one_run_differ():
    assert sheet_id_of(5, (1, 2, 3)) == sheet_id_of(5, (1, 2, 3))
    assert sheet_id_of(5, (1, 2, 3)) != sheet_id_of(5, (1, 2, 4))
    assert len(sheet_id_of(5, (1, 2, 3))) == 6


def test_item_ids_number_the_sheet_position_not_the_row():
    assert item_id("a91e3c", "A", 16) == "a91e3c-A017"
    assert item_id("a91e3c", "B", 1234) == "a91e3c-B1235"


# ---------------------------------------------------------------------------
# Input hashes: an edit shows, a re-sort does not (13.5)
# ---------------------------------------------------------------------------


def test_the_hash_ignores_outer_spaces_newline_style_and_unicode_form():
    base = input_hash("café\nline two")

    assert input_hash("  café\r\nline two \n") == base
    assert input_hash("café\rline two") == base


def test_the_hash_sees_every_other_edit():
    base = input_hash("a  b")

    assert input_hash("a b") != base
    assert input_hash("A  b") != base
    assert input_hash("a  b!") != base
    assert input_hash("") != base


# ---------------------------------------------------------------------------
# The sample, redrawn from the manifest (13.8)
# ---------------------------------------------------------------------------


def _view(n=300):
    values = {"intent": tuple(LABELS[i % 2] for i in range(n))}
    return DeliveredValues(
        run_id="run-1",
        row_count=n,
        statuses=(RowStatus.ACCEPTED,) * n,
        models=("m",) * n,
        values=values,
        fingerprint=_fingerprint(values),
        checked=True,
        contract_fingerprint="c" * 8,
    )


def _saved(sample):
    return {
        "seed": sample.seed,
        "rows": list(sample.rows),
        "population_size": sample.population_size,
        "rule": sample.rule,
        "fields": list(sample.fields),
    }


def test_a_sample_restores_to_the_same_rows_without_its_inputs():
    view = _view()
    drawn = draw(view, ["intent"], pd.DataFrame({"x": range(300)}), size=40)

    restored = type(drawn).restore(_saved(drawn), view, ["intent"])

    assert restored.rows == drawn.rows
    assert restored.seed == drawn.seed
    assert restored.rule == drawn.rule
    assert restored.census == drawn.census
    assert restored.run_id == drawn.run_id
    assert restored.fingerprint == drawn.fingerprint
    assert restored.inputs.empty
    assert list(restored.inputs.index) == list(drawn.rows)


def test_a_precision_sample_restores_with_its_own_rule():
    view = _view(1000)
    drawn = draw(view, ["intent"], pd.DataFrame({"x": range(1000)}), precision=0.1)

    restored = type(drawn).restore(_saved(drawn), view, ["intent"])

    assert restored.rows == drawn.rows
    assert restored.rule == drawn.rule


@pytest.mark.parametrize(
    "change",
    [
        lambda s: {**s, "rows": s["rows"][::-1]},
        lambda s: {**s, "seed": s["seed"] + 1},
        lambda s: {**s, "population_size": s["population_size"] + 1},
    ],
)
def test_a_sample_the_redraw_does_not_reproduce_is_refused(change):
    view = _view()
    drawn = draw(view, ["intent"], pd.DataFrame({"x": range(300)}), size=40)

    with pytest.raises(CertificationError, match="redraw"):
        type(drawn).restore(change(_saved(drawn)), view, ["intent"])


def test_a_view_of_other_values_than_the_sample_was_drawn_from_is_refused():
    view = _view()
    drawn = draw(view, ["intent"], pd.DataFrame({"x": range(300)}), size=40)
    other = _view(299)

    with pytest.raises(CertificationError):
        type(drawn).restore(_saved(drawn), other, ["intent"])


def test_check_refuses_a_sample_of_another_run_values_or_fields():
    view = _view()
    drawn = draw(view, ["intent"], pd.DataFrame({"x": range(300)}), size=40)

    with pytest.raises(CertificationError, match="drawn from run"):
        drawn.check(replace(view, run_id="run-2"), ["intent"])
    with pytest.raises(CertificationError, match="delivered values changed"):
        drawn.check(replace(view, fingerprint="0" * 64), ["intent"])
    with pytest.raises(CertificationError, match="not drawn for"):
        drawn.check(view, ["intent", "other"])
    drawn.check(view, ["intent"])
