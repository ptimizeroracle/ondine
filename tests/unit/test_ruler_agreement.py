"""Labeller agreement: Cohen's kappa, the agreement counts, and how they print.

Design: ``.claude/designs/certified-columns.md`` section 13.6.
"""

from __future__ import annotations

import pytest

from ondine.ruler.certificate import (
    LabellerAgreement,
    _render_agreement,
)
from ondine.ruler.ruler import Outcome, cohens_kappa, labeller_agreement


def _pairs(table: dict[tuple[str, str], int]) -> list[tuple[str, str]]:
    return [pair for pair, count in table.items() for _ in range(count)]


# ---------------------------------------------------------------------------
# Kappa against hand counts
# ---------------------------------------------------------------------------


def test_kappa_of_the_textbook_two_by_two_table():
    # A says yes on 25, B on 30, of 50. They agree on 20 yes + 15 no = 35.
    # Observed 35/50 = 0.7. By chance: (25*30 + 25*20) / 50**2 = 0.5.
    # kappa = (0.7 - 0.5) / (1 - 0.5) = 0.4.
    pairs = _pairs(
        {("yes", "yes"): 20, ("yes", "no"): 5, ("no", "yes"): 10, ("no", "no"): 15}
    )

    assert cohens_kappa(pairs) == pytest.approx(0.4)


def test_kappa_is_one_for_perfect_agreement_and_minus_one_for_a_swap():
    assert cohens_kappa([("a", "a"), ("b", "b")]) == 1.0
    assert cohens_kappa([("a", "b"), ("b", "a")]) == -1.0


def test_kappa_with_three_labels_by_hand():
    # 10 pairs: aa aa aa bb bb cc | ab ab bc ca. Observed 6/10.
    # A: a=5 b=3 c=2.  B: a=4 b=4 c=2.  Chance (5*4 + 3*4 + 2*2)/100 = 0.36.
    # kappa = (0.6 - 0.36) / 0.64 = 0.375.
    pairs = _pairs(
        {
            ("a", "a"): 3,
            ("b", "b"): 2,
            ("c", "c"): 1,
            ("a", "b"): 2,
            ("b", "c"): 1,
            ("c", "a"): 1,
        }
    )

    assert cohens_kappa(pairs) == pytest.approx(0.375)


def test_kappa_is_undefined_when_both_always_use_the_same_single_label():
    assert cohens_kappa([("a", "a")] * 5) is None


def test_kappa_counts_no_value_as_a_label_of_its_own():
    assert cohens_kappa([(None, None), ("a", "a"), (None, "a"), ("a", None)]) == 0.0


# ---------------------------------------------------------------------------
# Agreement counts
# ---------------------------------------------------------------------------


def _doubled(first, second, state="right", cause=None):
    return Outcome(state, "d", tuple(first), cause, (tuple(first), tuple(second)))


def test_agreement_counts_doubled_rows_both_answered_shared_and_split():
    outcomes = {
        0: _doubled(["a"], ["a"]),  # both, agree
        1: _doubled(["a"], ["b"], "unresolved", "labellers disagree"),  # both, split
        2: _doubled(["a", "b"], ["b"]),  # both, share b
        3: _doubled(["a"], []),  # one answered
        4: _doubled([], []),  # none
        5: Outcome("right", "d", ("a",)),  # not doubled: not counted
    }

    agreement = labeller_agreement(outcomes, 2, "labels", 0.0)

    assert (
        agreement.doubled,
        agreement.both_answered,
        agreement.agree,
        agreement.split,
    ) == (5, 3, 2, 1)
    # Kappa uses only the rows where each gave exactly one answer: (a,a), (a,b).
    # Observed 1/2; A always says a, B says a and b once: chance 1 * 0.5.
    assert agreement.kappa_rows == 2
    assert agreement.kappa == pytest.approx(0.0)


def test_numbers_agree_within_the_tolerance_and_have_no_kappa():
    outcomes = {
        0: _doubled([3], [4]),
        1: _doubled([3], [6]),
    }

    agreement = labeller_agreement(outcomes, 2, "integer", 1.0)

    assert (agreement.both_answered, agreement.agree) == (2, 1)
    assert agreement.kappa is None
    assert agreement.kappa_rows == 0


def test_two_answers_of_no_value_agree():
    agreement = labeller_agreement({0: _doubled([None], [None])}, 2, "labels", 0.0)

    assert agreement.agree == 1


def test_one_labeller_measures_nothing():
    agreement = labeller_agreement({0: Outcome("right", "d", ("a",))}, 1, "labels", 0.0)

    assert agreement == LabellerAgreement(1, 0, 0, 0, 0, None, 0)


# ---------------------------------------------------------------------------
# How it prints
# ---------------------------------------------------------------------------


def _text(**counts):
    base = dict(
        labellers=2,
        doubled=40,
        both_answered=40,
        agree=36,
        split=3,
        kappa=0.89,
        kappa_rows=37,
    )
    return _render_agreement(LabellerAgreement(**{**base, **counts}))


def test_the_line_matches_the_design_example():
    lines = _text()

    assert lines[0] == (
        "    Labellers: 2 (one sheet each); 40 rows labelled twice: agree on 36 "
        "(90.0%), kappa 0.89 on 37 single-answer rows; "
        "3 split on the delivered value (unresolved)"
    )
    assert len(lines) == 1


def test_rows_labelled_twice_but_answered_by_one_are_said():
    line = _text(doubled=42, both_answered=40)[0]

    assert "42 rows labelled twice, 40 answered by both: agree on 36 (90.0%)" in line


@pytest.mark.parametrize(
    ("agree", "kappa", "warned"),
    [
        (32, 0.9, False),  # 80.0% and kappa 0.9: no warning
        (31, 0.9, True),  # 77.5%
        (36, 0.6, False),  # kappa exactly 0.60
        (36, 0.59, True),
        (36, None, False),  # kappa undefined: raw agreement decides
    ],
)
def test_the_warning_shows_under_80_percent_agreement_or_kappa_below_0_6(
    agree, kappa, warned
):
    lines = _text(agree=agree, kappa=kappa)

    shown = any("Warning: labellers often disagree" in line for line in lines)
    assert shown is warned


def test_the_warning_says_the_labels_are_noisy_and_accuracy_loose():
    warning = _text(agree=20)[1]

    assert warning.startswith("    Warning: labellers often disagree;")
    assert "labels are noisy" in warning
    assert "accuracy against them is loose" in warning


def test_fewer_than_twenty_rows_answered_by_both_is_marked_few_rows():
    assert "few rows" in _text(doubled=19, both_answered=19, agree=19, kappa=1.0)[0]
    assert "few rows" not in _text(doubled=20, both_answered=20, agree=20, kappa=1.0)[0]


def test_an_undefined_kappa_is_said_not_hidden():
    line = _text(kappa=None, kappa_rows=40, agree=40)[0]

    assert "kappa not defined (one label used)" in line


def test_a_numeric_field_prints_no_kappa():
    line = _text(kappa=None, kappa_rows=0)[0]

    assert "kappa" not in line


def test_with_no_row_both_answered_agreement_is_not_measurable():
    line = _text(
        doubled=5, both_answered=0, agree=0, split=0, kappa=None, kappa_rows=0
    )[0]

    assert "no row answered by both" in line


def test_one_labeller_says_agreement_is_not_measured():
    assert _render_agreement(LabellerAgreement(1, 0, 0, 0, 0, None, 0)) == [
        "    Labellers: 1; agreement not measured (one labeller)"
    ]
