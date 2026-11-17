"""The human pilot script, run on a synthetic 100-row layout.

No real labels exist yet, so two scripted "people" fill the 100 sheet rows
from the dataset's own labels, with known mistakes. The script's numbers must
equal a hand count of the same rows. Real data (the golden intent set and
Haiku's recorded answers) but no model call; skipped when the golden data is
absent. The script writes only where it is told: never ``results/``.

Design: ``.claude/designs/certified-columns.md`` section 13.9.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
from pathlib import Path

import openpyxl
import pytest

from ondine.ruler import ReviewSheetError

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "benchmarks" / "ruler" / "pilot.py"

pytestmark = pytest.mark.skipif(
    not (
        ROOT / "benchmarks" / "golden" / "data" / "labels" / "intent" / "haiku.jsonl"
    ).exists(),
    reason="golden data not built (benchmarks/golden/build.py, label.py)",
)


@pytest.fixture(scope="module")
def pilot():
    spec = importlib.util.spec_from_file_location("ruler_pilot", PILOT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _row_of(marker: str) -> int:
    return int(marker.split("-")[1])


def _fill(path: Path, answer) -> None:
    """Write each row's label, reading only the marker, as a person reads a row."""
    book = openpyxl.load_workbook(path)
    sheet = book["Labels"]
    header = [c.value for c in sheet[1]]
    marker, label = header.index("input: marker"), header.index("label: label")
    for r in range(2, sheet.max_row + 1):
        sheet.cell(r, label + 1).value = answer(
            _row_of(sheet.cell(r, marker + 1).value)
        )
    book.save(path)


def _people(pilot, rows, gold, allowed, wrong_every):
    """A answers every adjudicated row right and marks the dropped rows ``?``.
    B answers every row, wrong on each ``wrong_every``-th, several on some."""
    dev = {i: r["label"] for i, r in enumerate(rows)}

    def other(row):
        return next(a for a in allowed if a != dev[row] and a != gold.get(row))

    def a(row):
        return gold.get(row, "?")

    def b(row):
        truth = gold.get(row, dev[row])
        if row % wrong_every == 0:
            return other(row)
        if row % 25 == 3:
            return f"{other(row)} | {truth}"
        return truth

    return a, b


def _run(pilot, tmp_path, wrong_every):
    pipeline, rows, adjudicated = pilot.replay()
    gold = {
        i: adjudicated[r["id"]] for i, r in enumerate(rows) if r["id"] in adjudicated
    }
    allowed = sorted({r["label"] for r in rows})
    review = pilot.export(tmp_path / "p")
    a, b = _people(pilot, rows, gold, allowed, wrong_every)
    _fill(review.sheets["A"], a)
    _fill(review.sheets["B"], b)
    return rows, gold, review


def _hand_count(rows, gold, sample, wrong_every):
    covered = [r for r in sample if r in gold]
    dropped = [r for r in sample if r not in gold]
    several = [r for r in sample if r % wrong_every != 0 and r % 25 == 3]
    several_covered = [r for r in covered if r in several]
    b_agree = [r for r in covered if r % wrong_every != 0 and r % 25 != 3]
    return covered, dropped, several, b_agree, several_covered


def test_the_export_gives_both_people_the_same_one_hundred_rows(pilot, tmp_path):
    results = ROOT / "benchmarks" / "ruler" / "results"
    before = sorted(p.name for p in results.iterdir())

    review = pilot.export(tmp_path / "p")

    markers = {
        letter: sorted(
            _row_of(row[1].value)
            for row in openpyxl.load_workbook(path)["Labels"].iter_rows(min_row=2)
        )
        for letter, path in review.sheets.items()
    }
    assert (review.rows, review.doubled) == (100, 100)
    assert markers["A"] == markers["B"]
    assert len(markers["A"]) == 100
    assert sorted(p.name for p in results.iterdir()) == before


def test_the_scores_equal_a_hand_count_and_a_good_pair_passes(pilot, tmp_path):
    rows, gold, review = _run(pilot, tmp_path, wrong_every=20)
    report = tmp_path / "pilot.md"

    numbers = pilot.score(tmp_path / "p", 50, 80, report)

    sample = pilot.load_review(tmp_path / "p").manifest.sample["rows"]
    covered, dropped, several, b_agree, b_several = _hand_count(rows, gold, sample, 20)
    a, b = numbers["people"]["A"], numbers["people"]["B"]
    assert (a["answered"], a["covered"], a["agree"]) == (
        100 - len(dropped),
        len(covered),
        len(covered),
    )
    assert (b["answered"], b["covered"], b["agree"]) == (
        100,
        len(covered),
        len(b_agree),
    )
    assert (a["several"], b["several"]) == (0, len(b_several))
    assert (a["per_row"], b["per_row"]) == (0.5, 0.8)
    assert numbers["passed"] is (len(b_agree) / len(covered) >= pilot.PASS_RATE)
    assert numbers["unsure_rows"] == len(dropped)
    assert numbers["unsure_dropped"] == len(dropped)
    assert numbers["sampled_dropped"] == len(dropped)
    assert numbers["sampled_dropped_flagged"] == len(dropped)
    assert numbers["several_rows"] == len(several)
    assert numbers["agreement"].doubled == 100
    text = report.read_text()
    assert f"| B | 100 | {len(covered)} | {len(b_agree)} |" in text
    for letter in "AB":
        path = tmp_path / "p" / "for_labellers" / f"sheet-{letter}.xlsx"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert f"Sheet {letter}: sha256 `{digest}`" in text
        assert numbers["sha256"][letter] == digest
    assert "Haiku's accuracy is measured on the 218 adjudicated rows" in text
    assert not (ROOT / "benchmarks" / "ruler" / "results" / "pilot.md").exists()


def test_a_pair_that_disagrees_with_the_gold_fails_the_fixed_rule(pilot, tmp_path):
    _run(pilot, tmp_path, wrong_every=3)

    numbers = pilot.score(tmp_path / "p", 60, 60, tmp_path / "pilot.md")

    assert numbers["people"]["B"]["rate"] < pilot.PASS_RATE
    assert numbers["passed"] is False
    assert "**Result: FAIL.**" in (tmp_path / "pilot.md").read_text()


def test_a_first_attempt_with_a_typo_is_listed_then_counted_in_the_report(
    pilot, tmp_path
):
    _run(pilot, tmp_path, wrong_every=20)
    sheet_a = tmp_path / "p" / "for_labellers" / "sheet-A.xlsx"
    book = openpyxl.load_workbook(sheet_a)
    sheet = book["Labels"]
    header = [c.value for c in sheet[1]]
    column = header.index("label: label") + 1
    cell = next(
        sheet.cell(r, column)
        for r in range(2, 102)
        if sheet.cell(r, column).value != "?"
    )
    typed = cell.value
    cell.value = typed[:-1]
    book.save(sheet_a)

    with pytest.raises(ReviewSheetError) as raised:
        pilot.score(tmp_path / "p", 50, 80, tmp_path / "pilot.md")
    assert len(raised.value.problems) == 1
    assert not (tmp_path / "pilot.md").exists()

    cell.value = typed
    book.save(sheet_a)
    pilot.score(tmp_path / "p", 50, 80, tmp_path / "pilot.md")

    assert "found 1 problems: 1.0 per 100 rows" in (tmp_path / "pilot.md").read_text()


def test_several_answers_are_never_agreement_even_when_one_is_the_gold(pilot, tmp_path):
    pipeline, rows, adjudicated = pilot.replay()
    gold = {
        i: adjudicated[r["id"]] for i, r in enumerate(rows) if r["id"] in adjudicated
    }
    allowed = sorted({r["label"] for r in rows})
    review = pilot.export(tmp_path / "p")

    def hedge(row):
        truth = gold.get(row, rows[row]["label"])
        return f"{next(a for a in allowed if a != truth)} | {truth}"

    _fill(review.sheets["A"], hedge)
    _fill(review.sheets["B"], hedge)
    numbers = pilot.score(tmp_path / "p", 60, 60, tmp_path / "pilot.md")

    for person in numbers["people"].values():
        assert person["agree"] == 0
        assert person["several"] == person["covered"] > 0
        assert person["several_share"] == 1.0
    assert numbers["passed"] is False


def test_an_existing_report_is_not_replaced_without_force(pilot, tmp_path):
    _run(pilot, tmp_path, wrong_every=20)
    report = tmp_path / "pilot.md"
    pilot.score(tmp_path / "p", 50, 80, report)
    first = report.read_text()

    with pytest.raises(FileExistsError, match="--force"):
        pilot.score(tmp_path / "p", 10, 10, report)
    assert report.read_text() == first
    assert (
        pilot.main(
            [
                "score",
                "--folder",
                str(tmp_path / "p"),
                "--minutes-a",
                "10",
                "--minutes-b",
                "10",
                "--report",
                str(report),
            ]
        )
        == 2
    )
    assert report.read_text() == first

    pilot.score(tmp_path / "p", 10, 10, report, force=True)
    assert report.read_text() != first
