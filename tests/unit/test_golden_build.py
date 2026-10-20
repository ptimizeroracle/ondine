# ruff: noqa: S311 - seeded random.Random is what the builder takes; nothing here is cryptographic
"""The parts of benchmarks/golden/build.py that encode scoring rules.

Downloads are not exercised here; these tests cover the two pieces that
would silently corrupt a golden set if they regressed: how WDC-PAVE's
scored targets flatten to accepted values, and how SummEdits rows are
balanced without letting a document straddle splits.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

_BUILD = Path(__file__).resolve().parents[2] / "benchmarks" / "golden" / "build.py"
_spec = importlib.util.spec_from_file_location("golden_build", _BUILD)
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)

_ADJ = Path(__file__).resolve().parents[2] / "benchmarks" / "golden" / "adjudicate.py"
_adj_spec = importlib.util.spec_from_file_location("golden_adjudicate", _ADJ)
adjudicate = importlib.util.module_from_spec(_adj_spec)
_adj_spec.loader.exec_module(adjudicate)


class TestAcceptedValues:
    def test_keeps_every_positively_scored_spelling(self):
        # The source accepts both spellings; a prediction matching either
        # must count, so both must survive flattening.
        target = {
            "Part Number": {"ST9300503SS": {"score": 1}, "ST-9300503-SS": {"score": 1}}
        }

        assert build.accepted_values(target) == {
            "Part Number": ["ST9300503SS", "ST-9300503-SS"]
        }

    def test_absent_attribute_is_a_single_na(self):
        assert build.accepted_values({"Generation": {"n/a": 1}}) == {
            "Generation": ["n/a"]
        }

    def test_zero_scored_values_do_not_count(self):
        # A value an annotator rejected must not become an accepted answer.
        target = {"Color": {"Black": {"score": 1}, "Dark": {"score": 0}}}

        assert build.accepted_values(target) == {"Color": ["Black"]}


class TestBalanced:
    @staticmethod
    def rows_for(doc: str, consistent: int, inconsistent: int) -> list[dict]:
        labels = ["consistent"] * consistent + ["inconsistent"] * inconsistent
        return [
            {"id": f"{doc}-{i}", "doc": doc, "label": lab}
            for i, lab in enumerate(labels)
        ]

    def test_takes_half_of_each_label(self):
        by_doc = {"d1": self.rows_for("d1", 10, 10), "d2": self.rows_for("d2", 10, 10)}

        out = build._balanced(by_doc, ["d1", "d2"], 10, random.Random(0))

        assert sum(r["label"] == "consistent" for r in out) == 5
        assert sum(r["label"] == "inconsistent" for r in out) == 5

    def test_only_draws_from_the_named_documents(self):
        by_doc = {"d1": self.rows_for("d1", 10, 10), "d2": self.rows_for("d2", 10, 10)}

        out = build._balanced(by_doc, ["d1"], 10, random.Random(0))

        assert {r["doc"] for r in out} == {"d1"}

    def test_short_label_supply_takes_what_exists_not_a_substitute(self):
        # Two consistent rows exist; asking for five must yield two, never
        # pad with inconsistent rows relabelled or over-drawn.
        by_doc = {"d1": self.rows_for("d1", 2, 20)}

        out = build._balanced(by_doc, ["d1"], 10, random.Random(0))

        assert sum(r["label"] == "consistent" for r in out) == 2
        assert sum(r["label"] == "inconsistent" for r in out) == 5


class TestStratifiedTake:
    def test_removes_taken_rows_from_the_pool(self):
        # A later holdout draw from the same pool must never re-draw a dev row.
        groups = {"a": [{"id": i} for i in range(5)]}

        first = build.stratified_take(groups, 2, random.Random(0))
        second = build.stratified_take(groups, 2, random.Random(1))

        assert {r["id"] for r in first}.isdisjoint({r["id"] for r in second})
        assert len(groups["a"]) == 1


@pytest.mark.parametrize(
    "name", ["banking77_test", "wdcpave_extraction", "summedits_news"]
)
def test_every_source_is_pinned_to_a_hash(name):
    assert len(build.SOURCES[name]["sha256"]) == 16


class TestFinalAnswer:
    def test_takes_the_last_line_of_a_reply_that_reasons_first(self):
        reply = "The message is about a PIN, so...\n\ncard_arrival"

        assert adjudicate.final_answer(reply) == "card_arrival"

    def test_strips_markdown_decoration_from_the_label(self):
        assert adjudicate.final_answer("**inconsistent**") == "inconsistent"

    def test_empty_reply_is_no_answer(self):
        assert adjudicate.final_answer(None) is None


class TestSameValue:
    @pytest.mark.parametrize(
        ("got", "accepted"),
        [
            ("3 in", "3"),  # units ignored
            ("8.5 in", "8 1/2"),  # fraction read as a number
            ("15000 RPM", "15K"),  # K suffix
            ("Rack Server", "Rack"),  # containment
        ],
    )
    def test_equivalent_spellings_match(self, got, accepted):
        assert adjudicate.same_value(got, accepted)

    @pytest.mark.parametrize(
        ("got", "accepted"),
        [
            ("4 in", "3"),
            ("Hard Disk Drive", "Tape Drive"),
            ("10 in", "n/a"),  # a stated value is not an absent one
        ],
    )
    def test_different_values_do_not_match(self, got, accepted):
        assert not adjudicate.same_value(got, accepted)
