"""Certified columns on real data: the golden sets and real model answers.

Each run replays a model's real recorded answers (``benchmarks/golden``)
through the real pipeline and a real contract, then certifies against the
dataset's labels. Free: no model is called. Skipped when the golden data is
absent (it is not committed; ``benchmarks/golden/build.py`` and
``label.py`` make it).

Design: ``.claude/designs/certified-columns.md`` section 11.4, scenarios 1-3.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

from ondine import PipelineBuilder
from tests.e2e.conformance_harness import token_for
from tests.e2e.test_contract_delivery_e2e import ScriptedClient

GOLDEN = Path(__file__).resolve().parents[2] / "benchmarks" / "golden"
DATA = GOLDEN / "data"

pytestmark = pytest.mark.skipif(
    not (DATA / "labels" / "intent" / "haiku.jsonl").exists(),
    reason="golden data not built (benchmarks/golden/build.py, label.py)",
)


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def _allowed(task: str) -> list[str]:
    if task == "intent":
        return json.loads((DATA / "intent" / "labels.json").read_text())
    return ["consistent", "inconsistent"]


def _replay(task: str, model: str):
    """The task's dev rows, run with the model's recorded answers."""
    rows = _load(DATA / task / "dev.jsonl")
    answers = {
        r["id"]: r["answer"] for r in _load(DATA / "labels" / task / f"{model}.jsonl")
    }
    adjudicated = {
        r["id"]: r["label"] for r in _load(DATA / task / "dev-adjudicated.jsonl")
    }
    text = [r["text"] if task == "intent" else r["summary"] for r in rows]
    frame = pd.DataFrame(
        {
            "marker": [token_for(i) for i in range(len(rows))],
            "text": text,
            "gold": [r["label"] for r in rows],
            "adjudicated": [adjudicated.get(r["id"]) for r in rows],
        }
    )
    client = ScriptedClient({i: answers[r["id"]] for i, r in enumerate(rows)})
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            frame, input_columns=["marker", "text"], output_columns=["label"]
        )
        .with_prompt("{marker} {text}")
        .with_custom_llm_client(client)
        .with_batch_size(1)
        .with_concurrency(8)
        .with_output_field("label", allowed=_allowed(task))
        .build()
    )
    return pipeline, rows, answers


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _independent_right(rows, answers) -> int:
    """Right answers by the golden scorer's own rule, not Ondine's."""
    sys.path.insert(0, str(GOLDEN))
    from adjudicate import final_answer, norm

    return sum(norm(final_answer(answers[r["id"]])) == norm(r["label"]) for r in rows)


def test_a_census_of_real_answers_matches_the_golden_scorer_exactly():
    pipeline, rows, answers = _replay("intent", "haiku")
    result = pipeline.execute()

    est = pipeline.certify(result, gold="gold")["label"].estimate

    assert est.exact
    assert (est.population, est.right) == (len(rows), _independent_right(rows, answers))


def test_labels_missing_on_some_rows_give_a_check_not_a_certificate():
    # The adjudicated gold drops the rows with no defensible label.
    pipeline, rows, _ = _replay("intent", "haiku")
    result = pipeline.execute()

    cert = pipeline.certify(result, gold="adjudicated")

    assert cert.claim == "check"
    labelled = int(result.to_pandas()["adjudicated"].notna().sum())
    assert labelled < len(rows)
    assert cert["label"].estimate.compared == labelled


def test_ondine_drawn_samples_of_real_answers_hold_the_census_accuracy():
    # Each run has its own id, so its own sample: 200 runs, 200 samples.
    pipeline, rows, answers = _replay("intent", "haiku")
    truth = _independent_right(rows, answers) / len(rows)
    held = 0
    for _ in range(200):
        result = pipeline.execute()
        sample = pipeline.draw_sample(result, size=100)
        gold = result.to_pandas()["gold"]
        est = pipeline.certify(
            result, gold={row: gold[row] for row in sample.rows}, sample=sample
        )["label"].estimate
        held += est.low <= truth <= est.high
    # At a true coverage of 95%, fewer than 180 of 200 has odds of about 1 in 5,000.
    assert held >= 180, f"range held the census accuracy in {held} of 200 samples"


def test_real_repaired_and_rejected_answers_show_in_the_certificate():
    pipeline, rows, _ = _replay("faithfulness", "haiku")
    result = pipeline.execute()

    label = pipeline.certify(result, gold="gold")["label"]

    statuses = {g.name: g.estimate.compared for g in label.by_status}
    assert statuses["repaired"] == 2
    assert label.not_delivered == {"rejected": 1}
    assert label.estimate.population == len(rows) - 1
