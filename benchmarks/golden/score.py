"""Score every labelled model on every task: accuracy and cost per successful row.

Reads ``data/labels/<task>/<model>.jsonl`` and the dev split, and scores
against two golds: the dataset's original labels, and the adjudicated
labels (``dev-adjudicated.jsonl``, written by build.py from
``adjudications.jsonl``; dropped rows are left out).

    uv run python benchmarks/golden/score.py

Metrics, per task:

* ``intent``: accuracy (exact match on the last answer line).
* ``faithfulness``: balanced accuracy, because 37% of rows are consistent.
* ``attributes``: attribute-level accuracy, strict (normalised string
  equality) and lenient (same leading number, containment), plus the
  share of rows with every attribute right.

Cost per successful row divides the run's total cost by the number of rows
scored right; a failed row stays in the numerator. Costs come from
``data/labels/<task>/<model>.cost.json`` when label.py wrote one, else
they are reported as unknown.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE))
from adjudicate import (  # noqa: E402
    attribute_errors,
    final_answer,
    norm,
    parse_json_object,
)

TASKS = ("intent", "faithfulness", "attributes")
MODELS = ("haiku", "luna", "sonnet", "sol", "opus")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def strict_attribute_errors(
    target: dict[str, list[str]], answer: str | None
) -> list[str]:
    parsed = parse_json_object(answer)
    if parsed is None:
        return sorted(target)
    return [
        a
        for a, accepted in target.items()
        if not any(norm(parsed.get(a, "n/a")) == norm(v) for v in accepted)
    ]


def score(task: str, gold: list[dict], answers: dict[str, str | None]) -> dict:
    """Return the task's metrics for `answers` against `gold` rows."""
    if task == "attributes":
        total = strict_ok = lenient_ok = 0
        rows_all_ok = 0
        for g in gold:
            a = answers.get(g["id"])
            n = len(g["target"])
            total += n
            s = len(strict_attribute_errors(g["target"], a))
            le = len(attribute_errors(g["target"], a))
            strict_ok += n - s
            lenient_ok += n - le
            rows_all_ok += le == 0
        return {
            "rows": len(gold),
            "strict_acc": strict_ok / total if total else 0.0,
            "lenient_acc": lenient_ok / total if total else 0.0,
            "rows_all_ok": rows_all_ok,
            "successes": rows_all_ok,
        }
    right = 0
    per_class: dict[str, list[int]] = {}
    for g in gold:
        ok = norm(final_answer(answers.get(g["id"]))) == norm(g["label"])
        right += ok
        per_class.setdefault(g["label"], []).append(ok)
    out = {
        "rows": len(gold),
        "accuracy": right / len(gold) if gold else 0.0,
        "successes": right,
    }
    if task == "faithfulness":
        recalls = [sum(v) / len(v) for v in per_class.values() if v]
        out["balanced_acc"] = sum(recalls) / len(recalls) if recalls else 0.0
    return out


def main() -> int:
    report: dict[str, dict] = {}
    for task in TASKS:
        original = load_jsonl(DATA / task / "dev.jsonl")
        adjudicated = load_jsonl(DATA / task / "dev-adjudicated.jsonl") or original
        report[task] = {}
        for model in MODELS:
            labels = load_jsonl(DATA / "labels" / task / f"{model}.jsonl")
            if len(labels) < len(original):
                continue  # not run, or only a smoke test
            answers = {r["id"]: r["answer"] for r in labels}
            cost_file = DATA / "labels" / task / f"{model}.cost.json"
            cost = (
                json.loads(cost_file.read_text())["cost"]
                if cost_file.exists()
                else None
            )
            entry = {
                "original": score(task, original, answers),
                "adjudicated": score(task, adjudicated, answers),
                "cost_usd": cost,
            }
            if cost is not None:
                s = entry["adjudicated"]["successes"]
                entry["cost_per_successful_row"] = cost / s if s else None
            report[task][model] = entry

    out = DATA / "baseline-report.json"
    out.write_text(json.dumps(report, indent=2))
    for task, models in report.items():
        print(f"\n[{task}]")
        key = {
            "intent": "accuracy",
            "faithfulness": "balanced_acc",
            "attributes": "lenient_acc",
        }[task]
        for model, e in models.items():
            o, a = e["original"][key], e["adjudicated"][key]
            c = e["cost_usd"]
            cps = e.get("cost_per_successful_row")
            print(
                f"  {model:7s} {key}: original {o:.3f}  adjudicated {a:.3f}"
                + (f"  cost ${c:.3f}" if c is not None else "  cost ?")
                + (f"  per successful row ${cps * 1000:.2f}/1k" if cps else "")
            )
    print(f"\nreport -> {out.relative_to(HERE.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
