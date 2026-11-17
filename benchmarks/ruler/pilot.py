"""The human pilot for the review sheet: export the 100 rows, then score the labels.

Design section 8, step 3, and section 13.9. It makes no API call: it replays
Haiku's recorded intent answers through the real pipeline, as
``tests/e2e/test_certify_golden_e2e.py`` does, so it costs nothing in API spend.
The pass rule is fixed in ``PILOT.md`` before anyone labels.

    uv run python benchmarks/ruler/pilot.py export --out pilot-review
    # send each person their own file from pilot-review/for_labellers/, and time them
    uv run python benchmarks/ruler/pilot.py score --folder pilot-review \\
        --minutes-a 55 --minutes-b 70

``export`` draws 100 rows and gives both people all of them (``double=1.0``).
``score`` reads the returned sheets, certifies against them, and writes
``results/pilot.md``. Send nobody ``pilot-review/keep_private/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from ondine import PipelineBuilder  # noqa: E402
from ondine.ruler import ReviewSheetError  # noqa: E402
from ondine.ruler.sheet import FOR_LABELLERS, KEEP_PRIVATE, load_review  # noqa: E402
from tests.e2e.conformance_harness import token_for  # noqa: E402
from tests.e2e.test_contract_delivery_e2e import ScriptedClient  # noqa: E402

DATA = HERE.parent / "golden" / "data"
RESULTS = HERE / "results"
FIELD = "label"
SIZE = 100
#: The pass rule, fixed in PILOT.md before anyone labels: the adjudicated gold
#: matches people if each person agrees with it on at least this share of the
#: rows both cover.
PASS_RATE = 0.85
#: Written beside the manifest the first time a read fails, so the report can
#: say how many problems people's first attempt had.
FIRST_READ = "first-read-problems.txt"

GUIDE = """\
Each row is a message a bank customer sent to a support chat. Pick the one
intent, out of the 77 listed in the Values sheet, that the message is about.

The marker column is a row tag. Ignore it.

If two intents could both be right, type both, separated by |. If you cannot
tell, type ? alone. Do not guess to fill a cell."""


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def replay() -> tuple[Any, list[dict[str, Any]], dict[str, str]]:
    """The pipeline that replays Haiku's recorded answers on the intent dev split.

    Returns:
        The pipeline, the dev rows in run order, and the adjudicated gold by row id.
    """
    rows = _load(DATA / "intent" / "dev.jsonl")
    answers = {
        r["id"]: r["answer"] for r in _load(DATA / "labels" / "intent" / "haiku.jsonl")
    }
    adjudicated = {
        r["id"]: r["label"] for r in _load(DATA / "intent" / "dev-adjudicated.jsonl")
    }
    allowed = json.loads((DATA / "intent" / "labels.json").read_text())
    frame = pd.DataFrame(
        {
            "marker": [token_for(i) for i in range(len(rows))],
            "text": [r["text"] for r in rows],
        }
    )
    client = ScriptedClient({i: answers[r["id"]] for i, r in enumerate(rows)})
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(frame, input_columns=["marker", "text"], output_columns=[FIELD])
        .with_prompt("{marker} {text}")
        .with_custom_llm_client(client)
        .with_batch_size(1)
        .with_concurrency(8)
        .with_output_field(FIELD, allowed=allowed)
        .build()
    )
    return pipeline, rows, adjudicated


def export(out: Path, size: int = SIZE) -> Any:
    """Draw ``size`` rows and write the sheets: both people label every row."""
    pipeline, _, _ = replay()
    result = pipeline.execute()
    sample = pipeline.draw_sample(result, size=size)
    return pipeline.export_review(
        result, out, sample=sample, labellers=2, double=1.0, task=GUIDE
    )


def _answers(raw: Any) -> list[str]:
    return [str(a) for a in raw] if isinstance(raw, list) else [str(raw)]


def score(
    folder: Path,
    minutes_a: float,
    minutes_b: float,
    report: Path | None = None,
    sheets: dict[str, Path] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Read the returned sheets, certify, and write the pilot report.

    Returns:
        The numbers the report states, and whether the pass rule held.

    Raises:
        FileExistsError: The report exists and ``force`` is not set: a pilot
            report is not rewritten until the numbers look good.
        ReviewSheetError: The sheets have problems. Their number is kept in
            the folder, so the report can say how many the first read found.
    """
    sheets = sheets or {}
    report = report or RESULTS / "pilot.md"
    if report.exists() and not force:
        raise FileExistsError(f"{report} exists; pass --force to replace it")
    pipeline, rows, adjudicated = replay()
    try:
        labels = pipeline.read_review(folder, sheets=sheets)
    except ReviewSheetError as error:
        first = Path(folder) / KEEP_PRIVATE / FIRST_READ
        if not first.exists():
            first.write_text(str(len(error.problems)))
        raise
    first = Path(folder) / KEEP_PRIVATE / FIRST_READ
    problems = int(first.read_text()) if first.exists() else 0

    cert = pipeline.certify(labels.run, gold=labels)
    field = cert[FIELD]
    agreement = field.labellers
    sample = labels.sample.rows
    gold = {
        i: adjudicated[r["id"]] for i, r in enumerate(rows) if r["id"] in adjudicated
    }
    dropped = {i for i, r in enumerate(rows) if r["id"] not in adjudicated}

    first_answers = labels.reference.answers[FIELD]
    second_answers = labels.reference.labellers.second[FIELD]  # type: ignore[union-attr]
    people = {"A": first_answers, "B": second_answers}
    minutes = {"A": minutes_a, "B": minutes_b}
    per_person = {}
    for letter, answers in people.items():
        covered = [row for row in sample if row in answers and row in gold]
        # A row agrees only with exactly one answer, and it is the gold's.
        agree = [row for row in covered if _answers(answers[row]) == [gold[row]]]
        several = [row for row in covered if len(_answers(answers[row])) > 1]
        per_person[letter] = {
            "answered": sum(row in answers for row in sample),
            "covered": len(covered),
            "agree": len(agree),
            "rate": len(agree) / len(covered) if covered else 0.0,
            "several": len(several),
            "several_share": len(several) / len(covered) if covered else 0.0,
            "minutes": minutes[letter],
            "per_row": minutes[letter] / len(sample),
        }
    several = {
        row
        for answers in people.values()
        for row in sample
        if row in answers and len(_answers(answers[row])) > 1
    }
    saved = load_review(folder)
    unsure = {
        row
        for by_field in labels.states.values()
        for row, state in by_field[FIELD].items()
        if state == "unsure"
    }
    in_sample_dropped = dropped & set(sample)
    digests = {}
    for letter, plan in saved.manifest.sheets.items():
        path = sheets.get(letter, Path(folder) / FOR_LABELLERS / Path(plan.file).name)
        digests[letter] = hashlib.sha256(path.read_bytes()).hexdigest()

    census = pipeline.certify(labels.run, gold=gold)[FIELD].estimate
    haiku = census.accuracy
    low, high = field.estimate.low, field.estimate.high
    passed = all(p["rate"] >= PASS_RATE for p in per_person.values())
    numbers = {
        "passed": passed,
        "people": per_person,
        "agreement": agreement,
        "unsure_rows": len(unsure),
        "unsure_dropped": len(unsure & dropped),
        "several_rows": len(several),
        "several_dropped": len(several & dropped),
        "sampled_dropped": len(in_sample_dropped),
        "sampled_dropped_flagged": len(in_sample_dropped & (unsure | several)),
        "problems": problems,
        "problems_per_100": 100 * problems / len(sample),
        "haiku_accuracy": haiku,
        "adjudicated_rows": len(gold),
        "population": field.estimate.population,
        "sha256": digests,
        "range": (low, high),
        "contains": low is not None
        and high is not None
        and haiku is not None
        and low <= haiku <= high,
    }
    text = _report(numbers, len(sample), labels.sheet_id)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    print(text)
    return numbers


def _pct(share: float | None) -> str:
    return "n/a" if share is None else f"{share:.1%}"


def _report(n: dict[str, Any], rows: int, sheet: str) -> str:
    a = n["agreement"]
    lines = [
        f"# Human pilot ({date.today().isoformat()})",
        "",
        f"Two people labelled the same {rows} intent rows, blind, with sheet {sheet}. "
        "The pass rule was fixed before they started (`PILOT.md`).",
        "",
        f"**Pass rule:** each person agrees with the adjudicated gold on at least "
        f"{PASS_RATE:.0%} of the rows both cover. A row agrees only if the person "
        "gave exactly one answer and it equals the gold; several answers never "
        "count as agreement. **Result: "
        f"{'PASS' if n['passed'] else 'FAIL'}.**",
        "",
        "## Each person against the adjudicated gold",
        "",
        "| Person | Rows answered | Rows both cover | Agree (one answer, equal to the gold) | Rate | Several answers (rows both cover) | Minutes | Minutes per row |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for letter, p in n["people"].items():
        lines.append(
            f"| {letter} | {p['answered']} | {p['covered']} | {p['agree']} | "
            f"{_pct(p['rate'])} | {p['several']} ({_pct(p['several_share'])}) | {p['minutes']:g} | {p['per_row']:.2f} |"
        )
    kappa = "not defined" if a.kappa is None else f"{a.kappa:.2f}"
    lines += [
        "",
        "Rows with several answers stay in the rows both cover, and are not agreement.",
        "",
        "## Agreement between the two",
        "",
        f"- Rows both answered: {a.both_answered} of {a.doubled}.",
        f"- Answer sets that share a value: {a.agree} "
        f"({_pct(a.agree / a.both_answered) if a.both_answered else 'n/a'}).",
        f"- Cohen's kappa: {kappa}, on {a.kappa_rows} rows where each gave one answer.",
        f"- Rows split on the delivered value (unresolved): {a.split}.",
        "",
        "## Rows people could not label with one answer",
        "",
        f"- Marked `?` by either person: {n['unsure_rows']}; "
        f"{n['unsure_dropped']} of them are among the 13 rows the adjudicated gold drops.",
        f"- Given several answers by either person: {n['several_rows']}; "
        f"{n['several_dropped']} of them are among the dropped rows.",
        f"- Dropped rows in this sample: {n['sampled_dropped']}; "
        f"{n['sampled_dropped_flagged']} of them were marked `?` or given several answers.",
        "",
        "## Import problems",
        "",
        f"- The first read of the returned sheets found {n['problems']} problems: "
        f"{n['problems_per_100']:.1f} per 100 rows.",
        "",
        "## The range",
        "",
    ]
    low, high = n["range"]
    if low is None or high is None:
        lines.append("- The people's labels gave no range (a census).")
    else:
        lines.append(
            f"- The range from people's labels: {_pct(low)}–{_pct(high)}. Haiku's "
            f"accuracy against the adjudicated gold: {_pct(n['haiku_accuracy'])}. "
            f"The range {'contains' if n['contains'] else 'does not contain'} it."
        )
    lines.append(
        f"- Not the same rows: Haiku's accuracy is measured on the "
        f"{n['adjudicated_rows']} adjudicated rows, while the range describes all "
        f"{n['population']} delivered rows, estimated from the {rows}-row sample "
        "against people's labels. Read the comparison as a rough check, not a test."
    )
    lines += ["", "## The sheets scored", ""]
    lines += [
        f"- Sheet {letter}: sha256 `{digest}`" for letter, digest in n["sha256"].items()
    ]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser(
        "export", help="draw 100 rows, write the sheets"
    )
    export_parser.add_argument("--out", type=Path, required=True)
    export_parser.add_argument("--size", type=int, default=SIZE)
    score_parser = commands.add_parser(
        "score", help="read the sheets, write the report"
    )
    score_parser.add_argument("--folder", type=Path, required=True)
    score_parser.add_argument("--minutes-a", type=float, required=True)
    score_parser.add_argument("--minutes-b", type=float, required=True)
    score_parser.add_argument("--report", type=Path, default=None)
    score_parser.add_argument("--sheet-a", type=Path, default=None)
    score_parser.add_argument("--sheet-b", type=Path, default=None)
    score_parser.add_argument(
        "--force", action="store_true", help="replace an existing report"
    )
    args = parser.parse_args(argv)

    logging.disable(logging.CRITICAL)
    if args.command == "export":
        print(export(args.out, args.size).render())
        return 0
    sheets = {
        letter: path
        for letter, path in (("A", args.sheet_a), ("B", args.sheet_b))
        if path is not None
    }
    try:
        outcome = score(
            args.folder,
            args.minutes_a,
            args.minutes_b,
            args.report,
            sheets,
            args.force,
        )
    except (ReviewSheetError, FileExistsError) as error:
        print(error)
        return 2
    return 0 if outcome["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
