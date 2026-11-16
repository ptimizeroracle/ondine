"""Escalation on live models: a weak model, with a strong one for its failures.

Three runs over a golden dev split, each scored against the dataset's labels:

- **weak**: a small model, with ``--re-ask`` rounds (default 1). Rows it cannot fix stay rejected.
- **weak + escalation**: the same, then the strong model for the rows still
  failing after re-ask.
- **strong**: the strong model on every row, same policy: the cost
  escalation avoids.

Escalation only sees answers that fail a check. With the strict prompt a
small model answers with an allowed value, right or wrong, so nothing is
escalated; ``--prompt loose`` (the re-ask benchmark's wording) makes it
explain itself, which is what the contract rejects.

Early stop is off in all three: a weak model may fail more than half of its
first 50 rows, and this benchmark measures what happens to every row.

    uv run python benchmarks/escalation/live_escalation.py --task faithfulness \\
        --prompt loose --weak openrouter/openai/gpt-oss-20b --strong luna

Costs a few cents; a hard cap stops each run at $0.50.
Writes ``results/live-<task>-<prompt>-<weak>-<strong>.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLDEN = HERE.parent / "golden"
sys.path.insert(0, str(GOLDEN))
from label import MODEL_PARAMS, MODELS, load, prepare  # noqa: E402

sys.path.insert(0, str(HERE.parent / "reask"))
from live_re_ask import LOOSE  # noqa: E402

from ondine import PipelineBuilder  # noqa: E402

DATA = GOLDEN / "data"


def _model(name: str) -> tuple[str, dict]:
    if name in MODELS:
        return MODELS[name], MODEL_PARAMS.get(name, {"temperature": 0.0})
    return name, {"temperature": 0.0}


def _task(task: str, style: str) -> tuple:
    frame, inputs, prompt = prepare(task, load(task))
    if task == "intent":
        labels = json.loads((DATA / "intent" / "labels.json").read_text())
    else:
        labels = ["consistent", "inconsistent"]
    if style == "loose":
        inputs, prompt = LOOSE[task]
        if task == "intent":
            prompt = prompt.format(labels="\n".join(labels))
    return frame, inputs, prompt, labels


def run(name: str, args: argparse.Namespace, escalate: bool) -> dict:
    frame, inputs, prompt, labels = _task(args.task, args.prompt)
    weak, strong, budget = args.weak, args.strong, args.budget
    model, params = _model(strong if name == "strong" else weak)
    builder = (
        PipelineBuilder.create()
        .from_dataframe(frame, input_columns=inputs, output_columns=["answer"])
        .with_prompt(prompt)
        .with_llm(model=model, **params)
        .with_output_field("answer", allowed=labels)
        .with_contract_policy(re_ask=args.re_ask, early_stop_fraction=None)
        .with_concurrency(4)
        .with_max_budget(budget)
    )
    if escalate:
        strong_model, strong_params = _model(strong)
        builder = builder.with_escalation(model=strong_model, **strong_params)
    pipeline = builder.build()
    started = time.time()
    result = pipeline.execute()
    seconds = time.time() - started
    out = result.to_pandas()
    gold = frame["label"]
    delivered = out["answer"].notna()
    right = (out["answer"] == gold) & delivered
    escalated = out["repair_method"] == "escalate"
    return {
        "name": name,
        "rows": len(out),
        "seconds": seconds,
        "cost": float(result.costs.total_cost),
        "statuses": dict(Counter(out["status"])),
        "repairs": dict(Counter(out["repair_method"])),
        "delivered": int(delivered.sum()),
        "right": int(right.sum()),
        "escalated": int(escalated.sum()),
        "escalated_right": int((right & escalated).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", choices=sorted(LOOSE), default="intent")
    parser.add_argument("--prompt", choices=["strict", "loose"], default="strict")
    parser.add_argument(
        "--re-ask",
        type=int,
        default=1,
        help="re_ask for all three runs; 0 escalates straight after the run",
    )
    parser.add_argument("--weak", required=True)
    parser.add_argument("--strong", default="luna")
    parser.add_argument("--budget", type=float, default=0.5)
    args = parser.parse_args(argv)

    results = [
        run("weak", args, False),
        run("weak + escalation", args, True),
        run("strong", args, False),
    ]
    strong_name = _model(args.strong)[0]
    lines = [
        f"# Escalation, live: {args.task} ({args.prompt} prompt), "
        f"{args.weak} -> {strong_name}",
        "",
        f"{results[0]['rows']} rows; `re_ask={args.re_ask}` in all three runs; "
        "early stop off.",
        "Right: delivered value equals the dataset's label (rejected rows count",
        "as wrong).",
        "",
        "| run | delivered | right | accuracy | escalated (right) | cost | time |"
        " statuses | repair methods |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['delivered']} | {r['right']} | "
            f"{r['right'] / r['rows']:.1%} | {r['escalated']} ({r['escalated_right']}) "
            f"| ${r['cost']:.4f} | {r['seconds']:.0f} s | {r['statuses']} | "
            f"{r['repairs']} |"
        )
    text = "\n".join(lines) + "\n"
    slug = re.sub(
        r"[^a-z0-9.-]+",
        "-",
        f"{args.task}-{args.prompt}-reask{args.re_ask}-{args.weak}-{args.strong}".lower(),
    )
    dest = HERE / "results" / f"live-{slug}.md"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
