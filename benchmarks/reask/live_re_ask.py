"""Re-ask on a live model with a loosely worded prompt: what it recovers.

The prompt does not ask for the value alone, so the model explains its
choice, and the contract rejects explanations that name two labels or
qualify one. One run with ``re_ask=1`` shows before and after on the same
answers: the rows that failed on their first answer, what the re-ask did
with each, and how many recovered values the dataset's labels confirm.

    uv run python benchmarks/reask/live_re_ask.py --task intent --model luna

Costs a few cents per task; a hard budget cap stops a run at $0.50.
Writes ``results/live-<task>-<model>.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLDEN = HERE.parent / "golden"
sys.path.insert(0, str(GOLDEN))
from label import MODEL_PARAMS, MODELS, load  # noqa: E402

from ondine import PipelineBuilder  # noqa: E402

DATA = GOLDEN / "data"

LOOSE = {
    "intent": (
        ["text"],
        "A customer of a bank wrote this message. Which of these intents fits "
        "it best?\n\nIntents:\n{labels}\n\nMessage: {{text}}",
    ),
    "faithfulness": (
        ["doc", "summary"],
        "Is this summary consistent with the document?\n\nDocument:\n{doc}\n\n"
        "Summary:\n{summary}",
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", choices=sorted(LOOSE), default="intent")
    parser.add_argument("--model", choices=sorted(MODELS), default="luna")
    parser.add_argument("--budget", type=float, default=0.5)
    args = parser.parse_args(argv)

    frame = load(args.task)
    inputs, prompt = LOOSE[args.task]
    if args.task == "intent":
        allowed = json.loads((DATA / "intent" / "labels.json").read_text())
        prompt = prompt.format(labels="\n".join(allowed))
    else:
        allowed = ["consistent", "inconsistent"]
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(frame, input_columns=inputs, output_columns=["answer"])
        .with_prompt(prompt)
        .with_llm(
            model=MODELS[args.model],
            **MODEL_PARAMS.get(args.model, {"temperature": 0.0}),
        )
        .with_output_field("answer", allowed=allowed)
        .with_contract_policy(re_ask=1)
        .with_concurrency(6)
        .with_max_budget(args.budget)
        .build()
    )
    started = time.time()
    result = pipeline.execute()
    seconds = time.time() - started

    out = result.to_pandas()
    gold = out["label"]
    outcomes = sorted(result.row_ledger.rows, key=lambda o: o.row)
    first_failed = [o for o in outcomes if o.verdicts and not o.verdicts[0][1].passed]
    re_asked = [o for o in first_failed if len(o.attempts) > 1]
    fixed = [o for o in re_asked if out["repair_method"][o.row] == "re-ask"]
    fixed_right = [o for o in fixed if out["answer"][o.row] == gold[o.row]]
    first_cost = sum(o.attempts[0].cost for o in outcomes if o.attempts)
    re_ask_cost = sum(a.cost for o in outcomes for a in o.attempts[1:])
    cert = pipeline.certify(result, gold="label")

    counts = {status.value: n for status, n in result.report.counts.items() if n}
    lines = [
        f"# Re-ask, live: {args.task}, {MODELS[args.model]}, loosely worded prompt",
        "",
        f"{len(frame)} rows in {seconds:.0f} s; statuses {counts}.",
        "",
        "| | rows |",
        "|---|---:|",
        f"| first answer failed a check | {len(first_failed)} |",
        f"| asked again | {len(re_asked)} |",
        f"| fixed by the re-ask | {len(fixed)} |",
        f"| fixed and right against the labels | {len(fixed_right)} |",
        f"| still failing after the re-ask | {len(re_asked) - len(fixed)} |",
        "",
        f"Cost: first answers ${float(first_cost):.4f}, re-asks ${float(re_ask_cost):.4f}.",
        "",
        "```",
        cert.render(values=5, wrong=10),
        "```",
        "",
    ]
    text = "\n".join(lines)
    dest = HERE / "results" / f"live-{args.task}-{args.model}.md"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
