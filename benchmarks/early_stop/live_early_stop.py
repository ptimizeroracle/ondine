"""Early stop on a live model: a broken prompt stops, a working one does not.

Two runs over the same 200 faithfulness rows, each with the default policy
(early stop at more than 50% of the first 50 rows) and ``re_ask=1``:

- **loose**: "Is this summary consistent with the document?". The model
  answers "Yes"/"No" with an explanation; without early stop, 175 of 200 first
  answers failed (``benchmarks/reask/results/live-faithfulness-luna.md``).
  Expected: ``EarlyStopError`` after the first 50 rows.
- **strict**: the golden prompt, which asks for the value alone. Expected: no
  stop, every row answered.

    uv run python benchmarks/early_stop/live_early_stop.py --model luna

Costs a few cents; a hard budget cap stops each run at $0.50.
Writes ``results/live-faithfulness-<model>.md``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLDEN = HERE.parent / "golden"
sys.path.insert(0, str(GOLDEN))
from label import FAITHFULNESS_PROMPT, MODEL_PARAMS, MODELS, load  # noqa: E402

from ondine import EarlyStopError, PipelineBuilder  # noqa: E402
from ondine.harness.status import RowStatus  # noqa: E402

PROMPTS = {
    "loose": "Is this summary consistent with the document?\n\nDocument:\n{doc}"
    "\n\nSummary:\n{summary}",
    "strict": FAITHFULNESS_PROMPT,
}


def run(name: str, model: str, budget: float) -> dict[str, object]:
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            load("faithfulness"),
            input_columns=["doc", "summary"],
            output_columns=["answer"],
        )
        .with_prompt(PROMPTS[name])
        .with_llm(model=MODELS[model], **MODEL_PARAMS.get(model, {"temperature": 0.0}))
        .with_output_field("answer", allowed=["consistent", "inconsistent"])
        .with_contract_policy(re_ask=1)
        .with_concurrency(6)
        .with_max_budget(budget)
        .build()
    )
    started = time.time()
    error: EarlyStopError | None = None
    try:
        pipeline.execute()
    except EarlyStopError as stopped:
        error = stopped
    seconds = time.time() - started
    report = pipeline.run_report
    rows = pipeline.row_ledger.rows
    asked = sum(1 for o in rows if o.attempts)
    calls = sum(len(o.attempts) for o in rows)
    return {
        "name": name,
        "seconds": seconds,
        "stopped": error is not None,
        "message": str(error) if error else "",
        "rows": len(rows),
        "asked": asked,
        "calls": calls,
        "cost": float(report.cost_total),
        "counts": {s.value: n for s, n in report.counts.items() if n},
        "stop_reason": None if report.stop_reason is None else report.stop_reason.value,
        "failed_never_asked": sum(
            1 for o in rows if o.status is RowStatus.FAILED and not o.attempts
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=sorted(MODELS), default="luna")
    parser.add_argument("--budget", type=float, default=0.5)
    args = parser.parse_args(argv)

    results = [run(name, args.model, args.budget) for name in ("loose", "strict")]
    lines = [
        f"# Early stop, live: faithfulness, {MODELS[args.model]}",
        "",
        "Default policy (stop above 50% of the first 50 rows), `re_ask=1`.",
        "",
        "| prompt | stopped | rows asked | calls | cost | time | statuses |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {'yes' if r['stopped'] else 'no'} | "
            f"{r['asked']} of {r['rows']} | {r['calls']} | ${r['cost']:.4f} | "
            f"{r['seconds']:.0f} s | {r['counts']} |"
        )
    for r in results:
        if r["stopped"]:
            lines += [
                "",
                f"Error raised by the {r['name']} run "
                f"({r['failed_never_asked']} rows never asked, "
                f"stop reason: {r['stop_reason']}):",
                "",
                "```",
                str(r["message"]),
                "```",
            ]
    text = "\n".join(lines) + "\n"
    dest = HERE / "results" / f"live-faithfulness-{args.model}.md"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
