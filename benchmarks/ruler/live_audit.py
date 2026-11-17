"""Audit a live model run with a reference model, and check the audit against gold.

Runs a model on a golden dev split under an output contract, then measures
the delivered column two ways:

- **certificate**: against the dataset's labels, on every delivered row (the
  truth, for this split);
- **audit**: against a reference model, on Ondine's sample (here the whole
  split, a census, since the dev split is under 200 rows).

The gap between the two is what an audit cannot see: rows where the run and
the reference share a mistake, and rows where the reference is the one wrong.
The audit's wrong-row list is scored against the labels too: of the rows the
reference contradicts, how many are really wrong.

    uv run python benchmarks/ruler/live_audit.py --task intent \\
        --model openrouter/openai/gpt-oss-20b --reference luna

Costs a few cents; hard caps stop the run at $0.50 and the audit at --cap.
Writes ``results/live-audit-<task>-<model>-<reference>.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "golden"))
from label import MODEL_PARAMS, MODELS, load, prepare  # noqa: E402

from ondine import PipelineBuilder  # noqa: E402

DATA = HERE.parent / "golden" / "data"


def _model(name: str) -> tuple[str, dict]:
    if name in MODELS:
        return MODELS[name], MODEL_PARAMS.get(name, {"temperature": 0.0})
    return name, {"temperature": 0.0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", choices=["intent", "faithfulness"], default="intent")
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", default="luna")
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--cap", type=float, default=0.5)
    args = parser.parse_args(argv)

    frame, inputs, prompt = prepare(args.task, load(args.task))
    allowed = (
        json.loads((DATA / "intent" / "labels.json").read_text())
        if args.task == "intent"
        else ["consistent", "inconsistent"]
    )
    model, params = _model(args.model)
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(frame, input_columns=inputs, output_columns=["answer"])
        .with_prompt(prompt)
        .with_llm(model=model, **params)
        .with_output_field("answer", allowed=allowed)
        .with_contract_policy(re_ask=1, early_stop_fraction=None)
        .with_concurrency(6)
        .with_max_budget(args.budget)
        .build()
    )
    started = time.time()
    result = pipeline.execute()
    run_seconds = time.time() - started

    certificate = pipeline.certify(result, gold="label")
    reference, reference_params = _model(args.reference)
    estimate = pipeline.estimate_audit(result, model=reference, **reference_params)
    started = time.time()
    audit = pipeline.audit(
        result, model=reference, max_cost=args.cap, **reference_params
    )
    audit_seconds = time.time() - started

    # The same sample, certified against the labels: accuracy on the rows
    # the audit compared, so the two numbers differ only by the reference.
    sample = pipeline.draw_sample(result)
    labels = result.to_pandas()["label"]
    on_sample = pipeline.certify(
        result, gold={row: labels[row] for row in sample.rows}, sample=sample
    )
    flagged = {w.row for w in audit["answer"].wrong_rows}
    truly_wrong = {w.row for w in on_sample["answer"].wrong_rows}
    hits = len(flagged & truly_wrong)
    missed = len(truly_wrong - flagged)
    cert = on_sample["answer"].estimate
    census = certificate["answer"].estimate
    agree = audit["answer"].estimate

    text = "\n".join(
        [
            f"# Live audit: {args.task}, {model} audited by {reference}",
            "",
            f"Run: {len(frame)} rows in {run_seconds:.0f} s, "
            f"${float(result.costs.total_cost):.4f}. "
            f"Audit: {audit_seconds:.0f} s, ${float(audit.reference.cost):.4f} "
            f"(estimated at most ${float(estimate.cost):.4f}).",
            "",
            "| measure | value |",
            "|---|---:|",
            f"| accuracy on every row (census against the labels) | {census.accuracy:.1%} |",
            f"| accuracy on the sample (against the labels) | {cert.accuracy:.1%} ({cert.low:.1%}–{cert.high:.1%}) |",
            f"| agreement with {reference} on the same sample (audit) | {agree.accuracy:.1%} ({agree.low:.1%}–{agree.high:.1%}) |",
            f"| gap on the sample (agreement − accuracy) | {(agree.accuracy - cert.accuracy) * 100:+.1f} points |",
            f"| sampled rows the audit flags | {len(flagged)} |",
            f"| of which really wrong (labels) | {hits} |",
            f"| really wrong sampled rows it missed | {missed} of {len(truly_wrong)} |",
            f"| unresolved in the audit | {agree.unresolved} |",
            "",
            "## Estimate",
            "",
            "```",
            estimate.render(),
            "```",
            "",
            "## Audit",
            "",
            "```",
            audit.render(values=5, wrong=10),
            "```",
            "",
            "## Certificate",
            "",
            "```",
            certificate.render(values=5, wrong=10),
            "```",
            "",
        ]
    )
    slug = re.sub(
        r"[^a-z0-9.-]+", "-", f"{args.task}-{args.model}-{args.reference}".lower()
    )
    dest = HERE / "results" / f"live-audit-{slug}.md"
    dest.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
