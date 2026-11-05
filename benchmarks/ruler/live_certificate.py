"""Certify a live model run end to end: real API, real contract, real labels.

Runs a model on a golden dev split under an output contract, then certifies
the delivered column twice against the dataset's labels: as a census (every
delivered row), and from a sample of 100 rows Ondine draws (labels looked up
only for those rows, as a person would label them). Plumbing proof, not
statistics: one run proves nothing about coverage (see simulate.py).

    uv run python benchmarks/ruler/live_certificate.py --task intent --model haiku

Costs about $0.03 per task with Haiku; a hard budget cap stops it at $0.50.
Writes ``results/live-<task>-<model>.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "golden"))
from label import MODEL_PARAMS, MODELS, load, prepare  # noqa: E402

from ondine import PipelineBuilder  # noqa: E402

DATA = HERE.parent / "golden" / "data"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", choices=["intent", "faithfulness"], default="intent")
    parser.add_argument("--model", choices=sorted(MODELS), default="haiku")
    parser.add_argument("--budget", type=float, default=0.5)
    args = parser.parse_args(argv)

    frame, inputs, prompt = prepare(args.task, load(args.task))
    allowed = (
        json.loads((DATA / "intent" / "labels.json").read_text())
        if args.task == "intent"
        else ["consistent", "inconsistent"]
    )
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(frame, input_columns=inputs, output_columns=["answer"])
        .with_prompt(prompt)
        .with_llm(
            model=MODELS[args.model],
            **MODEL_PARAMS.get(args.model, {"temperature": 0.0}),
        )
        .with_output_field("answer", allowed=allowed)
        .with_concurrency(6)
        .with_max_budget(args.budget)
        .build()
    )
    started = time.time()
    result = pipeline.execute()
    seconds = time.time() - started

    census = pipeline.certify(result, gold="label")
    sample = pipeline.draw_sample(result, size=100)
    gold = result.to_pandas()["label"]
    sampled = pipeline.certify(
        result, gold={row: gold[row] for row in sample.rows}, sample=sample
    )

    counts = {status.value: n for status, n in result.report.counts.items() if n}
    text = "\n".join(
        [
            f"# Live certificate: {args.task}, {MODELS[args.model]}",
            "",
            f"{len(frame)} rows in {seconds:.0f} s, cost ${float(result.costs.total_cost):.4f}; "
            f"statuses {counts}.",
            "",
            "## Census (labels for every delivered row)",
            "",
            "```",
            census.render(values=5, wrong=10),
            "```",
            "",
            "## Sample of 100 drawn by Ondine",
            "",
            "```",
            sampled.render(values=5, wrong=10),
            "```",
            "",
        ]
    )
    out = HERE / "results" / f"live-{args.task}-{args.model}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out.relative_to(HERE.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
