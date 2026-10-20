"""Label a golden-set split with one model, through Ondine.

Used for the adjudication step in README.md: run two strong models from
different families over the dev split, then hand-check the rows where
both disagree with the gold label. Doing it through Ondine means the run
also exercises the product and reports real cost.

    uv run python benchmarks/golden/label.py --task intent --model opus --limit 3
    uv run python benchmarks/golden/label.py --task intent --model opus

Output: ``data/labels/<task>/<model>.jsonl`` with one row per input row:
the row id, the model's raw answer, and the run's total cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ondine import PipelineBuilder

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

#: Short names for the models the benchmark uses. Prices come from LiteLLM.
MODELS = {
    "opus": "anthropic/claude-opus-5-5",
    "sol": "openai/gpt-6.1-sol",
    "sonnet": "anthropic/claude-sonnet-5-5",
    "haiku": "anthropic/claude-haiku-5-5",
    "luna": "openai/gpt-6-luna",
}

INTENT_PROMPT = """You are classifying customer-service messages from a bank's \
customers into exactly one intent.

Intents:
{labels}

Message: {{text}}

Reply with only the intent name, exactly as written above."""

FAITHFULNESS_PROMPT = """Below are a document and a summary of it. Decide whether \
the summary is factually consistent with the document. A summary is \
inconsistent if any statement in it is contradicted by the document, or is \
not supported by it.

Document:
{doc}

Summary:
{summary}

Reply with only one word: consistent or inconsistent."""

ATTRIBUTES_PROMPT = """Extract product attributes from this offer.

Category: {category}
Title: {title}
Description: {description}

Attributes to extract: {attributes}

Reply with only a JSON object whose keys are exactly those attributes. Use the \
value as stated in the offer, normalized (units written in a standard form, \
names spelled out). Use "n/a" for any attribute the offer does not state."""


def load(task: str) -> pd.DataFrame:
    path = DATA / task / "dev.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing; run benchmarks/golden/build.py first")
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    return pd.DataFrame(rows)


def prepare(task: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    """Return the frame, its input columns and the prompt for `task`."""
    if task == "intent":
        labels = json.loads((DATA / task / "labels.json").read_text())
        return frame, ["text"], INTENT_PROMPT.format(labels="\n".join(labels))
    if task == "faithfulness":
        return frame, ["doc", "summary"], FAITHFULNESS_PROMPT
    if task == "attributes":
        frame = frame.copy()
        frame["attributes"] = frame["target"].map(lambda t: ", ".join(sorted(t)))
        cols = ["category", "title", "description", "attributes"]
        return frame, cols, ATTRIBUTES_PROMPT
    raise SystemExit(f"unknown task {task}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--task", required=True, choices=["intent", "attributes", "faithfulness"]
    )
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--limit", type=int, help="first N rows only (smoke test)")
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="only the rows adjudicate.py listed (the tiebreak run)",
    )
    parser.add_argument(
        "--budget", type=float, default=2.0, help="USD ceiling for this run"
    )
    args = parser.parse_args(argv)

    frame = load(args.task)
    if args.candidates:
        listed = DATA / "adjudication" / f"{args.task}.jsonl"
        ids = {json.loads(line)["id"] for line in listed.open(encoding="utf-8")}
        frame = frame[frame["id"].isin(ids)].reset_index(drop=True)
    if args.limit:
        frame = frame.head(args.limit)
    frame, inputs, prompt = prepare(args.task, frame)

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(frame, input_columns=inputs, output_columns=["answer"])
        .with_prompt(prompt)
        .with_llm(model=MODELS[args.model])
        .with_concurrency(6)
        .with_max_budget(args.budget)
        .with_error_policy("skip")
        .build()
    )
    result = pipeline.execute()
    out = result.to_pandas()

    suffix = "-candidates" if args.candidates else ""
    dest = DATA / "labels" / args.task / f"{args.model}{suffix}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cost = float(result.costs.total_cost)
    with dest.open("w", encoding="utf-8") as f:
        for rid, answer in zip(frame["id"], out["answer"], strict=True):
            record = {"id": rid, "answer": None if pd.isna(answer) else answer}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    lost = result.lost_row_indices
    print(
        f"{args.task}/{args.model}: {len(frame)} rows, {len(lost)} lost, "
        f"cost ${cost:.4f}, tokens {result.costs.total_tokens}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
