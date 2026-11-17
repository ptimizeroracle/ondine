"""Label a golden-set split with one model, through Ondine.

Used for the adjudication step in README.md: run two strong models from
different families over the dev split, then hand-check the rows where
both disagree with the gold label. Doing it through Ondine means the run
also exercises the product and reports real cost.

    uv run python benchmarks/golden/label.py --task intent --model opus --limit 3
    uv run python benchmarks/golden/label.py --task intent --model opus

Output: ``data/labels/<task>/<model>.jsonl`` with one row per input row:
the row id, the model's raw answer, and the run's total cost.

Extra Haiku samples (Planner Addendum 1, ``benchmarks/planner/DESIGN-addendum-1.md``)
are drawn with ``--sample N``. They go to ``haiku-s<N>.jsonl`` (never to
``haiku.jsonl``), refuse to overwrite a file, refuse a response cache, and
stop when the cost files already on disk plus this run's budget would pass
``--total-budget``:

    uv run python benchmarks/golden/label.py --task intent --model haiku --sample 1 --limit 3

``--sample`` runs go through OpenRouter unless ``--route direct`` is given:
the Anthropic account the original run used has no credit, and the
pre-registration names the OpenRouter route.
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
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

#: Per-model call settings. Haiku 5.5 rejects any temperature but 1 and
#: thinks by default (billed as output), so it runs at the lowest effort.
#: Everything else runs deterministic.
MODEL_PARAMS = {
    "haiku": {"temperature": 1.0, "reasoning_effort": "low"},
}

#: The OpenRouter route for Haiku (Addendum 1). LiteLLM lists no price for
#: this route, so the run would see $0 and its budget cap would never bind.
#: These explicit prices (LiteLLM's ``claude-haiku-5-5`` entry, the original
#: run's) are Ondine's fallback when the provider reports no cost.
OPENROUTER_MODEL = "openrouter/anthropic/claude-haiku-5-5"
OPENROUTER_PRICES = {
    "input_cost_per_1k_tokens": Decimal("0.0001"),
    "output_cost_per_1k_tokens": Decimal("0.0005"),
}

#: Addendum 1 limits: the per-run budget of a ``--sample`` run and the cap over
#: every ``haiku-s*`` cost file on disk.
SAMPLE_RUN_BUDGET = 0.05
SAMPLE_TOTAL_BUDGET = 0.30
RUN_BUDGET = 2.0
SAMPLE_INDICES = range(1, 10)

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


def llm_settings(model: str, route: str) -> tuple[str, dict]:
    """The model id and the ``with_llm`` keyword arguments for `model` on `route`."""
    params = dict(MODEL_PARAMS.get(model, {"temperature": 0.0}))
    if route == "openrouter":
        return OPENROUTER_MODEL, {**params, **OPENROUTER_PRICES}
    return MODELS[model], params


def sample_destination(task: str, sample: int, suffix: str, data: Path = DATA) -> Path:
    """Where ``--sample`` writes: ``labels/<task>/haiku-s<N><suffix>.jsonl``."""
    return data / "labels" / task / f"haiku-s{sample}{suffix}.jsonl"


def _served_models(result) -> list[str]:
    """The models the provider says answered, from the run's ledger."""
    ledger = getattr(result, "row_ledger", None)
    if ledger is None:
        return []
    return sorted({o.delivered_by for o in ledger.rows if o.delivered_by})


def sample_spend(data: Path = DATA) -> float:
    """USD already recorded in every ``labels/*/haiku-s*.cost.json`` (smoke runs too)."""
    total = 0.0
    for path in sorted((data / "labels").glob("*/haiku-s*.cost.json")):
        total += float(json.loads(path.read_text())["cost"])
    return total


def check_total_budget(run_budget: float, total_budget: float, spent: float) -> None:
    """Exit before any call when `spent` plus this run's budget passes the cap."""
    if round(spent + run_budget, 9) > round(total_budget, 9):
        raise SystemExit(
            f"total budget: ${spent:.4f} already spent on haiku-s* files + this "
            f"run's budget ${run_budget:.4f} exceeds ${total_budget:.4f}; nothing called"
        )


def refuse_response_cache(pipeline) -> None:
    """Exit when a response cache is on: a cache would fake unanimous samples."""
    llm = pipeline.specifications.llm
    litellm = sys.modules.get("litellm")
    cached = (
        getattr(llm, "cache_config", None)
        or (getattr(llm, "extra_params", None) or {}).get("caching")
        or getattr(litellm, "cache", None) is not None
    )
    if cached:
        raise SystemExit(
            "a response cache is configured; --sample draws would repeat one "
            "answer. Disable the cache and rerun; nothing called"
        )


def check_sample_args(args: argparse.Namespace) -> None:
    """Refuse a combination of the Addendum 1 options that could overwrite or mislead."""
    if args.sample is None:
        if args.route is not None or args.total_budget is not None:
            raise SystemExit("--route and --total-budget are only for --sample runs")
        return
    if args.model != "haiku":
        raise SystemExit("--sample is for --model haiku only")
    if args.sample not in SAMPLE_INDICES:
        raise SystemExit("--sample must be an index from 1 to 9")
    if args.candidates:
        raise SystemExit("--sample cannot be combined with --candidates")


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
        "--budget",
        type=float,
        help=f"USD ceiling for this run (default {RUN_BUDGET}; "
        f"{SAMPLE_RUN_BUDGET} with --sample)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="write extra Haiku sample N to haiku-s<N>.jsonl (Addendum 1); "
        "refuses to overwrite and never writes haiku.jsonl",
    )
    parser.add_argument(
        "--route",
        choices=["direct", "openrouter"],
        default=None,
        help="API route for --sample runs: OpenRouter (default, as pre-registered) "
        "or Anthropic direct",
    )
    parser.add_argument(
        "--total-budget",
        type=float,
        help="USD cap over every haiku-s* cost file plus this run's --budget "
        f"(default {SAMPLE_TOTAL_BUDGET} with --sample)",
    )
    args = parser.parse_args(argv)
    check_sample_args(args)
    sampling = args.sample is not None
    if args.route is None:
        args.route = "openrouter" if sampling else "direct"
    budget = args.budget
    if budget is None:
        budget = SAMPLE_RUN_BUDGET if sampling else RUN_BUDGET

    # A smoke run (--limit) must never overwrite a full run's labels.
    suffix = "-candidates" if args.candidates else ""
    if args.limit:
        suffix += f"-smoke{args.limit}"
    if sampling:
        dest = sample_destination(args.task, args.sample, suffix, DATA)
        for path in (dest, dest.with_suffix(".cost.json")):
            if path.exists():
                raise SystemExit(f"{path} exists; a sample is never overwritten")
        total = args.total_budget
        check_total_budget(
            budget, SAMPLE_TOTAL_BUDGET if total is None else total, sample_spend(DATA)
        )
    else:
        dest = DATA / "labels" / args.task / f"{args.model}{suffix}.jsonl"

    frame = load(args.task)
    if args.candidates:
        listed = DATA / "adjudication" / f"{args.task}.jsonl"
        ids = {json.loads(line)["id"] for line in listed.open(encoding="utf-8")}
        frame = frame[frame["id"].isin(ids)].reset_index(drop=True)
    if args.limit:
        frame = frame.head(args.limit)
    frame, inputs, prompt = prepare(args.task, frame)
    model_id, llm_params = llm_settings(args.model, args.route)

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(frame, input_columns=inputs, output_columns=["answer"])
        .with_prompt(prompt)
        .with_llm(model=model_id, **llm_params)
        .with_concurrency(6)
        .with_max_budget(budget)
        .with_error_policy("skip")
        .build()
    )
    if sampling:
        refuse_response_cache(pipeline)
        try:
            result = pipeline.execute()
        except BaseException:
            # Spend from a run that never wrote its cost file would escape the
            # total cap: charge it at its whole budget, under a name the
            # addendum ignores but sample_spend counts.
            aborted = dest.with_name(f"{dest.stem}-aborted.cost.json")
            aborted.parent.mkdir(parents=True, exist_ok=True)
            aborted.write_text(
                json.dumps({"cost": float(budget), "aborted": True, "rows": 0})
            )
            raise
    else:
        result = pipeline.execute()
    out = result.to_pandas()

    dest.parent.mkdir(parents=True, exist_ok=True)
    cost = float(result.costs.total_cost)
    with dest.open("w", encoding="utf-8") as f:
        for rid, answer in zip(frame["id"], out["answer"], strict=True):
            record = {"id": rid, "answer": None if pd.isna(answer) else answer}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    record = {"cost": cost, "tokens": result.costs.total_tokens, "rows": len(frame)}
    if sampling:
        record = {
            "route": args.route,
            "model": model_id,
            **record,
            "input_tokens": result.costs.input_tokens,
            "output_tokens": result.costs.output_tokens,
            "served_models": _served_models(result),
            "source": f"label.py --sample {args.sample} (result.costs)",
        }
    dest.with_suffix(".cost.json").write_text(json.dumps(record))

    lost = result.lost_row_indices
    if sampling:
        print(f"served by: {record['served_models']}")
    print(
        f"{args.task}/{args.model}: {len(frame)} rows, {len(lost)} lost, "
        f"cost ${cost:.4f}, tokens {result.costs.total_tokens}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
