"""List the dev rows a human should check, from two models' labels.

Reads ``data/labels/<task>/<model>.jsonl`` for two models and the gold in
``data/<task>/dev.jsonl``. A row is a *candidate* when both models
disagree with the gold. Those are the rows where the gold is most likely
wrong; the rows where only one model disagrees are far more likely the
model's mistake, and checking them would measure the models, not the gold.

    uv run python benchmarks/golden/adjudicate.py --task intent --a sol --b sonnet

Writes ``data/adjudication/<task>.jsonl`` with one candidate per line and
prints agreement figures. The human verdicts then go in
``benchmarks/golden/adjudications.jsonl`` (committed): row id, original
label, verdict, reason.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def norm(value: object) -> str:
    """Compare values ignoring case, spacing and punctuation."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def final_answer(answer: str | None) -> str | None:
    """The answer line of a reply that may reason first.

    Some models explain before they answer despite the instruction to reply
    with the label only. The label is on the last non-empty line, so score
    that rather than counting a correct, wordy reply as wrong. How often a
    model needs this is itself worth reporting (see `wordy` below).
    """
    if not answer:
        return None
    lines = [ln.strip() for ln in answer.strip().splitlines() if ln.strip()]
    return lines[-1].strip("*`\"' .") if lines else None


def parse_json_object(answer: str | None) -> dict | None:
    """The first JSON object in a model answer, tolerating a code fence."""
    if not answer:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip())
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)?)(?:\s+(\d+)/(\d+))?\s*(k?)\b", re.I)


def leading_number(value: object) -> float | None:
    """The number a value starts with: "8 1/2" -> 8.5, "15K" -> 15000, "3 in" -> 3."""
    m = _NUMBER.match(str(value))
    if not m:
        return None
    number = float(m.group(1))
    if m.group(2):
        number += int(m.group(2)) / int(m.group(3))
    return number * 1000 if m.group(4) else number


def same_value(got: object, accepted: object) -> bool:
    """Lenient match for *finding* disputed gold, not for scoring models.

    Equal after normalising, or the same leading number (units ignored),
    or one contains the other ("Rack" vs "Rack Server"). Strict scoring
    belongs to the benchmark; here a false candidate wastes a human's time.
    """
    g, a = norm(got), norm(accepted)
    if g == a:
        return True
    ng, na = leading_number(got), leading_number(accepted)
    if ng is not None and na is not None:
        return abs(ng - na) < 1e-9
    return bool(g) and bool(a) and a != "na" and g != "na" and (a in g or g in a)


def attribute_errors(target: dict[str, list[str]], answer: str | None) -> list[str]:
    """Attributes whose answer matches none of the accepted values."""
    parsed = parse_json_object(answer)
    if parsed is None:
        return sorted(target)  # an unparseable answer gets every attribute wrong
    wrong = []
    for attribute, accepted in target.items():
        got = parsed.get(attribute, "n/a")
        if not any(same_value(got, a) for a in accepted):
            wrong.append(attribute)
    return wrong


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def row_wrong(task: str, gold: dict, answer: str | None) -> bool:
    """Did the model get this row wrong against the gold?"""
    if task == "attributes":
        return bool(attribute_errors(gold["target"], answer))
    return norm(final_answer(answer)) != norm(gold["label"])


def report_attributes(gold: dict, labels: dict, args: argparse.Namespace) -> int:
    """Per-attribute scoring: a row is never "wrong" as a whole.

    Rows carry 8-20 attributes, so requiring all of them right marks every
    row wrong. The unit that matters is the attribute value, and the
    candidates for the human check are the (row, attribute) pairs both
    models get wrong.
    """
    pairs: dict[str, set[tuple[str, str]]] = {m: set() for m in labels}
    total = 0
    for rid, g in gold.items():
        total += len(g["target"])
        for m in labels:
            wrong_here = attribute_errors(g["target"], labels[m].get(rid))
            pairs[m] |= {(rid, a) for a in wrong_here}
    both = sorted(pairs[args.a] & pairs[args.b])
    print(f"attributes: {len(gold)} dev rows, {total} attribute values")
    for m in labels:
        share = len(pairs[m]) / total
        print(f"  {m:8s} wrong vs gold: {len(pairs[m]):4d} ({share:.1%})")
    print(
        f"  both wrong:      {len(both):4d} ({len(both) / total:.1%})"
        "  <- candidates for the human check"
    )
    by_attr: dict[str, int] = {}
    for _, attribute in both:
        by_attr[attribute] = by_attr.get(attribute, 0) + 1
    top = sorted(by_attr.items(), key=lambda kv: -kv[1])[:6]
    print(f"  most-disputed attributes: {top}")

    out = DATA / "adjudication" / "attributes.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rid, attribute in both:
            g = gold[rid]
            got = {
                m: (parse_json_object(labels[m].get(rid)) or {}).get(
                    attribute, "<unparsed>"
                )
                for m in labels
            }
            record = {
                "id": rid,
                "attribute": attribute,
                "title": g["title"],
                "gold": g["target"][attribute],
                **got,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  candidates -> {out.relative_to(HERE.parent.parent)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--task", required=True, choices=["intent", "attributes", "faithfulness"]
    )
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    args = parser.parse_args(argv)

    gold = {r["id"]: r for r in load_jsonl(DATA / args.task / "dev.jsonl")}
    labels = {
        m: {
            r["id"]: r["answer"]
            for r in load_jsonl(DATA / "labels" / args.task / f"{m}.jsonl")
        }
        for m in (args.a, args.b)
    }

    if args.task == "attributes":
        return report_attributes(gold, labels, args)
    wrong = {
        m: {i for i in gold if row_wrong(args.task, gold[i], labels[m].get(i))}
        for m in labels
    }
    both = sorted(wrong[args.a] & wrong[args.b])
    n = len(gold)
    print(f"{args.task}: {n} dev rows")
    for m in labels:
        print(f"  {m:8s} wrong vs gold: {len(wrong[m]):3d} ({len(wrong[m]) / n:.1%})")
    print(
        f"  both wrong:      {len(both):3d} ({len(both) / n:.1%})  <- candidates for the human check"
    )
    print(f"  only one wrong:  {len(wrong[args.a] ^ wrong[args.b]):3d}")

    out = DATA / "adjudication" / f"{args.task}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rid in both:
            g = gold[rid]
            f.write(
                json.dumps(
                    {
                        "id": rid,
                        "input": {
                            k: g[k]
                            for k in ("text", "doc", "summary", "title", "description")
                            if k in g
                        },
                        "gold": g.get("label") or g.get("target"),
                        args.a: labels[args.a].get(rid),
                        args.b: labels[args.b].get(rid),
                        **(
                            {
                                "wrong_attributes": sorted(
                                    set(
                                        attribute_errors(
                                            g["target"], labels[args.a].get(rid)
                                        )
                                    )
                                    & set(
                                        attribute_errors(
                                            g["target"], labels[args.b].get(rid)
                                        )
                                    )
                                )
                            }
                            if args.task == "attributes"
                            else {}
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"  candidates -> {out.relative_to(HERE.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
