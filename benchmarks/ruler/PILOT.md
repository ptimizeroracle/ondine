# Human pilot for the review sheet

Design: `.claude/designs/certified-columns.md`, sections 8 (step 3) and 13.9.

**This file was written before any person labelled a row. Do not edit the pass rule after the first sheet goes out.**

## What the pilot checks

Two people label 100 intent rows, blind, with the "several answers" and `?` options. We learn four things:

1. How often the two people agree, and Cohen's kappa.
2. Whether our adjudicated gold (`dev-adjudicated.jsonl`, 218 of the 231 dev rows) matches what people say.
3. How long labelling takes, in minutes per row.
4. How many problems their first attempt has when Ondine reads the sheets, per 100 rows.

It makes no API call. It replays Haiku's recorded intent answers through the real pipeline, so it costs nothing in API spend. The people's time is the only cost: about 1 to 2 hours each, at 0.5 to 1 minute per row. The pilot measures it.

## The pass rule (fixed)

> The adjudicated gold **matches people** if each person agrees with it on at least **85%** of the rows both cover.

- A row is covered by a person if they gave at least one answer (not `?`, not empty).
- A row is covered by the gold if it is one of the 218 adjudicated rows.
- A person **agrees** on a row only if they gave **exactly one answer** and it equals the adjudicated label. A row with several answers is never agreement, even if one of them is the gold. It stays in the rows both cover.
- Both people must reach 85%. One person at 84.9% is a fail.

The rule was tightened before anyone labelled. Counting a several-answer row as agreement would let a person who lists two labels on every row agree 100% of the time.

`pilot.md` also reports, without a pass rule: each person's several-answer rows (count and share of the rows both cover), the agreement between the two, kappa, the rows marked `?` or given several answers (and how many are among the 13 rows the adjudicated gold drops), minutes per row, import problems per 100 rows, and whether the range from people's labels contains Haiku's accuracy against the adjudicated gold. That last comparison is rough: Haiku's accuracy is on the 218 adjudicated rows, while the range describes all delivered rows, from a 100-row sample.

`pilot.md` also records the sha256 of each returned sheet. `score` refuses to overwrite an existing `pilot.md` unless you pass `--force`, so the numbers cannot be rerun until they look good.

## What a pass and a fail mean

- **Pass:** the adjudicated gold can be called a label set people agree with, on this task.
- **Fail:** it cannot. Say so wherever the adjudicated gold is cited. Read `pilot.md` to see whether the disagreement is the gold's (rows people agree with each other against it) or the people's (rows they split on).

The pilot does not gate the merge of the review sheet. It decides what the guide may claim (see "Labels from people" in `docs/guides/certified-columns.md`).

## How to run it

```bash
uv run python benchmarks/ruler/pilot.py export --out pilot-review
```

1. Send person A `pilot-review/for_labellers/sheet-A.xlsx` and person B `sheet-B.xlsx`. Send nothing else. **Never send `pilot-review/keep_private/`.**
2. Time each person from the first row to the last.
3. Put the returned files back where they were, or pass `--sheet-a` and `--sheet-b`.

```bash
uv run python benchmarks/ruler/pilot.py score --folder pilot-review \
    --minutes-a 55 --minutes-b 70
```

If the sheets have problems, `score` lists every one and exits with 2. Fix the cells and run it again. The number of problems found on the first read is kept, and the report states it.

`score` writes `benchmarks/ruler/results/pilot.md` and exits 0 on a pass, 1 on a fail.
