# Review-next

An offline, pre-registered experiment. The Ruler labels 100 sampled rows and
reports an error rate. Review-next asks whether a ranking signal can order
the other delivered rows so that a person reviewing the top k finds more
errors per row than random review, and by enough more than the signal the
Ruler gives for free (per-value error rates) to ship.

It uses the recorded Haiku, Luna and extra Haiku-sample answers on the golden
dev rows (`benchmarks/golden/data/labels/`). It makes no API call and reads no
held-out split. Tasks: intent and faithfulness.

**The design is pre-registered:** [`DESIGN.md`](DESIGN.md) (locked). Every
constant is fixed there and in the `PREREG_REVIEW` dict of `review_next.py`.
The results embed the sha256 of both, and of the two parent designs.
**Do not tune a constant, a ranker or a threshold after viewing a result.**
If the design is wrong, write a new pre-registration and say what the old one
produced.

## What it compares

- **R0** random (the null), **R1** per-value error rate fitted on the 100
  labelled rows (the comparator), **R2** self-consistency (Haiku's dissent D
  over 3 extra samples, then R1), **R4** dissent plus the Haiku/Luna pair
  disagreement (D + P, then R1). R2 and R4 are confirmatory; eight
  exploratory rankers (E1-E8) never decide the verdict.
- **Metric:** errors per 100 reviewed at k = 10% of the test rows (5% and 20%
  as sensitivity), lift L over random and gain G in points over R1. Ties are
  scored in expectation, exactly.
- **Statistics:** the parent's splits (S = 1000) and cluster bootstrap
  (B = 1000, 20 splits per draw), with a one-sided Bonferroni bound at q = 0.0125.
- **Verdict:** PASS needs L >= 2, LB(L) > 1, G >= 10 and LB(G) > 0; KILL is
  G < 5 or L < 1.5. The thresholds are compared exactly, with no tolerance.
  BUILD_GENERAL, BUILD_SCOPED, KILL or INCONCLUSIVE follows (DESIGN section 5).
- **Selection rule:** R2 unless R4 beats R2 by >= 5 points with paired LB > 0
  on every task in scope. When different rankers pass on different tasks, the
  verdict lists each `BUILD_SCOPED(task, ranker)`.

## Run it

```bash
uv run python benchmarks/review_next/review_next.py --check   # checks only; R0 is the only ranker run
uv run python benchmarks/review_next/review_next.py           # the one full run: S = 1000 splits, B = 1000 draws
```

`--check` runs the parents' integrity and sample checks (V1-V4, the parent's
C1 reproduction), compares the 19 data files with the PREREG hashes, the
design with its locked hash and the parent designs with theirs, and
recomputes every count the design discloses. It prints the test rows per
split and k, and checks that R0's lift is 1 on every split. It evaluates no
other ranker. Exit code 2 means a failed check; the study does not start. The
script needs numpy only.

**There is one full run, from the clean, reviewed commit.** A full run on a
tree with uncommitted changes under `benchmarks/review_next`,
`benchmarks/planner`, `benchmarks/golden` or the study's tests exits 2, and so
does a second full run that would overwrite `results/review-next.*`.

**Do not run `--quick` before the full run.** It also uses S = 1000 splits and
prints the verdict, so on the real data it reveals the outcome. It is refused
(exit 2) unless `--out` names a scratch directory (for a fixture) or
`results/review-next.json` already exists.

## Where results go

`benchmarks/review_next/results/`:

| File | Content |
|---|---|
| `review-next.md` | verdict, checks, per-task tables, adjudicated control, mechanism, break-even |
| `review-next.json` | everything, with the design, PREREG and script hashes, the git state and the data hashes |

## Tests

`tests/unit/test_review_next.py` checks the tie handling, k rounding, the
signals, the key order, the bootstrap pairing and bound, the verdict rules and
`--check` on synthetic data and a golden-layout fixture. It reads no real
policy number.

## Result (2026-11-17)

One full run from clean commit `a1b4367`: **BUILD_SCOPED(faithfulness, R2)**.
Full tables: [`results/review-next.md`](results/review-next.md).

At k = 10% of the unsampled rows, original gold, mean over 1000 splits:

| Task | Ranker | errors per 100 reviewed | lift L (LB) | gain G over R1 (LB) | status |
|---|---|---|---|---|---|
| faithfulness | R1 (free, per value) | 16.1 | 0.90 | - | - |
| faithfulness | R2 (dissent) | 54.9 | 3.06 (1.72) | +38.8 (+11.5) | PASS |
| faithfulness | R4 (dissent + pair) | 56.5 | 3.16 (1.81) | +40.4 (+12.1) | PASS |
| intent | R1 (free, per value) | 34.7 | 1.82 | - | - |
| intent | R2 (dissent) | 52.2 | 2.73 (1.50) | +17.5 (-7.4) | INCONCLUSIVE |
| intent | R4 (dissent + pair) | 53.9 | 2.82 (1.63) | +19.2 (-8.0) | INCONCLUSIVE |

- R4 does not beat R2 by the pre-registered margin (faithfulness +1.6 points,
  paired LB -2.8), so R2 is the selected ranker: it needs one model only.
- Intent is underpowered, as the design predicted: only 20 rows are flagged,
  and the bootstrap SE of G is about 11 points. A gain of +17.5 with a lower
  bound of -7.4 is not a finding in either direction. Intent carries the
  SUBSET flag (one of the two-sample variants gains under 10 points).
- Faithfulness carries no flag. On the adjudicated gold R2 reaches lift 7.6
  and recall 0.78 at k = 10%.
- The faithfulness gain is largely implied by Addendum 1's flag precision, and
  R1 cannot rank a two-value column (lift 0.90). Dev was reused: the flag was
  found on these rows. **Nothing ships on dev alone.** The next step is a
  pre-registered check on the held-out faithfulness rows.
- Exploratory, not a verdict: the free per-value ranker R1 lifts intent
  review to 1.82x random with no extra model call.

## Confirmation

The result above was found on dev rows, so it is confirmed on fresh rows before
anything ships. The design is pre-registered and locked:
[`DESIGN-confirm.md`](DESIGN-confirm.md). Every constant is in the
`PREREG_CONFIRM` dict of `confirm.py`; the results embed its hash and the
parent's. It reuses `review_next.py`, `replay.py` and `addendum1.py` by import,
changes none of them, and reads no held-out split: every data read goes through
`guard_confirm`, which allows only the confirm rows, the Haiku sample files, the
dev rows (ids only) and the intent label names.

**One ranker per task, R2** (dissent D over three extra Haiku samples, then R1),
against random (R0) and the free per-value rate (R1). Rows: 770 intent and 320
faithfulness, never used by dev. Faithfulness bootstraps over documents. The
verdict is CONFIRMED_GENERAL, CONFIRMED(faithfulness), NOT_REPLICATED or
INCONCLUSIVE (design section 5).

Run order (each step needs the one before):

```bash
# 1. Build the rows. Deterministic; never downloads, never writes dev or the second split.
uv run python benchmarks/golden/build.py --confirm
# 2. Commit the manifest's confirm hashes and fill confirm_split_sha256 in PREREG_CONFIRM.
# 3. Smoke (3 rows, $0.01 cap), then four sample runs per task: OpenRouter Haiku, $1.00 total cap.
uv run python benchmarks/golden/label.py --task faithfulness --model haiku --split confirm --sample 1 --limit 3
uv run python benchmarks/golden/label.py --task faithfulness --model haiku --split confirm --sample 1   # ... --sample 4, then intent
# 4. Check, lock the data, commit the lock file, then the one full run from the clean commit.
uv run python benchmarks/review_next/confirm.py --check    # exit 0 ok, 2 failed, 3 data not all there yet
uv run python benchmarks/review_next/confirm.py --lock
uv run python benchmarks/review_next/confirm.py
```

Rules:

- `label.py --split confirm` needs `--model haiku` and `--sample 1` to `5`, always
  uses OpenRouter and writes `labels/<task>/confirm/haiku-s<N>.jsonl`. Every cost
  file there (smoke and aborted runs too) counts toward one $1.00 cap, checked
  before any call. Per-run defaults: intent $0.14, faithfulness $0.09, `--limit`
  $0.01; `--budget` may only lower them. `haiku-s5` is the single replacement for
  one file that fails V4, and `label.py` refuses it until one of `haiku-s1` to
  `s4` exists for that task. A run writes a hold file for its whole budget before
  any call and deletes it only once its cost file exists, so a crash at any point
  stays on the books; `confirm.py --lock` refuses while a hold file exists.
- `--check` exits 3 and lists what is missing until the rows, the four sample
  files per task and the lock exist. It never reports ok on incomplete data.
- The full run exits 2 on uncommitted changes under `benchmarks/review_next`
  (not `results/`), `benchmarks/planner`, `benchmarks/golden` or `tests/unit`, on
  an existing `results/confirm.json`, or when a data file no longer matches the
  lock. `--quick` is refused on the real data until the full result exists.
- Results go to `results/confirm.md` and `results/confirm.json`.

Tests: `tests/unit/test_review_next_confirm.py`, `tests/unit/test_golden_build_confirm.py`
and `tests/unit/test_golden_label_confirm.py`, all on synthetic data.

### Confirmation result (2026-11-17)

One full run from clean commit `e87e8fc`. Overall verdict **INCONCLUSIVE** (design
section 5: intent PASS, faithfulness INCONCLUSIVE). Spend $0.6206 of the $1.00
cap. Full tables: [`results/confirm.md`](results/confirm.md).

At k = 10% of the unsampled rows, original gold, mean over 1000 splits:

| Task | Ranker | errors per 100 reviewed | lift L (LB) | gain G over R1 (LB) | status |
|---|---|---|---|---|---|
| intent (770 fresh, independent rows) | R1 (free) | 32.1 | 1.78 | - | - |
| intent | R2 (dissent) | 53.9 | 2.98 (2.41) | +21.8 (+10.6) | PASS |
| faithfulness (320 fresh rows, dev documents) | R1 (free) | 19.0 | 1.04 | - | - |
| faithfulness | R2 (dissent) | 35.0 | 1.92 (1.16) | +16.0 (+1.8) | INCONCLUSIVE |

- Intent passes all four pre-registered conditions on rows that share nothing
  with dev. Dev had left it INCONCLUSIVE (lower bound -7.4); the extra 540 rows
  settle it. Its lower bound on the gain is +10.6, only just above the +10
  threshold's neighbourhood, so read the gain as "about +20 +- 12", not +21.8.
- Faithfulness misses one condition: mean lift 1.92 is under the 2.0 floor. Its
  other three conditions pass (G +16.0, LB(G) +1.8, LB(L) 1.16). It is not a
  KILL (G >= 5 and L >= 1.5). The effect shrank from dev (54.9 errors per 100
  reviewed, G +38.8): the regression the design warned about, larger than the
  ~30-point gain it expected.
- DOMAIN flag: the gain is negative in 3 of 8 domains (news, qmsumm,
  sales_call). Each domain has about 2 reviewed rows per split, so this is
  descriptive and noisy.
- The design has no outcome "intent passes, faithfulness inconclusive": the
  rule gives INCONCLUSIVE, and that stands. The per-task PASS is the finding.
- Faithfulness shares documents with dev; only the release split tests new
  documents. The release split was never read.
