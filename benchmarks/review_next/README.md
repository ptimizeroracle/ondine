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

## Result: pending

No ranker has been evaluated on the real dev rows. This section is filled in
after the one full run from the reviewed commit.
