# Planner replay

An offline, pre-registered experiment. It asks one question before any
Planner is built: does routing between models beat the cheapest strategy
that needs no Planner, which is to pick a single model or a random mix of
two?

The replay uses the five models' recorded answers on the golden dev rows
(`benchmarks/golden/data/labels/`). It makes no API call and reads no
held-out split.

**The design is pre-registered:** [`DESIGN.md`](DESIGN.md).
Every constant, policy, metric, procedure and pass rule is fixed there and
in the `PREREG` dict of `replay.py`. The results embed the sha256 of both.
**Do not tune a constant, a policy or a threshold after viewing a result.**
If the design is wrong, write a new pre-registration and say what the old
one produced.

## What it compares

- **Comparator:** the single-model hull, the upper concave envelope of the
  five models' (cost per row, metric) points on the same rows. A random mix
  of two models reaches any point on it.
- **Statistics:** `delta A = a_p - F(c_p)` (lift over the hull at equal cost)
  and `D2 = a_p - F(2 c_p)` (matches what the hull buys with twice the money).
- **Confirmatory policies** (only these decide the verdict): C1, a cascade
  that asks Haiku and Luna and escalates to the strong model `s` on
  disagreement; C2, a value router fed by 100 labelled rows; C3, both.
- **Exploratory grid** (75 configs per task; 78 on attributes, adding the three whole-row variants design section 5 asks for; frontier plot only), and
  unreachable oracles O1-O3 for reference.

## Run it

```bash
uv run python benchmarks/planner/replay.py --check   # integrity only, no policy is evaluated
uv run python benchmarks/planner/replay.py --quick   # B = 100: a smoke run, not citeable
uv run python benchmarks/planner/replay.py           # S = 1000 splits, B = 1000 bootstrap draws
```

`--check` loads the data and proves the loading is right: every dev id has
an answer from every model, `score.score` reproduces
`benchmarks/golden/data/baseline-report.json` to 1e-9 for every model and
task on both golds, the arrays the study uses reproduce `score.score`, and
the loader refuses the held-out split. Exit code 2 means an integrity
failure; the study does not start. Run time: full under a minute,
`--quick` under ten seconds. The script needs numpy only.

## Where results go

`benchmarks/planner/results/` (`results/quick/` for `--quick`):

| File | Content |
|---|---|
| `replay.md` | integrity, verdict, per-task tables, diagnostics, the full exploratory grid |
| `replay.json` | everything, with the two hashes, the constants and the seeds |
| `frontier-<task>.svg` | cost against metric: single models, hull, policies, grid |

A `--quick` run is a smoke test; only a full run under the reviewed
`PREREG` hash counts.

## Tests

`tests/unit/test_planner_replay.py` checks every estimator, the hull, the
bootstrap, the Bonferroni bound and the verdict logic on synthetic tasks
where the right answer follows from how they are built (routing that cannot
help is killed; a planted weak value lifts C2; C2 never reads a test row's
label). It uses no real policy number.

## Scope

Dev rows only: the point of the study is to decide whether a Planner is
worth building, not to certify one. A positive verdict is re-tested on the
held-out split at release, by someone who has not seen these results.

## Caveats fixed before the run (independent review, 2026-11-17)

- **C3 cannot win rule 2's "beat C1" branch.** It asks the same two models as C1 and routes a superset of C1's rows, so its cost is never below C1's. When C1 passes, only C2 can earn the Ruler-fed verdict. This is a property of the design, not a bug.
- **The bootstrap fit is slightly conservative for C2 and C3.** A labelled row drawn twice counts twice, so the fit sees fewer distinct rows than the point estimate. `delta_boot_mean` in `replay.json` sits a little below the point estimate.
- **Intent length weighting counts only `text`.** Every intent prompt also carries the 77-name label list, so per-row cost varies less than the weights say. It cannot cause a false pass: a policy must pass under both cost models.
- **Provenance:** results record the design's hash, the PREREG hash, the script's own hash, and the git commit with a dirty flag.

## Result (2026-11-17): KILL

Run once, unchanged, from the pre-registration commit `91e5352` (clean tree). Full tables: [`results/replay.md`](results/replay.md).

No confirmatory policy passes on any task. On intent (the deciding task, original gold, out of sample):

| Policy | cost vs Opus | accuracy | lift over the hull at equal cost (lower bound) |
|---|---:|---:|---:|
| C1 cheap-pair cascade | 14% | 83.1% | +0.8 (−1.7) |
| C2 value router, 100 labels | 34% | 84.0% | −0.0 (−3.0) |
| C3 both | 41% | 85.3% | +0.9 (−2.0) |

Why, from the diagnostics:

- **The money is there; the signals cannot find it.** The unreachable per-row oracle reaches 91.3% at 7% of Opus's cost (Opus alone: 88.3%).
- **Cheap models share their mistakes.** Haiku and Luna agree on 75% of Haiku's wrong intent answers (61% on faithfulness), so disagreement misses most errors.
- **100 labels are too thin per value.** On 77 intents, C2 fitted on every row would gain 5.7 points over C2 fitted on 100 (O3 − C2): the per-value error rates the Ruler can measure from a sample do not transfer.

What this does and does not kill: routing on agreement or on per-value error rates from ~100 labels. It does not test signals we hold no data for (token probabilities, self-consistency across samples, a verifier model); those need new API runs.

## Addendum 1 (2026-11-17): Haiku self-consistency, KILL STANDS

Pre-registered in [`DESIGN-addendum-1.md`](DESIGN-addendum-1.md) (commit `a9dd6db`) before any sample was drawn; run once from that commit, clean tree. Full tables: [`results/addendum-1.md`](results/addendum-1.md). Three extra Haiku samples per row, through OpenRouter, cost $0.19; every route check passed (accuracy 79.7–81.0% vs 81.0% originally, 93–97 output tokens per row vs 95, agreement among new samples 95.4% vs 95.2% with the original).

**Intent (decides): neither policy passes.**

| Policy | cost vs Opus | accuracy | lift at equal cost (lower bound) |
|---|---:|---:|---:|
| C4: route unless all 4 Haiku answers agree | 19% | 83.1% | +0.2 (−2.7) |
| C5: C4 or Haiku–Luna disagreement | 25% | 84.0% | +0.4 (−2.6) |

**Mechanism: not useful.** Self-consistency flags 4 of the 33 rows Haiku and Luna get wrong together (12%; pre-registered "useful" needed ≥ 14). Haiku repeats its own mistakes: the errors the cheap pair shares are stable across samples too.

**Faithfulness (scope only, cannot change the verdict):** C4 passes by CUT, narrowly (lift +2.7, lower bound +0.2; D2 lower bound −1.9 against −2.0). On adjudicated gold it flags 10 of Haiku's 13 wrong rows with 5% false flags. This is a lead for judgment tasks, not a finding: it was seen after the fact on one task, so it needs its own pre-registration and the faithfulness holdout before anyone builds on it.
