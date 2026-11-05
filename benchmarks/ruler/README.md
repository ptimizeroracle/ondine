# Ruler validation

Proof that the intervals behind certified columns ("the Ruler") cover the
truth at least 95% of the time, before any of them ships. The code under
test, `intervals.py`, uses numpy only; phase 2 moves it into Ondine as is.

```bash
uv run python benchmarks/ruler/exact_coverage.py   # step 1, about 3 minutes
uv run python benchmarks/ruler/simulate.py         # step 2, seconds
```

Step 2 needs the golden data and its labels (`benchmarks/golden/build.py`,
then `label.py`). Each script writes its table to `results/` and exits 1
when its pass rule fails.

| Step | What | Pass rule | Result |
|---|---|---|---|
| 1 | Exact coverage of the single-source interval, from hypergeometric and binomial sums (scipy as the independent reference): n 2–2,000, N to 10⁶, accuracy 0.5–0.999 | minimum ≥ 95.0% | **pass**, 95.0012% |
| 2 | Coverage on populations built from the golden sets' real rows, 2,000 draws per cell: plain samples, unresolved rows (random and among wrong rows), dropped rows, status strata, two-tier | ≥ 94% in every cell | **pass**, lowest 94.40% |

Both rules were fixed before the first run.

What the widths showed (results/simulation.md):

- Status strata, combined by the union bound, are valid but 1.2–1.5× wider
  than a plain random sample of the same size. The headline uses a plain
  sample.
- Two-tier (a model labels the sample, people label part of it), combined
  by the union bound, is valid but 1.4–2.3× wider than the same number of
  people labels alone. It is out of v1.
- Against the adjudicated gold, Opus seems to share none of the run's
  intent mistakes, because that gold was settled partly by model votes.
  Against the original gold it shares 33–70% of them. Two-tier cells use
  the original-gold rate.
