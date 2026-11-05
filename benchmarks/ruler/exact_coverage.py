"""Step 1 of the Ruler's proof: exact coverage of the single-source interval.

Computed from hypergeometric and binomial sums, not simulated: for each
population, sample size and true accuracy, the coverage is the probability,
summed over every possible sample result, that the interval contains the
truth. The bounds come from ``ondine.ruler.intervals`` (the code that ships; the
binomial form from ``combinations.py``); the
probabilities come from scipy, an independent implementation.

Pass rule (fixed in the design before running): the minimum coverage over
the whole grid is at least 95.0%.

    uv run python benchmarks/ruler/exact_coverage.py

Writes ``results/exact-coverage.md`` and exits 1 if the rule fails.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
from combinations import binom_bounds
from scipy import stats

from ondine.ruler.intervals import hypergeom_bounds

HERE = Path(__file__).resolve().parent
ALPHA = 0.05
PASS = 0.95
SAMPLES = (2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
POPULATIONS = (1.1, 2, 10, 10_000, 100_000, 1_000_000)  # < 100: times n
RATES = np.linspace(0.5, 0.999, 400)


def populations(n: int) -> list[int]:
    sizes = {math.ceil(f * n) if f < 100 else int(f) for f in POPULATIONS}
    return sorted(N for N in sizes if math.ceil(1.1 * n) <= N)


def hypergeom_case(n: int, N: int) -> dict:
    x = np.arange(n + 1)
    low, high = hypergeom_bounds(x, n, N, ALPHA)
    counts = np.unique(np.round(RATES * N).astype(np.int64))
    pmf = stats.hypergeom.pmf(x[None, :], N, counts[:, None], n)
    covered = (low[None, :] <= counts[:, None]) & (counts[:, None] <= high[None, :])
    coverage = (pmf * covered).sum(axis=1)
    worst = int(np.argmin(coverage))
    at = int(round(0.9 * n))
    return {
        "n": n,
        "N": N,
        "min": float(coverage[worst]),
        "at": float(counts[worst] / N),
        "mean": float(coverage.mean()),
        "width90": float((high[at] - low[at]) / N),
    }


def binom_case(n: int) -> dict:
    x = np.arange(n + 1)
    low, high = binom_bounds(x, n, ALPHA)
    pmf = stats.binom.pmf(x[None, :], n, RATES[:, None])
    covered = (low[None, :] <= RATES[:, None]) & (RATES[:, None] <= high[None, :])
    coverage = (pmf * covered).sum(axis=1)
    worst = int(np.argmin(coverage))
    at = int(round(0.9 * n))
    return {
        "n": n,
        "N": "∞",
        "min": float(coverage[worst]),
        "at": float(RATES[worst]),
        "mean": float(coverage.mean()),
        "width90": float(high[at] - low[at]),
    }


def main() -> int:
    started = time.time()
    rows = [hypergeom_case(n, N) for n in SAMPLES for N in populations(n)]
    rows += [binom_case(n) for n in SAMPLES]
    worst = min(rows, key=lambda r: r["min"])
    passed = worst["min"] >= PASS

    lines = [
        "# Exact coverage of the single-source interval",
        "",
        f"Accuracy {RATES[0]:.3f} to {RATES[-1]:.3f} ({RATES.size} points per case), "
        f"{len(rows)} cases, {time.time() - started:.0f} s. "
        f"Pass rule: minimum coverage ≥ {PASS:.1%}.",
        "",
        f"**{'PASS' if passed else 'FAIL'}**: minimum {worst['min']:.4%} "
        f"(n={worst['n']}, N={worst['N']}, accuracy {worst['at']:.3f}).",
        "",
        "N = ∞ is the binomial (Clopper–Pearson) form. Width is the full width",
        "of the interval at 90% sampled accuracy.",
        "",
        "| n | N | min coverage | at accuracy | mean coverage | width at 90% |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['n']} | {r['N']} | {r['min']:.4%} | {r['at']:.3f} | "
            f"{r['mean']:.4%} | {r['width90']:.4f} |"
        )
    out = HERE / "results" / "exact-coverage.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:6]))
    print(f"wrote {out.relative_to(HERE.parent.parent)}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
