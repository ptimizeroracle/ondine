"""Step 2 of the Ruler's proof: coverage on synthetic populations.

Populations are built from the golden sets' real rows (run answer, two
reference models, two golds), then sampled without replacement, 2,000 draws
per cell. Each draw is an exact random sample of that population (numpy's
multivariate hypergeometric sampler), and the bounds come from
``intervals.py``, the code that ships. A draw is covered when the bounds
contain the population's own accuracy (or agreement), counted on its rows.

Cell families (design section 8, step 2):

* ``real``: real rows resampled to N; plain sample of 200, and the same
  sample with 5% of rows unresolved, at random or five times as often among
  wrong rows. Run status comes from Ondine's judge on the raw answer.
* ``strata``: a stress population with small, worse strata (repaired 5% at
  70% accuracy, unverified 1% at 50%), both alpha splits.
* ``dropped``: intent against the adjudicated gold, where the 13 rows with
  no defensible label are unresolved; covered whatever they hold.
* ``two-tier``: a model labels the 200 sampled rows; people label 25, 50 or
  100 of them. Accuracy to 99.5% and disagreement 1-10%, with the
  shared-mistake rate measured on the real triples.

Pass rule (fixed in the design before running): coverage at least 94% in
every cell. The Monte Carlo band at 2,000 draws is about ±1 point.

    uv run python benchmarks/ruler/simulate.py

Needs the golden data (``benchmarks/golden/build.py``) and its labels.
Writes ``results/simulation.md`` and exits 1 if the rule fails.
"""

from __future__ import annotations

import json
import math
import sys
import time
from functools import cache
from pathlib import Path

import numpy as np
from intervals import (
    allocate,
    binom_bounds,
    hypergeom_bounds,
    stratum_alphas,
    two_tier_bounds,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "golden"))
from adjudicate import norm  # noqa: E402

from ondine.harness.contract import OutputField, build_contract  # noqa: E402
from ondine.harness.judge import judge  # noqa: E402

DATA = HERE.parent / "golden" / "data"
ALPHA = 0.05
PASS = 0.94
DRAWS = 2000
SAMPLE = 200
SIZES = (1_000, 10_000, 100_000)
SEED = 20261104


# ── real rows ────────────────────────────────────────────────────────────


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def _delivered(task: str, model: str) -> dict[str, tuple[str, str]]:
    """Row id -> (status, delivered value), as Ondine's judge decides."""
    if task == "intent":
        allowed = json.loads((DATA / "intent" / "labels.json").read_text())
    else:
        allowed = ["consistent", "inconsistent"]
    contract = build_contract(fields=[OutputField("label", allowed=allowed)])
    out = {}
    for row in _load(DATA / "labels" / task / f"{model}.jsonl"):
        verdict = judge(contract, row["answer"])
        if verdict.passed:
            status = "repaired" if verdict.format_fixed else "accepted"
            out[row["id"]] = (status, norm(verdict.values["label"]))
    return out


def real_rows(task: str, run: str) -> list[dict]:
    """One dict per delivered row: status, right under each gold, agreement."""
    original = {r["id"]: norm(r["label"]) for r in _load(DATA / task / "dev.jsonl")}
    adjudicated = {
        r["id"]: norm(r["label"]) for r in _load(DATA / task / "dev-adjudicated.jsonl")
    }
    refs = {m: _delivered(task, m) for m in ("opus", "sol")}
    rows = []
    for row_id, (status, value) in _delivered(task, run).items():
        rows.append(
            {
                "status": status,
                "original": value == original[row_id],
                "adjudicated": (value == adjudicated[row_id])
                if row_id in adjudicated
                else None,
                **{m: refs[m].get(row_id, (None, None))[1] == value for m in refs},
            }
        )
    return rows


def shared_mistake_rate(rows: list[dict], gold: str, ref: str = "opus") -> float:
    """P(reference agrees with the run | run is wrong).

    Measured against the original gold for the two-tier cells: the
    adjudicated gold was settled partly by model votes, Opus's among them,
    so against it Opus seems to share none of the run's intent mistakes.
    """
    wrong = [r for r in rows if r[gold] is False]
    return sum(r[ref] for r in wrong) / len(wrong)


# ── interval tables, cached per design ───────────────────────────────────


@cache
def exact_table(n: int, N: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    return hypergeom_bounds(np.arange(n + 1), n, N, alpha)


@cache
def cp_table(h: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    return binom_bounds(np.arange(h + 1), h, alpha)


# ── cells ────────────────────────────────────────────────────────────────


def _result(
    family: str, label: str, covered: np.ndarray, width: np.ndarray, **extra
) -> dict:
    return {
        "family": family,
        "cell": label,
        "coverage": float(covered.mean()),
        "width": float(width.mean()),
        **extra,
    }


def plain_cell(rng, colors: np.ndarray, label: str, family: str, truths=None) -> dict:
    """Simple random sample of SAMPLE rows; colors = right, wrong,
    unresolved-right, unresolved-wrong counts. ``truths`` overrides the
    accuracy the bounds must cover (several values: all must be covered)."""
    N = int(colors.sum())
    draws = rng.multivariate_hypergeometric(colors, SAMPLE, size=DRAWS)
    low, high = exact_table(SAMPLE, N, ALPHA)
    x, u = draws[:, 0], draws[:, 2] + draws[:, 3]
    lo, hi = low[x] / N, high[x + u] / N
    truths = truths if truths is not None else [(colors[0] + colors[2]) / N]
    covered = np.all([(lo <= t) & (t <= hi) for t in truths], axis=0)
    return _result(family, label, covered, hi - lo, accuracy=float(truths[0]))


def strata_cell(rng, strata: list[np.ndarray], label: str, split: str) -> dict:
    """Stratified sample; each stratum's colors = right, wrong counts."""
    sizes = np.array([int(c.sum()) for c in strata])
    N = int(sizes.sum())
    take = allocate(sizes, SAMPLE)
    census = take == sizes
    alphas = stratum_alphas(sizes / N, census, ALPHA, split)
    lo = np.zeros(DRAWS)
    hi = np.zeros(DRAWS)
    for colors, n_h, N_h, a_h, whole in zip(
        strata, take, sizes, alphas, census, strict=True
    ):
        x = rng.multivariate_hypergeometric(colors, int(n_h), size=DRAWS)[:, 0]
        if whole:
            lo += x / N
            hi += x / N
        else:
            low, high = exact_table(int(n_h), int(N_h), float(a_h))
            lo += low[x] / N
            hi += high[x] / N
    truth = sum(int(c[0]) for c in strata) / N
    covered = (lo <= truth) & (truth <= hi)
    # Plain: the same sample size, unstratified, on the same population.
    k = sum(int(c[0]) for c in strata)
    x = rng.hypergeometric(k, N - k, SAMPLE, size=DRAWS)
    p_low, p_high = exact_table(SAMPLE, N, ALPHA)
    plain = (p_high[x] - p_low[x]) / N
    return _result(
        "strata", label, covered, hi - lo, accuracy=truth, plain=float(plain.mean())
    )


def two_tier_cell(rng, cells: np.ndarray, people: int, label: str) -> dict:
    """cells = counts of (right, agree), (right, disagree), (wrong, agree),
    (wrong, disagree) in the population; agree = reference matches the run."""
    N = int(cells.sum())
    third = ALPHA / 3
    draws = rng.multivariate_hypergeometric(cells, SAMPLE, size=DRAWS)
    ra, rd, wa, wd = draws.T
    n_agree, n_disagree = ra + wa, rd + wd
    h_d = np.minimum(n_disagree, math.ceil(people / 2))
    h_a = np.minimum(n_agree, people - h_d)
    right_a = _labelled_right(rng, ra, wa, h_a)
    right_d = _labelled_right(rng, rd, wd, h_d)

    low, high = exact_table(SAMPLE, N, third)
    share = (low[n_disagree] / N, high[n_disagree] / N)
    agree_acc = _rate_bounds(right_a, h_a, third)
    disagree_acc = _rate_bounds(right_d, h_d, third)
    lo, hi = two_tier_bounds(share, agree_acc, disagree_acc)

    truth = (cells[0] + cells[1]) / N
    covered = (lo <= truth) & (truth <= hi)
    # People only: the same number of labels on a plain random sample.
    k = int(cells[0] + cells[1])
    x = rng.hypergeometric(k, N - k, people, size=DRAWS)
    p_low, p_high = exact_table(people, N, ALPHA)
    people_width = (p_high[x] - p_low[x]) / N
    return _result(
        "two-tier",
        label,
        covered,
        hi - lo,
        accuracy=truth,
        people_only=float(people_width.mean()),
    )


def _labelled_right(
    rng, good: np.ndarray, bad: np.ndarray, take: np.ndarray
) -> np.ndarray:
    """Right answers among ``take`` rows drawn from good+bad sampled rows."""
    out = np.zeros_like(take)
    some = take > 0
    out[some] = rng.hypergeometric(good[some], bad[some], take[some])
    return out


def _rate_bounds(right: np.ndarray, labelled: np.ndarray, alpha: float):
    lo = np.zeros(right.size)
    hi = np.ones(right.size)
    for h in np.unique(labelled):
        if h == 0:
            continue  # no labels: the rate is anywhere in [0, 1]
        mask = labelled == h
        low, high = cp_table(int(h), alpha)
        lo[mask], hi[mask] = low[right[mask]], high[right[mask]]
    return lo, hi


# ── populations ──────────────────────────────────────────────────────────


def resample(rng, rows: list[dict], N: int) -> list[dict]:
    return [rows[i] for i in rng.integers(0, len(rows), N)]


def colors_of(
    rows: list[dict], gold: str, unresolved: np.ndarray | None = None
) -> np.ndarray:
    right = np.array([bool(r[gold]) for r in rows])
    gone = np.zeros(len(rows), bool) if unresolved is None else unresolved
    return np.array(
        [
            (right & ~gone).sum(),
            (~right & ~gone).sum(),
            (right & gone).sum(),
            (~right & gone).sum(),
        ]
    )


def unresolved_mask(
    rng, rows: list[dict], gold: str, rate: float, wrong_weight: float
) -> np.ndarray:
    """Mark ``rate`` of rows unresolved; wrong rows ``wrong_weight`` times as likely."""
    weight = np.array([1.0 if r[gold] else wrong_weight for r in rows])
    pick = rng.choice(
        len(rows), size=round(rate * len(rows)), replace=False, p=weight / weight.sum()
    )
    mask = np.zeros(len(rows), bool)
    mask[pick] = True
    return mask


def by_cell(
    rng, N: int, accuracy: float, disagree: float, shared: float
) -> np.ndarray | None:
    """Population counts for (right, agree), (right, disagree), (wrong, agree),
    (wrong, disagree); None when the rates cannot coexist."""
    wrong = 1 - accuracy
    wa, wd = wrong * shared, wrong * (1 - shared)
    rd = disagree - wd
    if rd < 0:
        return None
    probs = np.array([accuracy - rd, rd, wa, wd])
    return rng.multinomial(N, probs)


# ── main ─────────────────────────────────────────────────────────────────


def main() -> int:
    started = time.time()
    rng = np.random.default_rng(SEED)
    results: list[dict] = []
    shared: dict[str, float] = {}

    for task in ("intent", "faithfulness"):
        for run in ("haiku", "luna"):
            rows = real_rows(task, run)
            for gold in ("original", "adjudicated"):
                shared[f"{task}/{run}, {gold}"] = shared_mistake_rate(rows, gold)
            for gold in ("original", "adjudicated"):
                pool = [r for r in rows if r[gold] is not None]
                for N in SIZES:
                    pop = resample(rng, pool, N)
                    base = f"{task}/{run}, {gold} gold, N={N:,}"
                    results.append(plain_cell(rng, colors_of(pop, gold), base, "real"))
                    for how, weight in (("at random", 1.0), ("5x among wrong", 5.0)):
                        mask = unresolved_mask(rng, pop, gold, 0.05, weight)
                        results.append(
                            plain_cell(
                                rng,
                                colors_of(pop, gold, mask),
                                f"{base}, 5% unresolved {how}",
                                "real",
                            )
                        )
            # The rows adjudication dropped are unresolved; their truth is
            # unknown, so every assignment must be covered.
            if task == "intent":
                for N in SIZES:
                    pop = resample(rng, rows, N)
                    gone = np.array([r["adjudicated"] is None for r in pop])
                    colors = np.array(
                        [
                            sum(1 for r in pop if r["adjudicated"] is True),
                            sum(1 for r in pop if r["adjudicated"] is False),
                            0,
                            int(gone.sum()),
                        ]
                    )
                    known = colors[0] / N
                    truths = [
                        known,
                        known + gone.sum() / N,
                        known
                        + sum(r["original"] for r in pop if r["adjudicated"] is None)
                        / N,
                    ]
                    results.append(
                        plain_cell(
                            rng,
                            colors,
                            f"intent/{run}, {int(gone.sum())} dropped rows unresolved, N={N:,}",
                            "dropped",
                            truths=truths,
                        )
                    )

    for accuracy in (0.9, 0.97, 0.995):
        for N in SIZES:
            strata = []
            for share, acc in ((0.94, accuracy), (0.05, 0.70), (0.01, 0.50)):
                size = round(share * N)
                right = int(rng.binomial(size, acc))
                strata.append(np.array([right, size - right]))
            for split in ("equal", "weight"):
                label = f"accepted {accuracy:.1%} + repaired 5% at 70% + unverified 1% at 50%, N={N:,}, {split} split"
                results.append(strata_cell(rng, strata, label, split))

    for task in ("intent", "faithfulness"):
        s = shared[f"{task}/haiku, original"]
        for accuracy in (0.9, 0.97, 0.995):
            for disagree in (0.01, 0.03, 0.10):
                cells = by_cell(rng, 10_000, accuracy, disagree, s)
                if cells is None:
                    continue
                for people in (25, 50, 100):
                    label = (
                        f"{task} shared mistakes {s:.0%}, accuracy {accuracy:.1%}, "
                        f"disagreement {disagree:.0%}, {people} people labels, N=10,000"
                    )
                    results.append(two_tier_cell(rng, cells, people, label))

    worst = min(results, key=lambda r: r["coverage"])
    passed = worst["coverage"] >= PASS
    write_report(results, shared, worst, passed, time.time() - started)
    return 0 if passed else 1


def write_report(results, shared, worst, passed, seconds) -> None:
    lines = [
        "# Coverage on synthetic populations",
        "",
        f"{len(results)} cells × {DRAWS:,} draws, sample {SAMPLE}, seed {SEED}, "
        f"{seconds:.0f} s. Pass rule: coverage ≥ {PASS:.0%} in every cell "
        "(Monte Carlo band about ±1 point).",
        "",
        f"**{'PASS' if passed else 'FAIL'}**: lowest {worst['coverage']:.2%} — {worst['cell']}.",
        "",
        "Shared-mistake rate, P(Opus agrees with the run | run wrong): "
        + ", ".join(f"{k} {v:.0%}" for k, v in shared.items())
        + ". Two-tier cells use the original-gold rate for haiku; the"
        " adjudicated gold was settled partly by model votes, Opus's among them.",
        "",
    ]
    for family, title in (
        ("real", "Real rows: plain sample and unresolved rows"),
        ("dropped", "Rows with no defensible label, as unresolved"),
        ("strata", "Strata (stress population)"),
        ("two-tier", "Two-tier: model on the sample, people on part of it"),
    ):
        cells = [r for r in results if r["family"] == family]
        lines += ["", f"## {title}", ""]
        extra = {
            "two-tier": ("people_only", "people-only width"),
            "strata": ("plain", "unstratified width"),
        }.get(family)
        lines.append(
            "| cell | accuracy | coverage | mean width |"
            + (f" {extra[1]} |" if extra else "")
        )
        lines.append("|---|---:|---:|---:|" + ("---:|" if extra else ""))
        for r in cells:
            line = f"| {r['cell']} | {r['accuracy']:.1%} | {r['coverage']:.2%} | {r['width']:.4f} |"
            if extra:
                line += f" {r[extra[0]]:.4f} |"
            lines.append(line)
    out = HERE / "results" / "simulation.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]))
    print(f"wrote {out.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    sys.exit(main())
