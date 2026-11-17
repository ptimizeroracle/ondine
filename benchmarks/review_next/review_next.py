"""Review-next: does a ranking signal find the wrong rows better than per-value rates?

Pre-registered offline experiment (design: ``benchmarks/review_next/DESIGN.md``;
parents ``benchmarks/planner/DESIGN.md`` and ``DESIGN-addendum-1.md``, both
locked). The Ruler labels 100 rows. A person then reviews the top k of the other
delivered rows. Rankers order those rows without a label:

* R0 random (the null),
* R1 per-value error rate fitted on the 100 labelled rows (the comparator),
* R2 self-consistency: Haiku's dissent D over its 3 extra samples, then R1,
* R4 dissent plus the Haiku/Luna pair disagreement P, then R1.

The splits, the cluster bootstrap and the Bonferroni bound come from
``replay.py``; the sample files and their checks from ``addendum1.py``. Neither
is copied. Every constant lives in ``PREREG_REVIEW``; the results embed its hash.
Do not change a constant after seeing a result.

    uv run python benchmarks/review_next/review_next.py --check    # no ranker but R0
    uv run python benchmarks/review_next/review_next.py            # S = 1000, B = 1000

There is one full run, from the clean commit. ``--quick`` (B = 100) also uses
S = 1000 and prints the verdict, so it is refused on the real data until the
full result exists; use ``--out`` with a fixture to try it.

``--check`` runs the parents' checks, the hash and disclosure reproductions and
the split sizes; it evaluates no ranker except R0, whose lift is 1 by
construction. Exit code 2 means a failed check; the study then does not start.
The study writes ``review-next.md`` and ``review-next.json`` to
``benchmarks/review_next/results/`` (``results/quick/`` for ``--quick``). A full
run on a dirty tree, or over an existing result, exits 2.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PLANNER = HERE.parent / "planner"
sys.path.insert(0, str(PLANNER))
import addendum1  # noqa: E402
import replay  # noqa: E402

REPO = replay.REPO
DATA = replay.DATA
DESIGN = HERE / "DESIGN.md"
RESULTS = HERE / "results"

#: Every constant of the design, in one place. Its sha256 goes into the results.
PREREG_REVIEW: dict[str, Any] = {
    "design": "benchmarks/review_next/DESIGN.md",
    "design_sha256": "57e8962a0173c860dc464e2fb4cc2a456c3ece2f711c3209412f17265cfb7bff",
    "parents": {
        "replay": {
            "design_sha256": "3d667e65acd04d682855bf71d868e93b57e7aea53f2b782d4c68daaf051e4a42",
            "prereg_sha256": "01872888f24b359b5bb24ad167aa50e04d7ee41816718d7d93685e9e41bb0e3a",
        },
        "addendum1": {
            "design_sha256": "420032cc81b4d9af9ef17d4f77a820b3c78af0aa446053fbcea853b51abf4f33",
            "prereg_sha256": "28d2f6a99ae5e8959a7ea60c7e01fc59317727e204d02e84316bce8d44e73cef",
            "verdict": "KILL_STANDS",
        },
    },
    "tasks": ["intent", "faithfulness"],
    "delivered": "haiku",
    "partner": "luna",
    "sample_files": ["haiku-s1", "haiku-s2", "haiku-s3"],
    "data_sha256": {  # check: equality, else exit 2
        "intent/dev.jsonl": "f8d3b90ef99131fd438caf91876766fcd66c1a48b80f7f9f32abc8c0523e9fa0",
        "intent/dev-adjudicated.jsonl": "90ca4889002f5b2813bfe51036a512357f9f11071b0b4e1408f5dd167f033d1c",
        "labels/intent/haiku.jsonl": "160651b02b020a41d1c39a63aafeba50e2c45c72978c4affdbab4aa7fe42949b",
        "labels/intent/luna.jsonl": "80c373eeedd8315190643a2e3398652ff805f4e5b1f9eb50b3c9f5387521715c",
        "labels/intent/haiku-s1.jsonl": "cdc0314b67a036b74708b28ae7c7d5e0b6cd382361656897ff6de535eb476f75",
        "labels/intent/haiku-s2.jsonl": "8e81ebefcaac2afebcf3cd3fde09b2767f869fd2367de6a77c2ee31d46bff263",
        "labels/intent/haiku-s3.jsonl": "dcc67b66ec0029bf6c9961569b4ca88a286871b35cf433fd82eb1116bea26722",
        "intent/labels.json": "e64b29ebfc59f9bdbf4d4756fadd29669ab8fe623203b66790cce6d1e1b8f5f8",
        "labels/intent/haiku.cost.json": "9ae9937c37673d3e6747fd09fbf9df1f1c5f7dcf3fec69e47dff2c1735862b29",
        "labels/intent/luna.cost.json": "2afc06ac46c31a7d2046e3f5f35c8db131343a518bf5fd9e00ffa4dd7e7c92e1",
        "faithfulness/dev.jsonl": "d0fb8fd50518e1d3fb0a6319738611998d5ce3646e22c90aada8220bf4a8f821",
        "faithfulness/dev-adjudicated.jsonl": "0328b6c4fa5a9f35c59cf0f5aef3d0d34185c9e8a218ad7b3176b6f00d1d89e3",
        "labels/faithfulness/haiku.jsonl": "5c86b324538cecbe714213f03928594c635771aac9fafc83f985a97481b7ce40",
        "labels/faithfulness/luna.jsonl": "19719aa36f806d2dcce8a13172bb663f4394e8b32b47ccb60381cdb5bc2ad93f",
        "labels/faithfulness/haiku-s1.jsonl": "06b772cc801e274f0e66d3dc8359ed0e5531c293753ecb971c0373cad57d713c",
        "labels/faithfulness/haiku-s2.jsonl": "7f0ce6a98642f61230fbe0815833aa2ae1074ec11ef090c83088529d1420383f",
        "labels/faithfulness/haiku-s3.jsonl": "c76d0b86cf7db2abc6753e17479c6e2ada693b7f03d0826d148b8eaa7c778949",
        "labels/faithfulness/haiku.cost.json": "c271f1390fb86b19edcd12476ab4b71698604a3317f9385026ffa53fa5feebc5",
        "labels/faithfulness/luna.cost.json": "1027b7863bca45b5dea5141cec39329443fca700bb963b3fedfc6b2c10c656cf",
    },
    "n_labelled": 100,
    "splits": 1000,
    "split_purpose": 0,
    "bootstrap_purpose": 1,
    "bootstrap": 1000,
    "bootstrap_quick": 100,
    "bootstrap_splits_per_draw": 20,
    "shrinkage_m": 4,
    "k_fractions": ["0.05", "0.10", "0.20"],
    "primary_k": "0.10",
    "k_rounding": "half_up_exact",
    "null": "R0",
    "comparator": "R1",
    "confirmatory": ["R2", "R4"],
    # Key columns, compared lexicographically; "r1" is the float tie-break.
    "rankers": {"R0": [], "R1": ["r1"], "R2": ["D", "r1"], "R4": ["D+P", "r1"]},
    "exploratory": {
        "E1": ["P", "r1"],
        "E2": ["D"],
        "E3": ["D", "P", "r1"],
        "E4": ["D:s1,s2", "r1"],
        "E5": ["D:s1,s3", "r1"],
        "E6": ["D:s2,s3", "r1"],
        "E7": ["D:s1", "r1"],
        "E8": ["r1_all_rows"],
    },
    "max_exploratory_per_task": 8,
    "calls_per_row": {
        "R1": [],
        "R2": ["haiku"] * 3,
        "R3": ["luna"],
        "R4": ["haiku"] * 3 + ["luna"],
    },
    "review_cost_usd_per_row_assumed": [0.25, 0.01],  # break-even display only
    "alpha": 0.05,
    "n_tests": 4,  # LB quantile = 0.0125
    "pass": {
        "lift_mean_min": 2.0,
        "lift_lb_gt": 1.0,
        "gain_mean_min_pts": 10.0,
        "gain_lb_gt_pts": 0.0,
    },
    "kill": {"gain_mean_lt_pts": 5.0, "lift_mean_lt": 1.5},
    "select": {"r4_over_r2_min_pts": 5.0, "r4_over_r2_lb_gt_pts": 0.0},
    "flags": {
        "sign_flip_gain_le_pts": 0.0,
        "k_fragile_gain_le_pts": 0.0,
        "ambiguity_gap_min": 0.20,
        "subset_gain_min_pts": 10.0,
    },
    "wilson_z": 1.959964,
    "gold_for_verdict": "original",
    "disclosure": {
        "intent": {
            "rows": 231,
            "wrong": 44,
            "test_rows": 131,
            "dissent": {0: 211, 1: 11, 2: 5, 3: 4},
            "pair": 22,
            "dissent_and_pair": 11,
            "distinct_values": 73,
            "flag_wrong": 11,
            "flag_right": 9,
            "adj": {
                "rows": 218,
                "wrong": 23,
                "test_rows": 118,
                "dissent": {0: 201, 1: 10, 2: 4, 3: 3},
                "flag_wrong": 9,
                "flag_right": 8,
            },
        },
        "faithfulness": {
            "rows": 200,
            "wrong": 36,
            "test_rows": 100,
            "dissent": {0: 181, 1: 5, 2: 7, 3: 7},
            "pair": 35,
            "dissent_and_pair": 12,
            "distinct_values": 2,
            "value_counts": [96, 104],
            "flag_wrong": 11,
            "flag_right": 8,
            "adj": {
                "rows": 198,
                "wrong": 13,
                "test_rows": 98,
                "dissent": {0: 179, 1: 5, 2: 7, 3: 7},
                "flag_wrong": 10,
                "flag_right": 9,
            },
        },
    },
}

#: The exploratory rankers whose gain the SUBSET flag reads (R2 on two samples).
SUBSET_RANKERS = ("E4", "E5", "E6")
#: The paired draw key: R4 minus R2.
PAIR = "R4-R2"
#: The choice between two passing rankers (DESIGN section 5), printed with the verdict.
SELECTION_RULE = (
    "R2 unless R4 beats R2 by >= 5 points with paired LB > 0 on every task in scope"
)


def _assert_prereg_matches_parent() -> None:
    """Fail at import if the parent's constants moved under this pre-registration."""
    p, parent = PREREG_REVIEW, replay.PREREG
    if parent["seed"] != 20261117:
        raise RuntimeError("the parent's seed is no longer 20261117")
    if list(replay.TASKS) != parent["tasks"]:
        raise RuntimeError("the parent's task order changed; the split streams differ")
    for key in (
        "splits",
        "n_labelled",
        "bootstrap",
        "bootstrap_quick",
        "bootstrap_splits_per_draw",
        "shrinkage_m",
        "alpha",
    ):
        if p[key] != parent[key]:
            raise RuntimeError(f"PREREG_REVIEW[{key!r}] differs from the parent's")
    if len(p["exploratory"]) > p["max_exploratory_per_task"]:
        raise RuntimeError("more exploratory rankers than the design allows")
    if p["n_tests"] != len(p["confirmatory"]) * len(p["tasks"]):
        raise RuntimeError("n_tests is not rankers x tasks")
    if not {*p["confirmatory"], p["null"], p["comparator"]} <= set(p["rankers"]):
        raise RuntimeError("a named ranker has no key spec")
    if not set(SUBSET_RANKERS) <= set(p["exploratory"]):
        raise RuntimeError("a SUBSET ranker is not an exploratory ranker")
    if set(p["rankers"]) & set(p["exploratory"]):
        raise RuntimeError("a ranker name is used twice")


_assert_prereg_matches_parent()


def effective_prereg(prereg: dict = PREREG_REVIEW) -> dict:
    """The parent's PREREG with this design's family of 4 (LB quantile 0.0125)."""
    return {
        **replay.PREREG,
        **{
            k: prereg[k]
            for k in (
                "splits",
                "n_labelled",
                "bootstrap",
                "bootstrap_quick",
                "bootstrap_splits_per_draw",
                "shrinkage_m",
                "alpha",
            )
        },
        "n_confirmatory": prereg["n_tests"],
    }


def all_rankers(prereg: dict = PREREG_REVIEW) -> dict[str, list[str]]:
    """Confirmatory, null, comparator and exploratory key specs by name."""
    return {**prereg["rankers"], **prereg["exploratory"]}


# ── k and the expected hits ──────────────────────────────────────────────


def k_at(frac: str, w_total: Any) -> Any:
    """k = floor(f * W + 1/2) in exact fractions, clipped to [1, W].

    `w_total` is an int or an integer array. Python's ``round`` goes half to
    even, which would give 6 for f = 0.05, W = 130; the design says 7.
    """
    f = Fraction(frac)
    w = np.asarray(w_total, dtype=np.int64)
    k = np.clip((2 * f.numerator * w + f.denominator) // (2 * f.denominator), 1, w)
    return int(k) if k.ndim == 0 else k


def _lex_order(keys: np.ndarray) -> np.ndarray:
    """(S, n) row order per split, highest key first, columns compared in turn.

    Stable sorts from the last column to the first give the lexicographic
    order for all splits at once. Order inside a tie is arbitrary and unused:
    ties are scored as a group, never by position.
    """
    s, n, c = keys.shape
    order = np.broadcast_to(np.arange(n), (s, n)).copy()
    for j in range(c - 1, -1, -1):
        col = np.take_along_axis(keys[:, :, j], order, axis=1)
        step = np.argsort(-col, axis=1, kind="stable")
        order = np.take_along_axis(order, step, axis=1)
    return order


@dataclass
class _Groups:
    """Per position in key order: weight above the group, group weight and errors."""

    first: np.ndarray
    above: np.ndarray
    weight: np.ndarray
    errors: np.ndarray


def _groups(keys: np.ndarray, hit: np.ndarray, w: np.ndarray) -> _Groups:
    """Group rows of equal key (all columns) and total each group's weight and hits."""
    s, n, _ = keys.shape
    order = _lex_order(keys)
    sorted_keys = np.take_along_axis(keys, order[:, :, None], axis=1)
    first = np.ones((s, n), bool)
    first[:, 1:] = (sorted_keys[:, 1:] != sorted_keys[:, :-1]).any(-1)
    last = np.ones((s, n), bool)
    last[:, :-1] = first[:, 1:]
    ws = np.take_along_axis(w, order, axis=1)
    es = ws * hit[order]
    cum_w, cum_e = np.cumsum(ws, 1), np.cumsum(es, 1)
    pos = np.arange(n)
    start = np.maximum.accumulate(np.where(first, pos, 0), axis=1)
    end = np.minimum.accumulate(np.where(last, pos, n - 1)[:, ::-1], axis=1)[:, ::-1]
    above = np.take_along_axis(cum_w - ws, start, axis=1)
    below_e = np.take_along_axis(cum_e - es, start, axis=1)
    return _Groups(
        first,
        above,
        np.take_along_axis(cum_w, end, axis=1) - above,
        np.take_along_axis(cum_e, end, axis=1) - below_e,
    )


def _hits_at(g: _Groups, k: np.ndarray) -> np.ndarray:
    """Expected hits in the first k copies, ties broken uniformly at random.

    A group fully above the cut counts all its hits; the boundary group
    (weight w_g, hits e_g) adds (k - W_above) * e_g / w_g.
    """
    kk = np.asarray(k, float)[:, None]
    full = g.above + g.weight <= kk
    part = (g.above < kk) & ~full
    safe = np.where(g.weight > 0, g.weight, 1.0)
    share = (kk - g.above) * g.errors / safe
    return np.asarray(
        (np.where(full, g.errors, np.where(part, share, 0.0)) * g.first).sum(1)
    )


def expected_hits_batch(
    keys: np.ndarray, hit: np.ndarray, w: np.ndarray, k: np.ndarray
) -> np.ndarray:
    """`expected_hits` for S splits at once: keys (S, n, c), w (S, n), k (S,)."""
    return _hits_at(_groups(keys, np.asarray(hit, bool), w), k)


def expected_hits(keys: np.ndarray, hit: np.ndarray, w: np.ndarray, k: int) -> float:
    """Expected hits in the top k of one split; keys (n, c), hit (n,), w (n,).

    A bootstrap weight w counts as w copies of the row; a row of weight 0 does
    not exist. Rows of equal key are one group and share the cut at random.
    """
    got = expected_hits_batch(
        keys[None], hit, np.asarray(w, float)[None], np.array([k])
    )
    return float(got[0])


# ── the signals and the rankers ──────────────────────────────────────────


def dissent(samples: addendum1.SampleSet, cols: list[int] | None = None) -> np.ndarray:
    """(n,) how many of the extra samples `cols` differ from the delivered answer.

    Column 0 is the delivered run. A null answer is the string "\x00null:{id}"
    and a missing id gets a negative code (`build_sample_set`); both differ from
    any real answer, so both count as dissent. The delivered answer is the
    reference, not the mode.
    """
    cols = list(range(1, samples.n_columns)) if cols is None else cols
    return np.asarray((samples.codes[:, cols] != samples.codes[:, :1]).sum(1))


def pair_disagree(
    task: replay.Task, delivered: str = "haiku", partner: str = "luna"
) -> np.ndarray:
    """(n,) 1 where the pair does not agree on a valid value (the parent's C1 signal)."""
    return np.asarray(~addendum1._pair_agrees(task, delivered, partner)[:, 0], int)


def r1_score(
    task: replay.Task,
    train_w: np.ndarray,
    m: float = PREREG_REVIEW["shrinkage_m"],
    delivered: str = "haiku",
) -> np.ndarray:
    """(S, n) 1 - p-hat of the delivered value, fitted on the labelled rows only.

    An unseen value gets the overall precision p-bar (`replay._fit_precision`).
    """
    table, _ = replay._fit_precision(task, delivered, train_w, m)
    return np.asarray(1.0 - table[:, task.cell[delivered][:, 0]])


def _column(
    col: str,
    task: replay.Task,
    samples: addendum1.SampleSet,
    train_w: np.ndarray,
    prereg: dict,
) -> np.ndarray:
    """One key column (S, n) of a ranker spec."""
    delivered, partner = prereg["delivered"], prereg["partner"]
    m = prereg["shrinkage_m"]
    if col == "r1":
        return r1_score(task, train_w, m, delivered)
    if col == "r1_all_rows":  # unreachable: fitted on every row's label
        return r1_score(task, np.ones_like(train_w), m, delivered)
    pair = pair_disagree(task, delivered, partner)
    if col == "P":
        per_row = pair
    elif col == "D+P":
        per_row = dissent(samples) + pair
    elif col == "D":
        per_row = dissent(samples)
    elif col.startswith("D:s"):
        per_row = dissent(
            samples, [int(x) for x in col[3:].replace("s", "").split(",")]
        )
    else:
        raise ValueError(f"unknown key column {col!r}")
    return np.broadcast_to(per_row, train_w.shape).astype(float)


def ranker_keys(
    name: str,
    task: replay.Task,
    samples: addendum1.SampleSet,
    train_w: np.ndarray,
    prereg: dict = PREREG_REVIEW,
) -> np.ndarray:
    """(S, n, c) lexicographic keys of ranker `name`; a higher key is reviewed first.

    The D and P tiers are integers held exactly in float64. The float r1 is only
    ever the last column, so no summed float lets a tier cross another. R0 has
    one constant column: every row ties.
    """
    addendum1._aligned(task, samples)
    spec = all_rankers(prereg)[name]
    if not spec:
        return np.zeros((*train_w.shape, 1))
    return np.stack([_column(t, task, samples, train_w, prereg) for t in spec], axis=-1)


# ── evaluation ───────────────────────────────────────────────────────────


def evaluate(
    task: replay.Task,
    samples: addendum1.SampleSet,
    train_w: np.ndarray,
    test_w: np.ndarray,
    names: list[str],
    fracs: list[str],
    prereg: dict = PREREG_REVIEW,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Per split, ranker and k: precision P, base rate b, recall and k, each (S,).

    A ranker reads the labels of the train rows and orders every row; the
    score counts the delivered answers that are wrong among the top k of the
    test rows (weight `test_w`).
    """
    wrong = ~task.correct[prereg["delivered"]][:, 0]
    w_total = test_w.sum(1)
    errors = test_w @ wrong
    base = errors / w_total
    ks = {f: k_at(f, np.rint(w_total).astype(np.int64)) for f in fracs}
    out: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for name in names:
        groups = _groups(
            ranker_keys(name, task, samples, train_w, prereg), wrong, test_w
        )
        out[name] = {}
        for f in fracs:
            hits = _hits_at(groups, ks[f])
            recall = np.divide(
                hits, errors, out=np.full(hits.shape, np.nan), where=errors > 0
            )
            out[name][f] = {"P": hits / ks[f], "b": base, "recall": recall, "k": ks[f]}
    return out


def r0_deviation(
    task: replay.Task,
    test_w: np.ndarray,
    fracs: list[str],
    prereg: dict = PREREG_REVIEW,
) -> float:
    """Max |L - 1| of the null ranker R0 over all splits and k fractions.

    R0's keys all tie, so its expected precision is the base rate and its lift
    is 1 whatever the labels are: a check of the tie handling that reads no
    outcome. It runs on the group scoring directly, not on `evaluate`.
    """
    wrong = ~task.correct[prereg["delivered"]][:, 0]
    w_total = test_w.sum(1)
    base = (test_w @ wrong) / w_total
    keys = np.zeros((*test_w.shape, 1))
    worst = 0.0
    for f in fracs:
        k = k_at(f, np.rint(w_total).astype(np.int64))
        lift = expected_hits_batch(keys, wrong, test_w, k) / k
        ok = base > 0
        if ok.any():
            worst = max(worst, float(np.abs(lift[ok] / base[ok] - 1.0).max()))
    return worst


def _group_mean(x: np.ndarray, groups: int, skip_nan: bool = False) -> np.ndarray:
    """Mean of consecutive blocks: (groups * m,) -> (groups,)."""
    r = np.asarray(x, float).reshape(groups, -1)
    if not skip_nan:
        return r.mean(1)
    ok = np.isfinite(r)
    return np.asarray(
        np.divide(
            np.where(ok, r, 0.0).sum(1),
            ok.sum(1),
            out=np.full(groups, np.nan),
            where=ok.any(1),
        )
    )


def reduce_splits(
    ev: dict, groups: int = 1, comparator: str = "R1"
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Mean precision, base rate, lift and gain over each block of splits.

    ``groups = 1`` is the point estimate over all S splits; ``groups = B`` takes
    the 20 splits of each bootstrap draw. L is the ratio of the means, not the
    mean of the ratios; G = 100 (P-bar_R - P-bar_comparator), in points.
    """
    out: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for name, by_frac in ev.items():
        out[name] = {}
        for f, m in by_frac.items():
            p, b = _group_mean(m["P"], groups), _group_mean(m["b"], groups)
            out[name][f] = {
                "P": p,
                "b": b,
                "L": np.divide(p, b, out=np.full(p.shape, np.nan), where=b > 0),
                "recall": _group_mean(m["recall"], groups, skip_nan=True),
            }
    if comparator in out:
        for by_frac in out.values():
            for f, r in by_frac.items():
                r["G"] = 100.0 * (r["P"] - out[comparator][f]["P"])
    return out


def aggregate(ev: dict, comparator: str = "R1") -> dict[str, dict[str, dict]]:
    """name -> k fraction -> {P, b, L, G, recall, per100, rows_per_error} over the splits."""
    out: dict[str, dict[str, dict]] = {}
    for name, by_frac in reduce_splits(ev, 1, comparator).items():
        out[name] = {}
        for f, r in by_frac.items():
            p = float(r["P"][0])
            out[name][f] = {
                "P": p,
                "b": float(r["b"][0]),
                "L": replay._clean(r["L"][0]),
                "G": replay._clean(r["G"][0]) if "G" in r else None,
                "recall": replay._clean(r["recall"][0]),
                "per100": 100.0 * p,
                "rows_per_error": 1.0 / p if p > 0 else None,
            }
    return out


# ── bootstrap ────────────────────────────────────────────────────────────


def lower_bound(draws: np.ndarray, prereg: dict = PREREG_REVIEW) -> float:
    """One-sided lower bound at alpha / 4 = 0.0125 (4 ranker-by-task tests)."""
    return float(replay.lower_bound(draws, effective_prereg(prereg)))


def bootstrap(
    task: replay.Task,
    samples: addendum1.SampleSet,
    n_draws: int,
    prereg: dict = PREREG_REVIEW,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Draw-level lift L and gain G of R2 and R4, and G of R4 minus R2, per k.

    The parent's cluster bootstrap on its stream (purpose 1): per draw, 20
    splits of the resampled ids. Every ranker sees the same weights, so G is
    paired: a ranker that equals R1 has G = 0 in every draw.
    """
    eff = effective_prereg(prereg)
    n_l, per_draw = prereg["n_labelled"], prereg["bootstrap_splits_per_draw"]
    fracs, rankers = prereg["k_fractions"], prereg["confirmatory"]
    comparator = prereg["comparator"]
    names = [comparator, *rankers]
    out: dict[str, dict[str, dict[str, list]]] = {
        r: {f: {"L": [], "G": []} for f in fracs} for r in rankers
    }
    out[PAIR] = {f: {"G": []} for f in fracs}
    rng = replay.split_rng(eff, task.name, "original", prereg["bootstrap_purpose"], n_l)
    for train_w, test_w, d in replay.bootstrap_batches(
        rng, task.n, n_l, n_draws, per_draw, chunk=1000
    ):
        red = reduce_splits(
            evaluate(task, samples, train_w, test_w, names, fracs, prereg),
            d,
            comparator,
        )
        for f in fracs:
            for r in rankers:
                out[r][f]["L"].append(red[r][f]["L"])
                out[r][f]["G"].append(red[r][f]["G"])
            out[PAIR][f]["G"].append(
                100.0 * (red[rankers[1]][f]["P"] - red[rankers[0]][f]["P"])
            )
    return {
        r: {
            f: {q: np.concatenate(v) for q, v in by_q.items()}
            for f, by_q in by_f.items()
        }
        for r, by_f in out.items()
    }


def draw_summary(draws: np.ndarray, prereg: dict = PREREG_REVIEW) -> dict:
    """Mean, lower bound at q = 0.0125, standard error and the number of draws."""
    return {
        "mean": replay._clean(draws.mean()),
        "lb": replay._clean(lower_bound(draws, prereg)),
        "se": replay._clean(draws.std(ddof=1)) if len(draws) > 1 else None,
        "draws": int(len(draws)),
    }


# ── verdict ──────────────────────────────────────────────────────────────


def task_status(entry: dict, prereg: dict = PREREG_REVIEW) -> str:
    """PASS, KILL or INCONCLUSIVE for one ranker on one task (design section 5).

    `entry` holds the mean lift L and gain G (points) at the primary k, their
    lower bounds L_lb and G_lb, and v_failed (V1-V3 failed on the task).
    """
    if entry.get("v_failed") or any(
        entry.get(x) is None for x in ("L", "L_lb", "G", "G_lb")
    ):
        return "INCONCLUSIVE"
    p, k = prereg["pass"], prereg["kill"]
    if (
        entry["L"] >= p["lift_mean_min"]
        and entry["L_lb"] > p["lift_lb_gt"]
        and entry["G"] >= p["gain_mean_min_pts"]
        and entry["G_lb"] > p["gain_lb_gt_pts"]
    ):
        return "PASS"
    if entry["G"] < k["gain_mean_lt_pts"] or entry["L"] < k["lift_mean_lt"]:
        return "KILL"
    return "INCONCLUSIVE"


def flags(entry: dict, prereg: dict = PREREG_REVIEW) -> list[str]:
    """Flags reported on a passing ranker-task pair; none of them kills.

    `entry` adds: G_adjudicated, G_by_frac (mean G per k fraction, original
    gold), ambiguity (section 6) and G_subsets (mean G of E4-E6).
    """
    f, out = prereg["flags"], []
    adj = entry.get("G_adjudicated")
    if adj is not None and adj <= f["sign_flip_gain_le_pts"]:
        out.append("SIGN_FLIP")
    others = [x for x in prereg["k_fractions"] if x != prereg["primary_k"]]
    by_frac = entry.get("G_by_frac", {})
    if any(
        by_frac.get(x) is not None and by_frac[x] <= f["k_fragile_gain_le_pts"]
        for x in others
    ):
        out.append("K_FRAGILE")
    if entry.get("ambiguity"):
        out.append("AMBIGUITY")
    if any(
        g is not None and g < f["subset_gain_min_pts"]
        for g in entry.get("G_subsets", {}).values()
    ):
        out.append("SUBSET")
    return out


def r4_beats_r2(pair_gain: float | None, pair_lb: float | None, prereg: dict) -> bool:
    """R4 - R2 >= 5 points with a paired lower bound above 0 (design section 5)."""
    s = prereg["select"]
    return (
        pair_gain is not None
        and pair_lb is not None
        and pair_gain >= s["r4_over_r2_min_pts"]
        and pair_lb > s["r4_over_r2_lb_gt_pts"]
    )


def verdict(
    statuses: dict[str, dict[str, str]],
    flag_lists: dict[str, dict[str, list[str]]],
    v_failed: set[str] | list[str],
    beats: dict[str, bool] | None = None,
    prereg: dict = PREREG_REVIEW,
) -> dict:
    """The overall decision (design section 5), from {ranker: {task: status}}.

    A V1-V3 failure on a task makes every ranker INCONCLUSIVE on it, even when
    the numbers pass. BUILD_GENERAL(R): R passes on every task. BUILD_SCOPED:
    a ranker passes on one task only. KILL: every ranker-task pair is KILL.
    Between R2 and R4 the simpler R2 wins unless `beats[task]` holds on every
    task in scope. `beats` is {task: R4 - R2 >= 5 points with a paired LB > 0}.
    """
    tasks, rankers = prereg["tasks"], prereg["confirmatory"]
    simple, rich = rankers[0], rankers[1]
    beats = beats or {}
    eff = {
        r: {t: "INCONCLUSIVE" if t in v_failed else statuses[r][t] for t in tasks}
        for r in rankers
    }
    passing = {r: [t for t in tasks if eff[r][t] == "PASS"] for r in rankers}

    def choose(candidates: list[str], scope: list[str]) -> str:
        if rich in candidates and simple in candidates:
            return str(rich if all(beats.get(t) for t in scope) else simple)
        return candidates[0]

    builds: list[dict] = []
    general = [r for r in rankers if len(passing[r]) == len(tasks)]
    if general:
        kind = "BUILD_GENERAL"
        builds = [{"ranker": choose(general, tasks), "tasks": list(tasks)}]
        name = f"BUILD_GENERAL({builds[0]['ranker']})"
    elif any(passing.values()):
        kind = "BUILD_SCOPED"
        for t in tasks:
            cands = [r for r in rankers if passing[r] == [t]]
            if cands:
                builds.append({"ranker": choose(cands, [t]), "tasks": [t]})
        name = " + ".join(
            f"BUILD_SCOPED({b['tasks'][0]}, {b['ranker']})" for b in builds
        )
    elif all(s == "KILL" for r in rankers for s in eff[r].values()):
        kind = name = "KILL"
    else:
        kind = name = "INCONCLUSIVE"
    reported = {
        f"{b['ranker']}/{t}": flag_lists.get(b["ranker"], {}).get(t, [])
        for b in builds
        for t in b["tasks"]
    }
    return {
        "verdict": name,
        "kind": kind,
        "builds": builds,
        "selection_rule": SELECTION_RULE,
        "statuses": eff,
        "flags": reported,
        "v_failed": sorted(v_failed),
    }


# ── mechanism and label-noise control ────────────────────────────────────


def noise_control(
    original: replay.Task,
    adjudicated: replay.Task,
    d_by_id: dict[str, int],
    prereg: dict = PREREG_REVIEW,
) -> dict:
    """Section 6: the flag (D >= 1) on Haiku's wrong rows, split by adjudication.

    A: still wrong. F: now right. X: dropped. N: right on the original gold,
    wrong on the adjudicated one. Wilson 95% intervals. AMBIGUITY if
    flag(F u X) - flag(A) >= 0.20: the flag finds disputed labels, not errors.
    """
    z, delivered = prereg["wilson_z"], prereg["delivered"]
    flag = np.array([d_by_id[i] >= 1 for i in original.ids], bool)
    wrong = ~original.correct[delivered][:, 0]
    position = {rid: k for k, rid in enumerate(adjudicated.ids)}
    right_now = adjudicated.correct[delivered][:, 0]
    group = {g: np.zeros(original.n, bool) for g in "AFXN"}
    for i, rid in enumerate(original.ids):
        k = position.get(rid)
        if wrong[i]:
            group["X" if k is None else "F" if right_now[k] else "A"][i] = True
        elif k is not None and not right_now[k]:
            group["N"][i] = True
    group["FX"] = group["F"] | group["X"]
    rates = {g: addendum1._rate(flag, rows, z) for g, rows in group.items()}
    rates["wrong"] = addendum1._rate(flag, wrong, z)
    rates["right"] = addendum1._rate(flag, ~wrong, z)
    a, fx = rates["A"], rates["FX"]
    gap, ambiguity = None, False
    if a["n"] and fx["n"]:
        exact = Fraction(fx["flagged"], fx["n"]) - Fraction(a["flagged"], a["n"])
        ambiguity = exact >= Fraction(str(prereg["flags"]["ambiguity_gap_min"]))
        gap = float(exact)
    return {"groups": rates, "gap_fx_minus_a": gap, "ambiguity": ambiguity}


# ── disclosure ───────────────────────────────────────────────────────────


def _block(
    task: replay.Task, samples: addendum1.SampleSet, prereg: dict
) -> dict[str, Any]:
    """The counts of one gold that the design's section 0 states."""
    d = dissent(samples)
    wrong = ~task.correct[prereg["delivered"]][:, 0]
    return {
        "rows": task.n,
        "wrong": int(wrong.sum()),
        "test_rows": task.n - prereg["n_labelled"],
        "dissent": {j: int((d == j).sum()) for j in range(samples.n_columns)},
        "flag_wrong": int(((d >= 1) & wrong).sum()),
        "flag_right": int(((d >= 1) & ~wrong).sum()),
    }


def disclosure_counts(
    original: tuple[replay.Task, addendum1.SampleSet],
    adjudicated: tuple[replay.Task, addendum1.SampleSet],
    prereg: dict = PREREG_REVIEW,
) -> dict[str, Any]:
    """Recompute the existing-label counts of design section 0 for one task."""
    task, samples = original
    pair = pair_disagree(task, prereg["delivered"], prereg["partner"])
    values, counts = np.unique(
        task.value[prereg["delivered"]][:, 0], return_counts=True
    )
    out = _block(task, samples, prereg)
    out.update(
        pair=int(pair.sum()),
        dissent_and_pair=int(((dissent(samples) >= 1) & (pair == 1)).sum()),
        distinct_values=int(len(values)),
        value_counts=sorted(int(c) for c in counts),
        adj=_block(*adjudicated, prereg),
    )
    return out


def diff_disclosure(got: dict, expected: dict) -> dict[str, tuple]:
    """{key: (got, expected)} for every expected count the data does not reproduce."""
    bad: dict[str, tuple] = {}
    for key, want in expected.items():
        if isinstance(want, dict) and key == "adj":
            bad.update(
                {f"adj.{k}": v for k, v in diff_disclosure(got[key], want).items()}
            )
        elif got.get(key) != want:
            bad[key] = (got.get(key), want)
    return bad


# ── checks ───────────────────────────────────────────────────────────────

TaskSet = dict[str, dict[str, tuple[replay.Task, addendum1.SampleSet]]]


@dataclass
class Checks:
    """What `--check` found. ``problems`` stop the run; ``v123`` make a task INCONCLUSIVE."""

    lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    v123: dict[str, list[str]] = field(default_factory=dict)
    sample: addendum1.Checks | None = None
    data_sha256: dict[str, str] = field(default_factory=dict)
    disclosure: dict[str, dict] = field(default_factory=dict)
    splits: dict[str, dict] = field(default_factory=dict)
    tasks: TaskSet = field(default_factory=dict, repr=False)

    @property
    def hard_ok(self) -> bool:
        return not self.problems

    def say(self, text: str) -> None:
        self.lines.append(text)

    def fail(self, text: str) -> None:
        self.problems.append(text)
        self.lines.append(f"FAIL: {text}")


def load_tasks(data_dir: Path = DATA, prereg: dict = PREREG_REVIEW) -> TaskSet:
    """Each task on each gold, with the delivered and extra Haiku answers aligned."""
    return {
        name: {
            g: (
                replay.load_task(name, g, data_dir),
                addendum1.load_samples(name, g, prereg["sample_files"], data_dir),
            )
            for g in replay.GOLDS
        }
        for name in prereg["tasks"]
    }


def _check_parents(rep: Checks, prereg: dict) -> None:
    parents = prereg["parents"]
    addendum_design = addendum1.ADDENDUM_DESIGN
    got = {
        "replay design": (replay.design_hash(), parents["replay"]["design_sha256"]),
        "replay PREREG": (replay.prereg_hash(), parents["replay"]["prereg_sha256"]),
        "addendum 1 design": (
            addendum1.sha256_file(addendum_design)
            if addendum_design.exists()
            else None,
            parents["addendum1"]["design_sha256"],
        ),
        "addendum 1 PREREG": (
            addendum1.prereg_addendum_hash(),
            parents["addendum1"]["prereg_sha256"],
        ),
    }
    bad = [k for k, (have, want) in got.items() if have != want]
    for k in bad:
        rep.fail(
            f"the parent's {k} hash is not the one this design was written against"
        )
    if not bad:
        rep.say("parent designs and PREREG hashes (replay, addendum 1): OK")


def _check_data_hashes(rep: Checks, data_dir: Path, prereg: dict) -> None:
    bad = 0
    for rel, want in prereg["data_sha256"].items():
        try:
            have = addendum1.sha256_file(replay.guard_path(data_dir / rel, data_dir))
        except (FileNotFoundError, PermissionError):
            rep.fail(f"{rel}: missing or not readable")
            bad += 1
            continue
        rep.data_sha256[rel] = have
        if have != want:
            rep.fail(f"{rel}: sha256 {have[:12]} differs from the PREREG {want[:12]}")
            bad += 1
    if not bad:
        rep.say(f"data files: {len(prereg['data_sha256'])} sha256 values match")


def _check_samples(
    rep: Checks, data_dir: Path, prereg: dict, sample_args: dict
) -> None:
    sample = addendum1.sample_checks(data_dir, **sample_args)
    rep.sample = sample
    rep.v123 = {t: list(v) for t, v in sample.v123.items() if v}
    rep.lines.extend(sample.lines)
    for problem in sample.problems:
        rep.problems.append(f"sample checks: {problem}")
    for task in prereg["tasks"]:
        used = sample.selected.get(task)
        if used is not None and used != prereg["sample_files"]:
            rep.fail(f"{task}: the files used {used} are not the PREREG's sample files")
    for task in prereg["tasks"]:
        failed = rep.v123.get(task)
        rep.say(
            f"[{task}] V1-V3: "
            + ("FAILED (R2 and R4 INCONCLUSIVE)" if failed else "ok")
        )


def _check_disclosure(rep: Checks, tasks: TaskSet, prereg: dict) -> None:
    for name, expected in prereg["disclosure"].items():
        if name not in tasks:
            continue
        got = disclosure_counts(
            tasks[name]["original"], tasks[name]["adjudicated"], prereg
        )
        rep.disclosure[name] = got
        bad = diff_disclosure(got, expected)
        if bad:
            rep.fail(
                f"{name}: existing labels differ from the disclosure (got, expected): {bad}"
            )
        else:
            rep.say(f"[{name}] the design's disclosed counts reproduce")


def _check_splits(rep: Checks, tasks: TaskSet, prereg: dict) -> None:
    """Test rows and k per gold, R0's lift, and no invalid delivered value."""
    eff, n_l, fracs = (
        effective_prereg(prereg),
        prereg["n_labelled"],
        prereg["k_fractions"],
    )
    for name, golds in tasks.items():
        expected = prereg["disclosure"].get(name, {})
        rep.splits[name] = {}
        for gold, (task, samples) in golds.items():
            if task.same_as_original:
                continue
            train_w, test_w, _ = replay._SplitCache(task, eff).get(n_l)
            sizes = np.unique(test_w.sum(1))
            want = (expected if gold == "original" else expected.get("adj", {})).get(
                "test_rows"
            )
            ks = [k_at(f, int(sizes[0])) for f in fracs]
            worst = r0_deviation(task, test_w, fracs, prereg)
            rep.splits[name][gold] = {
                "test_rows": int(sizes[0]),
                "k": ks,
                "r0_max_dev": worst,
            }
            rep.say(
                f"[{name}/{gold}] test rows per split {sizes.tolist()}, k at "
                f"{'/'.join(f'{float(Fraction(f)) * 100:g}%' for f in fracs)} = "
                f"{'/'.join(map(str, ks))}; R0 lift 1 to within {worst:.1e} on {len(test_w)} splits"
            )
            if len(sizes) != 1 or (want is not None and int(sizes[0]) != want):
                rep.fail(f"{name}/{gold}: test rows {sizes.tolist()}, expected {want}")
            if worst > 1e-12:
                rep.fail(
                    f"{name}/{gold}: R0's lift is not 1 (max deviation {worst:.2e})"
                )
            invalid = int((~task.valid[prereg["delivered"]]).sum())
            if invalid:
                rep.fail(f"{name}/{gold}: {invalid} invalid delivered values")


def check(
    data_dir: Path = DATA,
    prereg: dict = PREREG_REVIEW,
    *,
    integrity_prereg: dict | None = None,
    sample_args: dict | None = None,
    design_path: Path = DESIGN,
) -> Checks:
    """Every check the design lists. It evaluates no ranker except R0.

    `integrity_prereg`, `sample_args` (keyword arguments of
    ``addendum1.sample_checks``) and `design_path` let a test point the checks
    at a fixture.
    """
    rep = Checks()
    rep.say(f"PREREG_REVIEW sha256 {replay.prereg_hash(prereg)}")
    design_sha = addendum1.sha256_file(design_path) if design_path.exists() else None
    rep.say(f"review-next design sha256 {design_sha or 'MISSING'}")
    if design_sha is None:
        rep.fail(f"design file {design_path} not found")
    elif design_sha != prereg["design_sha256"]:
        rep.fail(
            f"the design file hash {design_sha[:12]} differs from the PREREG "
            f"{prereg['design_sha256'][:12]}: the design changed after it was locked"
        )
    _check_parents(rep, prereg)
    integrity = replay.check_integrity(data_dir, integrity_prereg or replay.PREREG)
    for problem in integrity.problems:
        rep.problems.append(f"parent integrity: {problem}")
    rep.say(
        "parent integrity (replay.check_integrity): "
        + ("OK" if integrity.ok else f"FAILED ({len(integrity.problems)} problems)")
    )
    if not integrity.ok:
        return rep
    _check_samples(rep, data_dir, prereg, sample_args or {})
    _check_data_hashes(rep, data_dir, prereg)
    try:
        rep.tasks = load_tasks(data_dir, prereg)
    except FileNotFoundError as e:
        rep.fail(f"missing file {e.filename}")
        return rep
    _check_disclosure(rep, rep.tasks, prereg)
    _check_splits(rep, rep.tasks, prereg)
    leaked = replay.guard_selftest(data_dir)
    if leaked:
        rep.fail(f"the loader guard let {leaked} through")
    else:
        rep.say("loader guard: held-out and unlisted files refused: OK")
    return rep


# ── the study ────────────────────────────────────────────────────────────


def unit_costs(task: replay.Task, prereg: dict = PREREG_REVIEW) -> dict[str, float]:
    """$ per 1k rows of each ranker, from the measured per-call cost of its models."""
    return {
        r: 1000.0 * sum(task.u_flat[m] for m in calls)
        for r, calls in prereg["calls_per_row"].items()
    }


def break_even(
    task: replay.Task, aggregated: dict, prereg: dict = PREREG_REVIEW
) -> dict[str, dict]:
    """P_R / P_R1 that pays for a ranker's cost c per row on every row (display only).

    With review budget f, c pays if c / f < h (P_R / P_R1 - 1), h the review cost.
    """
    f = Fraction(prereg["primary_k"])
    costs = unit_costs(task, prereg)
    base = aggregated[prereg["comparator"]][prereg["primary_k"]]["P"]
    out = {}
    for r in prereg["confirmatory"]:
        c_row = costs[r] / 1000.0
        out[r] = {
            "usd_per_1k": costs[r],
            "observed_ratio": (
                aggregated[r][prereg["primary_k"]]["P"] / base if base > 0 else None
            ),
            "needed_ratio": {
                str(h): 1.0 + c_row / float(f) / h
                for h in prereg["review_cost_usd_per_row_assumed"]
            },
        }
    return out


def _gold_result(task: replay.Task, samples: addendum1.SampleSet, prereg: dict) -> dict:
    """Point estimates over the S splits of one gold, for every ranker."""
    eff = effective_prereg(prereg)
    train_w, test_w, _ = replay._SplitCache(task, eff).get(prereg["n_labelled"])
    names, fracs = list(all_rankers(prereg)), prereg["k_fractions"]
    w_total = int(np.rint(test_w.sum(1)[0]))
    return {
        "n": task.n,
        "test_rows": w_total,
        "k": {f: k_at(f, w_total) for f in fracs},
        "rankers": aggregate(
            evaluate(task, samples, train_w, test_w, names, fracs, prereg),
            prereg["comparator"],
        ),
    }


def _entry(res: dict, boot: dict, ranker: str, v_failed: bool, prereg: dict) -> dict:
    """The flat numbers `task_status` and `flags` read, for one ranker."""
    primary = prereg["primary_k"]
    orig = res["golds"]["original"]["rankers"]
    adj = res["golds"].get("adjudicated", {}).get("rankers")
    nc = res.get("noise_control")
    return {
        "L": orig[ranker][primary]["L"],
        "G": orig[ranker][primary]["G"],
        "L_lb": boot[ranker][primary]["L"]["lb"],
        "G_lb": boot[ranker][primary]["G"]["lb"],
        "v_failed": v_failed,
        "G_by_frac": {f: orig[ranker][f]["G"] for f in prereg["k_fractions"]},
        "G_adjudicated": adj[ranker][primary]["G"] if adj else None,
        "ambiguity": bool(nc and nc["ambiguity"]),
        "G_subsets": {e: orig[e][primary]["G"] for e in SUBSET_RANKERS},
    }


def run_task(
    golds: dict[str, tuple[replay.Task, addendum1.SampleSet]],
    prereg: dict,
    n_draws: int,
    v_failed: bool,
    log: Any = print,
) -> dict:
    """Everything for one task: point estimates, bootstrap, statuses, mechanism."""
    t0 = time.perf_counter()
    task, samples = golds["original"]
    res: dict[str, Any] = {
        "name": task.name,
        "n_rows": task.n,
        "golds": {},
        "unit_cost_per_1k": unit_costs(task, prereg),
    }
    for gold, (t, s) in golds.items():
        res["golds"][gold] = (
            {"same_as_original": True}
            if t.same_as_original
            else _gold_result(t, s, prereg)
        )
    draws = bootstrap(task, samples, n_draws, prereg)
    res["bootstrap"] = {
        "draws": n_draws,
        "q": prereg["alpha"] / prereg["n_tests"],
        "rankers": {
            r: {
                f: {q: draw_summary(x, prereg) for q, x in by_q.items()}
                for f, by_q in by_f.items()
            }
            for r, by_f in draws.items()
        },
    }
    adjudicated, adj_samples = golds["adjudicated"]
    if not adjudicated.same_as_original:
        d = dissent(samples)
        res["noise_control"] = noise_control(
            task, adjudicated, dict(zip(task.ids, d.tolist(), strict=True)), prereg
        )
    boot = res["bootstrap"]["rankers"]
    entries = {
        r: _entry(res, boot, r, v_failed, prereg) for r in prereg["confirmatory"]
    }
    res["entries"] = entries
    res["status"] = {r: task_status(e, prereg) for r, e in entries.items()}
    res["flags"] = {r: flags(e, prereg) for r, e in entries.items()}
    primary = prereg["primary_k"]
    rich, simple = prereg["confirmatory"][1], prereg["confirmatory"][0]
    orig = res["golds"]["original"]["rankers"]
    res["r4_over_r2"] = {
        "gain": orig[rich][primary]["P"] * 100.0 - orig[simple][primary]["P"] * 100.0,
        "lb": boot[PAIR][primary]["G"]["lb"],
    }
    res["r4_beats_r2"] = r4_beats_r2(
        res["r4_over_r2"]["gain"], res["r4_over_r2"]["lb"], prereg
    )
    res["break_even"] = break_even(task, orig, prereg)
    res["seconds"] = time.perf_counter() - t0
    log(f"  {task.name}: {res['seconds']:.1f}s")
    return res


def _git_state() -> dict:
    """The commit and whether the tree is dirty, for this study as well as the parents.

    ``replay._git_state`` only looks at ``benchmarks/planner`` and
    ``benchmarks/golden``. This adds the study's own directory (not its
    results) and its test file, so an uncommitted edit cannot read as clean.
    """
    parent = replay._git_state()
    changed = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "benchmarks/review_next",
            ":(exclude)benchmarks/review_next/results",
            "tests/unit/test_review_next.py",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {"head": parent["head"], "dirty": bool(parent["dirty"] or changed)}


def run_study(
    tasks: TaskSet,
    checks: Checks,
    prereg: dict = PREREG_REVIEW,
    quick: bool = False,
    log: Any = print,
) -> dict:
    """Run the whole study on loaded tasks and samples and return the results dict."""
    t0 = time.perf_counter()
    n_draws = prereg["bootstrap_quick"] if quick else prereg["bootstrap"]
    out: dict[str, Any] = {"tasks": {}}
    for name in prereg["tasks"]:
        if name in tasks:
            out["tasks"][name] = run_task(
                tasks[name], prereg, n_draws, bool(checks.v123.get(name)), log
            )
    done = out["tasks"]
    out["verdict"] = verdict(
        {r: {t: done[t]["status"][r] for t in done} for r in prereg["confirmatory"]},
        {r: {t: done[t]["flags"][r] for t in done} for r in prereg["confirmatory"]},
        {t for t in done if checks.v123.get(t)},
        {t: done[t]["r4_beats_r2"] for t in done},
        {**prereg, "tasks": list(done)},
    )
    out["checks"] = {
        "lines": checks.lines,
        "v123": checks.v123,
        "disclosure": checks.disclosure,
        "splits": checks.splits,
    }
    out["meta"] = {
        "design": prereg["design"],
        "design_sha256": addendum1.sha256_file(DESIGN) if DESIGN.exists() else None,
        "prereg_review_sha256": replay.prereg_hash(prereg),
        "prereg_review": prereg,
        "parent_design_sha256": replay.design_hash(),
        "parent_prereg_sha256": replay.prereg_hash(),
        "addendum1_design_sha256": addendum1.sha256_file(addendum1.ADDENDUM_DESIGN),
        "addendum1_prereg_sha256": addendum1.prereg_addendum_hash(),
        "code_sha256": addendum1.sha256_file(Path(__file__)),
        "git": _git_state(),
        "data_sha256": checks.data_sha256,
        "quick": quick,
        "splits": prereg["splits"],
        "bootstrap": n_draws,
        "seed": replay.PREREG["seed"],
        "numpy": np.__version__,
        "seconds": time.perf_counter() - t0,
    }
    return out


# ── outputs ──────────────────────────────────────────────────────────────


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def _rate(r: dict) -> str:
    if r["rate"] is None:
        return "n/a"
    lo, hi = r["wilson"]
    return f"{r['flagged']}/{r['n']} = {r['rate']:.3f} [{lo:.3f}, {hi:.3f}]"


def _ranker_rows(agg: dict, names: list[str], f: str, boot: dict | None) -> list[str]:
    """Table rows: precision (errors per 100 reviewed), L, G and their bounds."""
    rows = []
    for n in names:
        a = agg[n][f]
        b = (boot or {}).get(n, {}).get(f)
        lb_l = replay._f(b["L"]["lb"], 2) if b else ""
        lb_g = replay._pts(b["G"]["lb"]) if b else ""
        rows.append(
            f"| {n} | {a['per100']:.1f} | {replay._f(a['L'], 2)} | {lb_l} "
            f"| {replay._pts(a['G'])} | {lb_g} | {replay._f(a['recall'], 2)} "
            f"| {replay._f(a['rows_per_error'], 2)} |"
        )
    return rows


_HEAD = (
    "| Ranker | errors per 100 reviewed | L | LB(L) | G (pts) | LB(G) | recall "
    "| rows per error |\n|---|---|---|---|---|---|---|---|"
)


def _task_section(t: str, r: dict, prereg: dict) -> list[str]:
    primary, fracs = prereg["primary_k"], prereg["k_fractions"]
    orig = r["golds"]["original"]
    boot = r["bootstrap"]["rankers"]
    shown = [prereg["null"], prereg["comparator"], *prereg["confirmatory"]]
    out = [f"## {t}\n"]
    out.append(
        f"{r['n_rows']} rows; {orig['test_rows']} test rows per split; k = "
        + "/".join(str(orig["k"][f]) for f in fracs)
        + f" at {'/'.join(f for f in fracs)}. Primary k fraction {primary}. "
        f"Bootstrap B = {r['bootstrap']['draws']}, LB at q = {r['bootstrap']['q']}.\n"
    )
    out.append(f"### Confirmatory, original gold, k fraction {primary}\n")
    out.append(_HEAD)
    out.extend(_ranker_rows(orig["rankers"], shown, primary, boot))
    out.append("")
    out.append(
        "Status: "
        + "; ".join(f"{x} **{r['status'][x]}**" for x in prereg["confirmatory"])
        + ". Flags: "
        + "; ".join(f"{x} {r['flags'][x] or 'none'}" for x in prereg["confirmatory"])
        + ".\n"
    )
    pair = boot[PAIR][primary]["G"]
    out.append(
        f"R4 minus R2 at {primary}: {replay._pts(r['r4_over_r2']['gain'])} points, "
        f"paired LB {replay._pts(pair['lb'])}; R4 beats R2 by the rule: {r['r4_beats_r2']}. "
        "SE of G (bootstrap): "
        + "; ".join(
            f"{x} {replay._f(boot[x][primary]['G']['se'], 1)}"
            for x in prereg["confirmatory"]
        )
        + ".\n"
    )
    out.append("### k sensitivity (mean G, points; original gold)\n")
    out.append("| Ranker | " + " | ".join(fracs) + " |")
    out.append("|---|" + "---|" * len(fracs))
    for x in prereg["confirmatory"]:
        cells = " | ".join(replay._pts(orig["rankers"][x][f]["G"]) for f in fracs)
        out.append(f"| {x} | {cells} |")
    out.append("")
    adj = r["golds"]["adjudicated"]
    if "rankers" in adj:
        out.append(f"### Label-noise control: adjudicated gold, k fraction {primary}\n")
        out.append(f"{adj['n']} rows; {adj['test_rows']} test rows per split.\n")
        out.append(_HEAD)
        out.extend(_ranker_rows(adj["rankers"], list(adj["rankers"]), primary, None))
        out.append("")
    out.append(f"### Exploratory (never decisive), original gold, {primary}\n")
    out.append(_HEAD)
    out.extend(
        _ranker_rows(orig["rankers"], list(prereg["exploratory"]), primary, None)
    )
    out.append("")
    out.extend(_mechanism(r))
    out.extend(_break_even(r, prereg))
    return out


def _mechanism(r: dict) -> list[str]:
    nc = r.get("noise_control")
    if not nc:
        return []
    g = nc["groups"]
    return [
        "### Mechanism (flag = D >= 1, all rows)\n",
        f"- Haiku wrong rows flagged: {_rate(g['wrong'])}; right rows: {_rate(g['right'])}",
        f"- A (still wrong after adjudication): {_rate(g['A'])}",
        f"- F (now right): {_rate(g['F'])}",
        f"- X (dropped): {_rate(g['X'])}",
        f"- F and X together: {_rate(g['FX'])}",
        f"- N (right on the original gold, wrong on the adjudicated): {_rate(g['N'])}",
        f"- flag(F u X) - flag(A) = {replay._f(nc['gap_fx_minus_a'])}; "
        f"AMBIGUITY (>= 0.20): {nc['ambiguity']}\n",
    ]


def _break_even(r: dict, prereg: dict) -> list[str]:
    hs = [str(h) for h in prereg["review_cost_usd_per_row_assumed"]]
    out = [
        "### Break-even (display only)\n",
        "| Ranker | $ per 1k rows | observed P_R / P_R1 | needed at h = "
        + " | needed at h = ".join(f"${h}" for h in hs)
        + " |",
        "|---|---|---|" + "---|" * len(hs),
    ]
    for x, b in r["break_even"].items():
        needed = " | ".join(replay._f(b["needed_ratio"][h], 2) for h in hs)
        out.append(
            f"| {x} | {b['usd_per_1k']:.3f} | {replay._f(b['observed_ratio'], 2)} | {needed} |"
        )
    out.append("")
    return out


def write_markdown(results: dict, path: Path) -> None:
    m, v = results["meta"], results["verdict"]
    prereg = m["prereg_review"]
    out = ["# Review-next: results\n"]
    out.append(
        f"Design: `{m['design']}` (sha256 `{m['design_sha256']}`). "
        f"PREREG_REVIEW sha256 `{m['prereg_review_sha256']}`. Parents: replay DESIGN "
        f"`{m['parent_design_sha256']}`, PREREG `{m['parent_prereg_sha256']}`; addendum 1 "
        f"DESIGN `{m['addendum1_design_sha256']}`, PREREG `{m['addendum1_prereg_sha256']}`.  \n"
        f"Script sha256 `{m['code_sha256']}`; git `{m['git']['head']}`"
        f"{' (dirty)' if m['git']['dirty'] else ' (clean)'}.  \n"
        f"Mode: {'QUICK (smoke run, not citeable)' if m['quick'] else 'full'}; "
        f"S = {m['splits']} splits, B = {m['bootstrap']} bootstrap draws, seed {m['seed']}; "
        f"run time {m['seconds']:.1f}s.\n"
    )
    out.append("## Verdict\n")
    out.append(f"**{v['verdict']}**\n")
    out.append(f"Selection rule: {v['selection_rule']}.\n")
    if v["v_failed"]:
        out.append(
            f"V1-V3 failed on: {', '.join(v['v_failed'])} (R2 and R4 INCONCLUSIVE there).\n"
        )
    out.extend(f"- FLAG {k}: {', '.join(f) or 'none'}" for k, f in v["flags"].items())
    out.append("")
    tasks = list(results["tasks"])
    out.append("| Ranker | " + " | ".join(tasks) + " |")
    out.append("|---|" + "---|" * len(tasks))
    for r, by_task in v["statuses"].items():
        out.append(f"| {r} | " + " | ".join(by_task[t] for t in tasks) + " |")
    out.append(
        "\nA pass is a ranking signal for the review step, not a router. The Planner KILL "
        "stands in every outcome.\n"
    )
    out.append("## Checks\n")
    out.append("```")
    out.extend(results["checks"]["lines"])
    out.append("```\n")
    out.append("| Data file | sha256 |")
    out.append("|---|---|")
    out.extend(f"| {k} | `{h}` |" for k, h in m["data_sha256"].items())
    out.append("")
    for t, r in results["tasks"].items():
        out.extend(_task_section(t, r, prereg))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_results(results: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "review-next.md", out_dir / "review-next.json"]
    write_markdown(results, paths[0])
    paths[1].write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return paths


def _refusal(args: argparse.Namespace) -> str | None:
    """Why this run must not start, or None (DESIGN section 9: one full run).

    ``--quick`` uses the full S = 1000 splits and prints the verdict, so on the
    real data it would leak the result; it is only allowed into a scratch
    directory (``--out``) or after the full run exists. A full run needs a clean
    tree and may not overwrite an earlier result.
    """
    full_json = (args.out or RESULTS) / "review-next.json"
    if args.quick and not args.out and not (RESULTS / "review-next.json").exists():
        return (
            "--quick reveals the outcome (S = 1000 splits, verdict printed): not before "
            "the full run. Use --out <scratch dir> for a fixture."
        )
    if args.quick:
        return None
    if full_json.exists():
        return f"{full_json} exists: a second full run would overwrite the result"
    if _git_state()["dirty"]:
        return (
            "the tree has uncommitted changes under benchmarks/review_next, "
            "benchmarks/planner, benchmarks/golden or the study's tests: run the "
            "one full run from the clean, reviewed commit"
        )
    return None


def main(argv: list[str] | None = None, **check_kwargs: Any) -> int:
    """CLI. `check_kwargs` go to `check` (tests point it at a fixture)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="checks only; R0 is the only ranker run"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="B = 100; reveals the outcome, so it needs --out or an existing result",
    )
    parser.add_argument("--out", type=Path, help="results directory")
    args = parser.parse_args(argv)

    refusal = None if args.check else _refusal(args)
    if refusal:
        print(f"REFUSED: {refusal}")
        return 2
    checks = check(**check_kwargs)
    print("\n".join(checks.lines))
    if not checks.hard_ok:
        print(f"\nCHECK FAILED ({len(checks.problems)} problems, listed above as FAIL)")
        return 2
    print("\nCHECK OK")
    if args.check:
        return 0

    prereg = check_kwargs.get("prereg", PREREG_REVIEW)
    results = run_study(checks.tasks, checks, prereg, quick=args.quick)
    out_dir = args.out or RESULTS / ("quick" if args.quick else "")
    for p in write_results(results, out_dir):
        print(f"wrote {p.relative_to(REPO) if p.is_relative_to(REPO) else p}")
    print(f"VERDICT: {results['verdict']['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
