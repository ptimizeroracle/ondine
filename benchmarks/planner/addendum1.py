"""Planner Addendum 1: does Haiku self-consistency find what the cheap pair misses?

Pre-registered offline experiment (design: ``benchmarks/planner/DESIGN-addendum-1.md``,
parent: ``DESIGN.md``, verdict KILL). It adds k = 3 fresh Haiku samples per row
(``label.py --sample``) to the parent's recorded answers and tests two policies
that need no labels: C4 (Haiku answers 4 times; unanimous, deliver; else ask the
strong model) and C5 (C4, or the cheap pair disagrees). The parent's comparator,
splits, bootstrap and pass rules are reused from ``replay.py`` unchanged. Every
constant lives in ``PREREG_ADDENDUM``; the results embed its hash. Do not change a
constant after seeing a result.

    uv run python benchmarks/planner/addendum1.py --check    # sample checks only
    uv run python benchmarks/planner/addendum1.py --quick    # B = 100, smoke run
    uv run python benchmarks/planner/addendum1.py            # S = 1000, B = 1000

``--check`` runs the parent's integrity check, the sample checks V1-V4 on every
sample file, the reproduction of the parent's intent C1 numbers, and prints the
flag rates; it evaluates no policy. Exit code 2 means a failed check; the study
(default and ``--quick``) then does not start. The study writes
``addendum-1.md`` and ``addendum-1.json`` to ``benchmarks/planner/results/``
(``results/quick/`` for ``--quick``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay  # noqa: E402

REPO = replay.REPO
DATA = replay.DATA
ADDENDUM_DESIGN = REPO / "benchmarks" / "planner" / "DESIGN-addendum-1.md"
RESULTS = HERE / "results"

#: Every constant of the addendum, in one place. Its sha256 goes into the results.
PREREG_ADDENDUM = {
    "design": "benchmarks/planner/DESIGN-addendum-1.md",
    "parent": {
        "design": "benchmarks/planner/DESIGN.md",
        "design_sha256": "3d667e65acd04d682855bf71d868e93b57e7aea53f2b782d4c68daaf051e4a42",
        "prereg_sha256": "01872888f24b359b5bb24ad167aa50e04d7ee41816718d7d93685e9e41bb0e3a",
        "verdict": "KILL",
    },
    # The parent's task order: `split_rng` indexes into it, so the splits and
    # bootstrap draws are the parent's, row for row.
    "tasks": list(replay.PREREG["tasks"]),
    "run_tasks": ["intent", "faithfulness"],
    "cheap": "haiku",
    "partner": "luna",
    "strong": {t: replay.PREREG["strong"][t] for t in ("intent", "faithfulness")},
    "k": 3,
    "sample_files": ["haiku-s1", "haiku-s2", "haiku-s3"],
    "replacement_file": "haiku-s4",
    "route": "openrouter",
    "model_id": "openrouter/anthropic/claude-haiku-5-5",
    "prices_per_1k_tokens": {"input": "0.0001", "output": "0.0005"},
    "prices_per_token": {"input": 1e-7, "output": 5e-7},
    "budget_usd": {"per_run": 0.05, "total": 0.30},
    "n_confirmatory": 2,
    "hull_cost_multiple": 2.0,
    "verdict": {"beat_c4_pts": 2.0},
    "v": {
        # V1: the sample's own accuracy against the original run's.
        "v1": {
            "intent": {"metric": "accuracy", "center": 0.810, "tol": 0.050},
            "faithfulness": {"metric": "balanced", "center": 0.820, "tol": 0.060},
        },
        # V2: output tokens per row, as a multiple of the original's implied value.
        "v2": {"low": 0.5, "high": 2.0},
        # V3: agreement among new samples minus agreement with the original.
        "v3_tol_pts": 5.0,
        # V4: every dev id; at most this share of null or invalid rows.
        "v4_max_bad_pct": 3.0,
    },
    "diagnostic": {
        "useful_r_blind_min": 0.40,
        "useful_ff_max": 0.10,
        "not_useful_r_blind_max": 0.20,
        "wilson_z": 1.959964,
    },
    "noise": {"flag_a_min": 0.40, "ambiguity_gap_min": 0.20},
    "exploratory": {
        "c4_samples": [1, 2],
        "other_strong": True,
        "modal_min_votes": 3,
        "vote4_point": True,
        "max_per_task": 8,
    },
    # Existing-label numbers the design discloses (section 0 and 6): the check
    # reproduces them, so the groups of section 6 are the ones the design means.
    "disclosure": {
        "intent": {
            "rows": 231,
            "haiku_wrong": 44,
            "blind": 33,
            "strong_right_on_wrong": 24,
            "strong_right_on_blind": 17,
            "strong_wrong_on_right": 7,
            "adjudicated_rows": 218,
            "adjudicated_haiku_wrong": 23,
            "groups": {"A": 17, "B": 9, "C": 7},
        },
        "faithfulness": {"rows": 200, "blind": 22, "strong_right_on_blind": 2},
    },
}

if replay.PREREG["hull_cost_multiple"] != PREREG_ADDENDUM["hull_cost_multiple"]:
    raise RuntimeError("the parent's hull_cost_multiple is no longer 2.0")
if PREREG_ADDENDUM["tasks"] != list(replay.TASKS):
    raise RuntimeError("the parent's task order changed; the split streams differ")

POLICIES = ("C4", "C5")


def prereg_addendum_hash(addendum: dict = PREREG_ADDENDUM) -> str:
    return replay.prereg_hash(addendum)


def effective_prereg(addendum: dict = PREREG_ADDENDUM) -> dict:
    """The parent's PREREG with the addendum's Bonferroni family of 2."""
    return {**replay.PREREG, "n_confirmatory": addendum["n_confirmatory"]}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(k: int, n: int, z: float) -> list[float] | None:
    """The Wilson score interval for k of n; None when n = 0."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(centre - half) / denom, (centre + half) / denom]


def at_least(k: int, n: int, threshold: float) -> bool:
    """k / n >= threshold, exactly (no float rounding at the boundary)."""
    return n > 0 and Fraction(k, n) >= Fraction(str(threshold))


def at_most(k: int, n: int, threshold: float) -> bool:
    """k / n <= threshold, exactly."""
    return n > 0 and Fraction(k, n) <= Fraction(str(threshold))


# ── the samples ──────────────────────────────────────────────────────────


@dataclass
class SampleSet:
    """Haiku's answers on a task's rows: column 0 is the original run, then k samples.

    ``codes`` are integer codes of ``norm(final_answer(.))`` (a missing row gets
    a code of its own that equals no other); ``valid`` marks a value in the task's
    allowed set; ``correct`` compares with the gold of the rows' gold file.
    """

    ids: list[str]
    codes: np.ndarray
    valid: np.ndarray
    correct: np.ndarray

    @property
    def n(self) -> int:
        return self.codes.shape[0]

    @property
    def n_columns(self) -> int:
        return self.codes.shape[1]


def build_sample_set(
    ids: list[str],
    columns: list[dict[str, str]],
    allowed: set[str],
    gold_norm: list[str],
) -> SampleSet:
    """A SampleSet from normalised final answers: one id -> answer dict per column."""
    n, m = len(ids), len(columns)
    vocab: dict[str, int] = {}
    codes = np.zeros((n, m), int)
    valid = np.zeros((n, m), bool)
    correct = np.zeros((n, m), bool)
    for j, column in enumerate(columns):
        for i, rid in enumerate(ids):
            final = column.get(rid)
            if final is None:
                codes[i, j] = -(i * m + j + 1)
                continue
            codes[i, j] = vocab.setdefault(final, len(vocab))
            valid[i, j] = final in allowed
            correct[i, j] = final == gold_norm[i]
    return SampleSet(list(ids), codes, valid, correct)


def allowed_values(task_name: str, data_dir: Path = DATA) -> set[str]:
    """The task's valid final answers (the parent's validity signal)."""
    if task_name == "intent":
        return {
            replay.norm(x)
            for x in replay.read_json(data_dir / "intent" / "labels.json", data_dir)
        }
    return {replay.norm(x) for x in replay.FAITHFULNESS_VALUES}


def _final_by_id(path: Path, data_dir: Path) -> tuple[dict[str, str], list[str], int]:
    """Normalised final answer per id, the ids in file order, and the null count."""
    rows = replay.read_jsonl(path, data_dir)
    # A null answer gets a code of its own, so two nulls never agree (V3, the
    # identical-file check); it is invalid either way, so routing is unchanged.
    finals = {
        r["id"]: f"\x00null:{r['id']}"
        if r["answer"] is None
        else replay.norm(replay.final_answer(r["answer"]))
        for r in rows
    }
    return finals, [r["id"] for r in rows], sum(r["answer"] is None for r in rows)


def label_path(task_name: str, stem: str, data_dir: Path = DATA) -> Path:
    return data_dir / "labels" / task_name / f"{stem}.jsonl"


def load_samples(
    task_name: str,
    gold: str,
    stems: list[str],
    data_dir: Path = DATA,
) -> SampleSet:
    """The original Haiku answers plus the `stems` samples, on the rows of one gold.

    Rows are the gold file's rows, in the order ``replay.load_task`` uses, so
    the arrays line up with the Task. Reads only what ``replay.guard_path`` allows.
    """
    dev = replay.read_jsonl(data_dir / task_name / "dev.jsonl", data_dir)
    adjudicated = replay.read_jsonl(
        data_dir / task_name / "dev-adjudicated.jsonl", data_dir, missing_ok=True
    )
    rows = dev if gold == "original" else (adjudicated or dev)
    columns = [
        _final_by_id(label_path(task_name, stem, data_dir), data_dir)[0]
        for stem in ["haiku", *stems]
    ]
    return build_sample_set(
        [r["id"] for r in rows],
        columns,
        allowed_values(task_name, data_dir),
        [replay.norm(r["label"]) for r in rows],
    )


def _aligned(task: replay.Task, samples: SampleSet) -> None:
    if samples.ids != task.ids:
        raise ValueError("the samples and the task do not cover the same rows")


# ── policies: the parent's signature (task, train_w, test_w, params) ─────
#
# `params["samples"]` is the SampleSet. Neither policy reads a label or a
# weight: they learn nothing, so train_w and test_w are ignored.


def _unanimous(samples: SampleSet, n_ask: int) -> np.ndarray:
    """(n,) True where all `n_ask` answers are valid and equal."""
    codes, valid = samples.codes[:, :n_ask], samples.valid[:, :n_ask]
    return valid.all(1) & (codes == codes[:, :1]).all(1)


def _pair_agrees(task: replay.Task, cheap: str, partner: str) -> np.ndarray:
    """(n, nf) True where the cheap pair agrees on a valid value (the parent's C1 test)."""
    return (
        (task.value[cheap] == task.value[partner])
        & task.valid[cheap][:, None]
        & task.valid[partner][:, None]
    )


def self_consistency(task, train_w, test_w, params):
    """C4: ask Haiku 4 times; deliver the original if all valid and equal, else ask s."""
    cheap, strong = params["cheap"], params["strong"]
    n_ask = params.get("n_ask", 4)
    samples = params["samples"]
    _aligned(task, samples)
    routed = (task.mask & ~_unanimous(samples, n_ask)[:, None])[None]
    right, routed = replay._deliver(task, routed, cheap, strong, params)
    cost = replay._cost(task, (cheap,) * n_ask, strong, routed.any(-1))
    return right, cost, routed


def sc_or_pair(task, train_w, test_w, params):
    """C5: route when C4 routes OR Haiku and Luna disagree."""
    cheap, partner, strong = params["cheap"], params["partner"], params["strong"]
    n_ask = params.get("n_ask", 4)
    samples = params["samples"]
    _aligned(task, samples)
    split = ~_unanimous(samples, n_ask)[:, None] | ~_pair_agrees(task, cheap, partner)
    routed = (task.mask & split)[None]
    right, routed = replay._deliver(task, routed, cheap, strong, params)
    cost = replay._cost(task, (cheap,) * n_ask + (partner,), strong, routed.any(-1))
    return right, cost, routed


def _modal(samples: SampleSet, n_ask: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the column of the modal answer and its vote count.

    Ties go to the original answer (column 0) when it is among the leaders,
    else to the first leader: no tie is broken by looking at the gold.
    """
    codes = samples.codes[:, :n_ask]
    votes = (codes[:, :, None] == codes[:, None, :]).sum(-1)
    top = votes.max(1)
    column = np.where(votes[:, 0] == top, 0, votes.argmax(1))
    return column, top


def modal_route(task, train_w, test_w, params):
    """Exploratory: deliver the modal answer if it has >= 3 of 4 votes (and is valid)."""
    cheap, strong = params["cheap"], params["strong"]
    n_ask, need = params.get("n_ask", 4), params["min_votes"]
    samples = params["samples"]
    _aligned(task, samples)
    column, top = _modal(samples, n_ask)
    rows = np.arange(task.n)
    split = (top < need) | ~samples.valid[rows, column]
    routed = (task.mask & split[:, None])[None]
    voted = samples.correct[rows, column][:, None]
    right = np.where(routed, task.correct[strong], voted[None])
    cost = replay._cost(task, (cheap,) * n_ask, strong, routed.any(-1))
    return right, cost, routed


def vote_point(task, train_w, test_w, params):
    """Exploratory reference point: Haiku's 4-vote answer at 4 times its cost."""
    cheap, n_ask = params["cheap"], params.get("n_ask", 4)
    samples = params["samples"]
    _aligned(task, samples)
    column, _ = _modal(samples, n_ask)
    right = samples.correct[np.arange(task.n), column][None, :, None]
    cost = replay._cost(task, (cheap,) * n_ask, None, np.zeros((1, task.n)))
    return right, cost, np.zeros((1, task.n, task.nf), bool)


def confirmatory_configs(task_name: str, samples: SampleSet, addendum: dict) -> dict:
    """name -> (policy function, params) for the two confirmatory policies."""
    base = {
        "cheap": addendum["cheap"],
        "partner": addendum["partner"],
        "strong": addendum["strong"][task_name],
        "samples": samples,
        "n_ask": addendum["k"] + 1,
    }
    return {"C4": (self_consistency, base), "C5": (sc_or_pair, base)}


def exploratory_configs(task_name: str, samples: SampleSet, addendum: dict) -> list:
    """The exploratory configs (design section 3): [(id, function, params)], at most 8."""
    ex = addendum["exploratory"]
    base = confirmatory_configs(task_name, samples, addendum)["C4"][1]
    strong = base["strong"]
    out = []
    for j in ex["c4_samples"]:
        out.append(
            (f"C4:{strong}:ask{j + 1}", self_consistency, {**base, "n_ask": j + 1})
        )
    if ex["other_strong"]:
        for other in ("sonnet", "sol", "opus"):
            if other == strong:
                continue
            out.append((f"C4:{other}", self_consistency, {**base, "strong": other}))
            out.append((f"C5:{other}", sc_or_pair, {**base, "strong": other}))
    out.append(
        (
            f"modal{ex['modal_min_votes']}of{base['n_ask']}:{strong}",
            modal_route,
            {**base, "min_votes": ex["modal_min_votes"]},
        )
    )
    if ex["vote4_point"]:
        out.append((f"vote{base['n_ask']}", vote_point, dict(base)))
    if len(out) > ex["max_per_task"]:
        raise RuntimeError(f"{len(out)} exploratory configs; the design allows 8")
    return out


# ── mechanism diagnostic and label-noise control ─────────────────────────


def blind_spot(task: replay.Task, cheap: str, partner: str) -> np.ndarray:
    """(n,) True where the cheap pair agrees on a valid value and is wrong."""
    agree = _pair_agrees(task, cheap, partner)[:, 0]
    return agree & ~task.correct[cheap][:, 0]


def _rate(flag: np.ndarray, rows: np.ndarray, z: float) -> dict:
    n, k = int(rows.sum()), int((flag & rows).sum())
    return {
        "n": n,
        "flagged": k,
        "rate": k / n if n else None,
        "wilson": wilson(k, n, z),
    }


def mechanism_diagnostic(
    task: replay.Task, flag: np.ndarray, addendum: dict = PREREG_ADDENDUM
) -> dict:
    """Flag rates of C4's trigger on the blind-spot rows, Haiku's wrong and right rows.

    ``flag`` is (n,) bool, True where C4 routes. Section 5 names them R33
    (blind-spot rows), R44 (Haiku's wrong rows) and FF (its right rows) for
    intent. ``class`` is useful / not_useful / weak by the pre-registered cuts.
    """
    d = addendum["diagnostic"]
    cheap, partner = addendum["cheap"], addendum["partner"]
    blind = blind_spot(task, cheap, partner)
    wrong = ~task.correct[cheap][:, 0]
    z = d["wilson_z"]
    out = {
        "blind": _rate(flag, blind, z),
        "wrong": _rate(flag, wrong, z),
        "right": _rate(flag, ~wrong, z),
    }
    r33, ff = out["blind"], out["right"]
    if at_least(r33["flagged"], r33["n"], d["useful_r_blind_min"]) and at_most(
        ff["flagged"], ff["n"], d["useful_ff_max"]
    ):
        out["class"] = "useful"
    elif at_most(r33["flagged"], r33["n"], d["not_useful_r_blind_max"]):
        out["class"] = "not_useful"
    else:
        out["class"] = "weak"
    return out


def blind_spot_groups(
    original: replay.Task, adjudicated: replay.Task, addendum: dict = PREREG_ADDENDUM
) -> dict[str, list[str]]:
    """Ids of the blind-spot rows (original gold) by what adjudication did to them.

    A: kept in the adjudicated gold and Haiku is still wrong against its label.
    B: kept, and Haiku is now right. C: no defensible label, so the row is
    absent from ``dev-adjudicated.jsonl`` (the verdict was "drop").
    """
    blind = blind_spot(original, addendum["cheap"], addendum["partner"])
    position = {rid: k for k, rid in enumerate(adjudicated.ids)}
    right_now = adjudicated.correct[addendum["cheap"]][:, 0]
    groups: dict[str, list[str]] = {"A": [], "B": [], "C": []}
    for rid, is_blind in zip(original.ids, blind, strict=True):
        if not is_blind:
            continue
        k = position.get(rid)
        groups["C" if k is None else "B" if right_now[k] else "A"].append(rid)
    return groups


def noise_control(
    original: replay.Task,
    adjudicated: replay.Task,
    flag_by_id: dict[str, bool],
    addendum: dict = PREREG_ADDENDUM,
) -> dict:
    """Section 6: flag rates on the adjudicated gold and on groups A, B, C."""
    cheap, z = addendum["cheap"], addendum["diagnostic"]["wilson_z"]
    n = addendum["noise"]
    flags = np.array([flag_by_id[rid] for rid in adjudicated.ids], bool)
    wrong = ~adjudicated.correct[cheap][:, 0]
    groups = blind_spot_groups(original, adjudicated, addendum)

    def group_rate(ids: list[str]) -> dict:
        arr = np.array([flag_by_id[i] for i in ids], bool)
        return _rate(arr, np.ones(len(ids), bool), z)

    a, b, c = (group_rate(groups[g]) for g in "ABC")
    b_or_c = group_rate(groups["B"] + groups["C"])
    finds = at_least(a["flagged"], a["n"], n["flag_a_min"])
    gap = None
    ambiguity = False
    if a["n"] and b_or_c["n"]:
        gap = Fraction(b_or_c["flagged"], b_or_c["n"]) - Fraction(a["flagged"], a["n"])
        ambiguity = gap >= Fraction(str(n["ambiguity_gap_min"]))
        gap = float(gap)
    return {
        "wrong": _rate(flags, wrong, z),
        "right": _rate(flags, ~wrong, z),
        "groups": {"A": a, "B": b, "C": c, "B_or_C": b_or_c},
        "finds_real_errors": finds,
        "tracks_ambiguity": ambiguity,
        "gap_b_or_c_minus_a": gap,
    }


def disclosure_counts(
    original: replay.Task, adjudicated: replay.Task, strong: str, addendum: dict
) -> dict:
    """The existing-label counts the design discloses, recomputed."""
    cheap, partner = addendum["cheap"], addendum["partner"]
    blind = blind_spot(original, cheap, partner)
    wrong = ~original.correct[cheap][:, 0]
    s_right = original.correct[strong][:, 0]
    groups = blind_spot_groups(original, adjudicated, addendum)
    return {
        "rows": original.n,
        "haiku_wrong": int(wrong.sum()),
        "blind": int(blind.sum()),
        "strong_right_on_wrong": int((s_right & wrong).sum()),
        "strong_right_on_blind": int((s_right & blind).sum()),
        "strong_wrong_on_right": int((~s_right & ~wrong).sum()),
        "adjudicated_rows": adjudicated.n,
        "adjudicated_haiku_wrong": int((~adjudicated.correct[cheap][:, 0]).sum()),
        "groups": {g: len(ids) for g, ids in groups.items()},
    }


# ── sample checks ────────────────────────────────────────────────────────


@dataclass
class Checks:
    """What the sample checks found. ``problems`` stop the run; ``v123`` make it inconclusive."""

    lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    v123: dict[str, list[str]] = field(default_factory=dict)
    files: dict[str, dict] = field(default_factory=dict)
    selected: dict[str, list[str]] = field(default_factory=dict)
    v3: dict[str, dict] = field(default_factory=dict)
    c1_reproduction: dict = field(default_factory=dict)
    disclosure: dict = field(default_factory=dict)

    @property
    def hard_ok(self) -> bool:
        return not self.problems

    @property
    def ok(self) -> bool:
        return self.hard_ok and not any(self.v123.values())

    def say(self, text: str) -> None:
        self.lines.append(text)

    def fail(self, text: str) -> None:
        self.problems.append(text)
        self.lines.append(f"FAIL: {text}")

    def fail_v(self, task: str, text: str) -> None:
        self.v123.setdefault(task, []).append(text)
        self.lines.append(f"FAIL ({task}): {text}")


def implied_original_output_tokens(cost: dict, addendum: dict) -> float:
    """Output tokens per row implied by the original run: (cost - p_in T) / (p_out - p_in) / rows."""
    p = addendum["prices_per_token"]
    out = (float(cost["cost"]) - p["input"] * float(cost["tokens"])) / (
        p["output"] - p["input"]
    )
    return out / int(cost["rows"])


def check_v1(value: float, spec: dict) -> dict:
    return {
        "value": value,
        "lo": spec["center"] - spec["tol"],
        "hi": spec["center"] + spec["tol"],
        "ok": abs(value - spec["center"]) <= spec["tol"] + 1e-9,
    }


def check_v2(per_row: float | None, implied: float, spec: dict) -> dict:
    if per_row is None:
        return {"value": None, "implied": implied, "ratio": None, "ok": False}
    ratio = per_row / implied if implied > 0 else float("inf")
    return {
        "value": per_row,
        "implied": implied,
        "ratio": ratio,
        "ok": spec["low"] <= ratio <= spec["high"],
    }


def agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float((a == b).mean())


def check_v3(codes: np.ndarray, spec_pts: float) -> dict:
    """Mean agreement among the new samples minus their mean agreement with the original.

    `codes` is (n, 1 + k): column 0 the original. Positive means the new
    samples agree with each other more than with the original.
    """
    k = codes.shape[1] - 1
    new = [
        agreement(codes[:, i], codes[:, j])
        for i in range(1, k + 1)
        for j in range(i + 1, k + 1)
    ]
    old = [agreement(codes[:, 0], codes[:, i]) for i in range(1, k + 1)]
    if not new:
        return {"among_new": None, "with_original": None, "diff_pts": None, "ok": False}
    diff = 100.0 * (np.mean(new) - np.mean(old))
    return {
        "among_new": float(np.mean(new)),
        "with_original": float(np.mean(old)),
        "diff_pts": float(diff),
        "ok": bool(abs(diff) <= spec_pts + 1e-9),
    }


def identical_columns(codes: np.ndarray) -> list[tuple[int, int]]:
    """Pairs of columns that give the same answer on every row."""
    m = codes.shape[1]
    return [
        (i, j)
        for i in range(m)
        for j in range(i + 1, m)
        if (codes[:, i] == codes[:, j]).all()
    ]


def reproduce_parent_c1(
    task: replay.Task, prereg: dict, reference: dict | None, tol: float = 1e-9
) -> dict:
    """Re-run the parent's C1 on intent and compare accuracy and Delta A to its results.

    `reference` is {"acc": ..., "delta_flat": ...} (the parent's recorded means
    over splits, accuracy as a fraction and Delta A in points). Fails loudly
    on a difference above `tol`: the splits, hull or data are not the parent's.
    """
    strong = prereg["strong"][task.name]
    params = replay.confirmatory_params(task.name, prereg)["C1"]
    cache = replay._SplitCache(task, prereg)
    got = replay.summarise(replay.run_splits(task, cache, "C1", params, strong))
    mine = {"acc": got["acc"]["mean"], "delta_flat": got["delta_flat"]["mean"]}
    out = {"got": mine, "reference": reference, "tol": tol, "ok": False}
    if reference is None:
        return out
    out["diff"] = {k: abs(mine[k] - reference[k]) for k in mine}
    out["ok"] = all(d <= tol for d in out["diff"].values())
    return out


def committed_reference(path: Path | None = None) -> dict | None:
    """The parent's committed intent C1 accuracy and Delta A, or None if absent."""
    path = path or RESULTS / "replay.json"
    if not path.exists():
        return None
    c1 = json.loads(path.read_text(encoding="utf-8"))["tasks"]["intent"]["golds"][
        "original"
    ]["confirmatory"]["C1"]
    return {"acc": c1["acc"]["mean"], "delta_flat": c1["delta_flat"]["mean"]}


@dataclass
class _Context:
    """What one task's file checks compare against."""

    task: replay.Task
    dev_ids: list[str]
    allowed: set[str]
    gold_norm: dict[str, str]
    original: dict[str, str]
    implied_tokens: float
    spec: dict


def _file_report(stem: str, task_name: str, ctx: _Context, data_dir: Path, add: dict):
    """Everything known about one sample file: V1, V2, V4 and its hashes."""
    path = label_path(task_name, stem, data_dir)
    cost_path = path.with_suffix(".cost.json")
    rep: dict = {"stem": stem, "exists": path.exists() and cost_path.exists()}
    if not rep["exists"]:
        return rep, None
    finals, ids, nulls = _final_by_id(path, data_dir)
    cost = replay.read_json(cost_path, data_dir)
    n = len(ctx.dev_ids)
    rows = cost.get("rows")
    right = np.array([finals.get(i) == ctx.gold_norm[i] for i in ctx.dev_ids])
    value = float(
        replay.metric(ctx.task, right[:, None], ctx.task.gold_class, np.ones(n))
    )
    bad = sum(
        finals.get(i) is None or finals[i] not in ctx.allowed for i in ctx.dev_ids
    )
    out_tokens = cost.get("output_tokens")
    rep.update(
        sha256=sha256_file(replay.guard_path(path, data_dir)),
        cost_sha256=sha256_file(replay.guard_path(cost_path, data_dir)),
        cost=float(cost.get("cost", float("nan"))),
        route=cost.get("route"),
        model=cost.get("model"),
        input_tokens=cost.get("input_tokens"),
        output_tokens=out_tokens,
        rows=rows,
        nulls=nulls,
        v1=check_v1(value, ctx.spec["v1"][task_name]),
        v2=check_v2(
            None if out_tokens is None or not rows else out_tokens / rows,
            ctx.implied_tokens,
            ctx.spec["v2"],
        ),
    )
    ids_ok = len(ids) == len(set(ids)) and set(ids) == set(ctx.dev_ids)
    provenance = (
        rep["route"] == add["route"]
        and rep["model"] == add["model_id"]
        and rep["rows"] == n
    )
    rep["v4"] = {
        "ids_ok": ids_ok,
        "n_ids": len(ids),
        "n_duplicates": len(ids) - len(set(ids)),
        "n_missing": len(set(ctx.dev_ids) - set(ids)),
        "n_extra": len(set(ids) - set(ctx.dev_ids)),
        "bad": bad,
        "bad_pct": 100.0 * bad / n,
        "bad_ok": bad * 100 <= ctx.spec["v4_max_bad_pct"] * n,
        "provenance_ok": provenance,
    }
    # Measured cost against the tokens at the explicit prices: reported only.
    p = add["prices_per_token"]
    if rep["input_tokens"] is not None and out_tokens is not None:
        rep["cost_from_tokens"] = (
            p["input"] * rep["input_tokens"] + p["output"] * out_tokens
        )
    return rep, finals


def sample_checks(
    data_dir: Path = DATA,
    addendum: dict = PREREG_ADDENDUM,
    prereg: dict | None = None,
    reference: dict | None | str = "committed",
) -> Checks:
    """Verify the sample files (V1-V4), the parent's reproduction and the disclosure.

    `reference` is the parent's C1 result to reproduce: "committed" reads
    ``results/replay.json``; None skips the reproduction (tests only).
    """
    prereg = prereg or effective_prereg(addendum)
    rep = Checks()
    rep.say(f"PREREG_ADDENDUM sha256 {prereg_addendum_hash(addendum)}")
    rep.say(
        "addendum design sha256 "
        + (sha256_file(ADDENDUM_DESIGN) if ADDENDUM_DESIGN.exists() else "MISSING")
    )
    parent = addendum["parent"]
    if replay.prereg_hash() != parent["prereg_sha256"]:
        rep.fail(
            "the parent PREREG hash is not the one this addendum was written against"
        )
    if replay.design_hash() != parent["design_sha256"]:
        rep.fail(
            "the parent DESIGN.md hash is not the one this addendum was written against"
        )
    if not rep.problems:
        rep.say("parent DESIGN.md and PREREG hashes: OK")

    tasks = {}
    for name in addendum["run_tasks"]:
        try:
            tasks[name] = {g: replay.load_task(name, g, data_dir) for g in replay.GOLDS}
        except FileNotFoundError as e:
            rep.fail(f"{name}: missing file {e.filename}")
    _check_disclosure(tasks, addendum, rep)
    _check_reproduction(tasks, prereg, reference, rep)

    for name, golds in tasks.items():
        original_cost = replay.read_json(
            data_dir / "labels" / name / "haiku.cost.json", data_dir
        )
        _check_task_files(
            name, golds["original"], original_cost, data_dir, addendum, rep
        )
    return rep


def _check_disclosure(tasks: dict, addendum: dict, rep: Checks) -> None:
    for name, expected in addendum.get("disclosure", {}).items():
        if name not in tasks or tasks[name]["adjudicated"].same_as_original:
            continue
        got = disclosure_counts(
            tasks[name]["original"],
            tasks[name]["adjudicated"],
            addendum["strong"][name],
            addendum,
        )
        rep.disclosure[name] = got
        wrong = {
            k: (got[k], v) for k, v in expected.items() if k in got and got[k] != v
        }
        if wrong:
            rep.fail(
                f"{name}: existing labels differ from the disclosure (got, expected): {wrong}"
            )
        else:
            rep.say(f"[{name}] the design's disclosed counts reproduce: {expected}")


def _check_reproduction(tasks: dict, prereg: dict, reference, rep: Checks) -> None:
    if reference is None or "intent" not in tasks:
        return
    ref = committed_reference() if reference == "committed" else reference
    if ref is None:
        rep.fail(
            "benchmarks/planner/results/replay.json missing: cannot reproduce the parent's C1"
        )
        return
    got = reproduce_parent_c1(tasks["intent"]["original"], prereg, ref)
    rep.c1_reproduction = got
    if got["ok"]:
        rep.say(
            "parent's intent C1 reproduced: accuracy "
            f"{got['got']['acc']:.9f}, Delta A {got['got']['delta_flat']:.9f} "
            f"(max |diff| {max(got['diff'].values()):.1e})"
        )
    else:
        rep.fail(f"the parent's intent C1 is not reproduced: {got}")


def _select_files(
    reports: dict[str, dict], addendum: dict, task: str, rep: Checks
) -> list[str]:
    """The k sample files to use: s1..sk, with one V4 failure replaced by s4."""
    stems = list(addendum["sample_files"])
    spare = addendum["replacement_file"]
    missing = [s for s in stems if not reports[s]["exists"]]
    for s in missing:
        rep.fail(f"{task}: labels/{task}/{s}.jsonl or its .cost.json is missing")
    if missing:
        return []

    def v4_ok(r):
        v = r["v4"]
        return v["ids_ok"] and v["bad_ok"] and v["provenance_ok"] and not r["identical"]

    failing = [s for s in stems if not v4_ok(reports[s])]
    spare_report = reports[spare]
    if not failing:
        if spare_report["exists"]:
            rep.fail(
                f"{task}: {spare} exists but no file fails V4; nothing may be replaced"
            )
        return stems
    only_v4 = [s for s in failing if reports[s]["v1"]["ok"] and reports[s]["v2"]["ok"]]
    if len(failing) > 1 or failing != only_v4:
        for s in failing:
            rep.fail(
                f"{task}/{s}: fails V4 {_v4_reason(reports[s])} and cannot be replaced"
            )
        return stems
    (bad,) = failing
    if not spare_report["exists"]:
        rep.fail(
            f"{task}/{bad}: fails V4 {_v4_reason(reports[bad])}; the design allows "
            f"one replacement: draw {spare}"
        )
        return stems
    if not v4_ok(spare_report):
        rep.fail(
            f"{task}/{spare}: the replacement also fails V4 {_v4_reason(spare_report)}"
        )
        return stems
    rep.say(
        f"[{task}] {bad} fails only V4 {_v4_reason(reports[bad])}: replaced by {spare}"
    )
    return [spare if s == bad else s for s in stems]


def _v4_reason(r: dict) -> str:
    v, why = r["v4"], []
    if not v["ids_ok"]:
        why.append(
            f"ids ({v['n_missing']} missing, {v['n_extra']} extra, {v['n_duplicates']} duplicated)"
        )
    if not v["bad_ok"]:
        why.append(f"{v['bad']} null or invalid rows = {v['bad_pct']:.1f}%")
    if not v["provenance_ok"]:
        why.append(f"route/model/rows {r['route']}/{r['model']}/{r['rows']}")
    if r["identical"]:
        why.append(f"identical on every row to {r['identical']}")
    return "(" + "; ".join(why) + ")"


def _earlier(stem: str, stems: list[str]) -> list[str]:
    """The sample files `stem` is compared with for "identical on every row".

    The original run always; then the files drawn before it (the replacement
    is compared with all of the first k).
    """
    return ["haiku", *stems[: stems.index(stem)]]


def _file_line(r: dict) -> str:
    if r["v2"]["ratio"] is None:
        return f"  {r['stem']}: no output_tokens in its cost file"
    v1, v2, v4 = r["v1"], r["v2"], r["v4"]
    mark = lambda ok: "ok" if ok else "FAIL"  # noqa: E731
    return (
        f"  {r['stem']}: V1 {v1['value']:.3f} in [{v1['lo']:.3f}, {v1['hi']:.3f}] "
        f"{mark(v1['ok'])}; V2 {v2['value']:.1f} tokens/row, x{v2['ratio']:.2f} of the "
        f"original {mark(v2['ok'])}; V4 ids {mark(v4['ids_ok'])}, bad {v4['bad']} "
        f"({v4['bad_pct']:.1f}%) {mark(v4['bad_ok'])}, provenance "
        f"{mark(v4['provenance_ok'])}; sha256 {r['sha256'][:12]}, measured cost "
        f"${r['cost']:.4f}"
    )


def _check_task_files(
    name: str,
    task: replay.Task,
    original_cost: dict,
    data_dir: Path,
    addendum: dict,
    rep: Checks,
) -> None:
    dev = replay.read_jsonl(data_dir / name / "dev.jsonl", data_dir)
    original, _, _ = _final_by_id(label_path(name, "haiku", data_dir), data_dir)
    ctx = _Context(
        task=task,
        dev_ids=[r["id"] for r in dev],
        allowed=allowed_values(name, data_dir),
        gold_norm={r["id"]: replay.norm(r["label"]) for r in dev},
        original=original,
        implied_tokens=implied_original_output_tokens(original_cost, addendum),
        spec=addendum["v"],
    )
    rep.say(
        f"[{name}] original Haiku: {ctx.implied_tokens:.1f} output tokens per row implied"
    )
    stems = [*addendum["sample_files"], addendum["replacement_file"]]
    reports, finals = {}, {"haiku": original}
    for stem in stems:
        reports[stem], found = _file_report(stem, name, ctx, data_dir, addendum)
        if found is not None:
            finals[stem] = found
    for stem, r in reports.items():
        r["identical"] = []
        if r["exists"]:
            r["identical"] = [
                o
                for o in _earlier(stem, stems)
                if o in finals
                and all(finals[stem].get(i) == finals[o].get(i) for i in ctx.dev_ids)
            ]
    known = {*stems}
    extra = [
        p.stem
        for p in sorted((data_dir / "labels" / name).glob("haiku-s*.jsonl"))
        if p.stem not in known and "smoke" not in p.stem
    ]
    if extra:
        rep.fail(f"{name}: unexpected sample files {extra}")
    before = len(rep.problems)
    selected = _select_files(reports, addendum, name, rep)
    rep.selected[name] = selected
    rep.files[name] = {
        s: {**r, "used": s in selected} for s, r in reports.items() if r["exists"]
    }
    if not selected or len(rep.problems) > before:
        return
    for stem in selected:
        r = reports[stem]
        rep.say(_file_line(r))
        v1, v2 = r["v1"], r["v2"]
        if not v1["ok"]:
            rep.fail_v(
                name,
                f"{stem} V1: {v1['value']:.3f} outside [{v1['lo']:.3f}, {v1['hi']:.3f}]",
            )
        if not v2["ok"]:
            rep.fail_v(
                name,
                f"{stem} V2: output tokens per row {v2['value']} "
                f"(x{v2['ratio']}) outside the band",
            )
    samples = load_samples(name, "original", selected, data_dir)
    v3 = check_v3(samples.codes, addendum["v"]["v3_tol_pts"])
    rep.v3[name] = v3
    rep.say(
        f"  V3: agreement among new {v3['among_new']:.3f}, with the original "
        f"{v3['with_original']:.3f}, difference {v3['diff_pts']:+.1f} points "
        f"{'ok' if v3['ok'] else 'FAIL'}"
    )
    if not v3["ok"]:
        rep.fail_v(name, f"V3: difference {v3['diff_pts']:+.1f} points")
    same = identical_columns(samples.codes)
    if same:
        rep.fail(f"{name}: samples identical on every row: columns {same}")
    flags = flag_rates(task, samples, addendum)
    rep.say(
        f"  flag rates on all {task.n} rows: C4 {flags['C4']:.3f} "
        f"({flags['C4_rows']} rows), C5 {flags['C5']:.3f} ({flags['C5_rows']} rows)"
    )


def flag_rates(task: replay.Task, samples: SampleSet, addendum: dict) -> dict:
    """Share of rows C4 and C5 route to the strong model."""
    cheap, partner = addendum["cheap"], addendum["partner"]
    c4 = ~_unanimous(samples, addendum["k"] + 1)
    c5 = c4 | ~_pair_agrees(task, cheap, partner)[:, 0]
    return {
        "C4": float(c4.mean()),
        "C5": float(c5.mean()),
        "C4_rows": int(c4.sum()),
        "C5_rows": int(c5.sum()),
    }


# ── the study ────────────────────────────────────────────────────────────


def _run(task, cache, fn, params, strong, full=True):
    """Per-split evaluation of one policy over the parent's splits."""
    train_w, test_w, pts = cache.get(params.get("n_l", cache.prereg["n_labelled"]))
    return replay.evaluate(
        task, fn(task, train_w, test_w, params), test_w, pts, params, strong, full
    )


def _params_json(params: dict) -> dict:
    return {k: v for k, v in params.items() if k != "samples"}


def run_gold(
    task: replay.Task, samples: SampleSet, addendum, prereg, exploratory: bool
):
    """Evaluate the policies on one gold (point estimates over S splits)."""
    strong = addendum["strong"][task.name]
    cache = replay._SplitCache(task, prereg)
    out: dict = {"n": task.n, "confirmatory": {}, "exploratory": {}}
    for name, (fn, params) in confirmatory_configs(
        task.name, samples, addendum
    ).items():
        out["confirmatory"][name] = replay.summarise(
            _run(task, cache, fn, params, strong)
        )
    if exploratory:
        for cid, fn, params in exploratory_configs(task.name, samples, addendum):
            out["exploratory"][cid] = {
                "params": _params_json(params),
                "summary": replay.summarise(_run(task, cache, fn, params, strong)),
            }
    return out


def _entries(res: dict, orig: dict, prereg: dict) -> tuple[dict, dict]:
    """The flat numbers `verdict_addendum` reads, in points and $ per row."""
    adj = res["golds"].get("adjudicated", {})
    conf = {}
    for pol in POLICIES:
        s = orig["confirmatory"][pol]
        boot = res["bootstrap"]["policies"][pol]
        entry = {
            "delta_flat": s["delta_flat"]["mean"],
            "delta_length": s["delta_length"]["mean"],
            "d2_flat": s["d2_flat"]["mean"],
            "d2_length": s["d2_length"]["mean"],
            "delta_lb": boot["delta_lb"],
            "d2_lb": boot["d2_lb"],
            "cost_flat": s["cost_flat"]["mean"] / 1000.0,
            "cost_length": s["cost_length"]["mean"] / 1000.0,
        }
        a = (
            orig["confirmatory"][pol]
            if adj.get("same_as_original")
            else adj.get("confirmatory", {}).get(pol)
        )
        if a is not None:
            entry["delta_adjudicated"] = a["delta_flat"]["mean"]
            entry["acc_adjudicated"] = a["acc"]["mean"]
            entry["sign_flip"] = entry["delta_flat"] * entry["delta_adjudicated"] < 0
        conf[pol] = entry
    paired = {
        f"diff_{kind}": conf["C5"][f"delta_{kind}"] - conf["C4"][f"delta_{kind}"]
        for kind in replay.COST_KINDS
    }
    paired["diff_lb"] = res["bootstrap"]["paired_c5_vs_c4"]["diff_lb"]
    return conf, paired


def run_task(
    golds: dict[str, tuple[replay.Task, SampleSet]],
    addendum: dict,
    prereg: dict,
    n_draws: int,
    log=print,
) -> dict:
    task, samples = golds["original"]
    name = task.name
    strong = addendum["strong"][name]
    t0 = time.perf_counter()
    full = np.ones(task.n)
    res: dict = {
        "name": name,
        "n_rows": task.n,
        "strong": strong,
        "metric_kind": task.metric_kind,
        "models": {
            m: {
                "cost_per_1k": 1000.0 * task.u_flat[m],
                "metric": replay._clean(
                    replay.metric(task, task.correct[m], task.gold_class, full)
                ),
            }
            for m in task.models
        },
        "flags": flag_rates(task, samples, addendum),
        "golds": {},
    }
    for gold, (t, s) in golds.items():
        if t.same_as_original:
            res["golds"][gold] = {"same_as_original": True}
            continue
        res["golds"][gold] = run_gold(t, s, addendum, prereg, gold == "original")
    orig = res["golds"]["original"]

    # Cluster bootstrap on the original gold: the parent's stream (purpose 1).
    configs = confirmatory_configs(name, samples, addendum)
    draws = replay.cluster_bootstrap(
        task,
        configs,
        prereg,
        replay.split_rng(prereg, name, "original", 1, prereg["n_labelled"]),
        n_draws,
        strong,
    )
    diff = draws["C5"]["delta"] - draws["C4"]["delta"]
    res["bootstrap"] = {
        "n_draws": n_draws,
        "alpha_each": prereg["alpha"] / prereg["n_confirmatory"],
        "policies": {
            pol: {
                "delta_lb": replay.lower_bound(d["delta"], prereg),
                "d2_lb": replay.lower_bound(d["d2"], prereg),
                "delta_boot_mean": float(d["delta"].mean()),
                "d2_boot_mean": float(d["d2"].mean()),
            }
            for pol, d in draws.items()
        },
        "paired_c5_vs_c4": {
            "diff_lb": replay.lower_bound(diff, prereg),
            "diff_boot_mean": float(diff.mean()),
        },
    }
    res["confirmatory"], res["paired"] = _entries(res, orig, prereg)

    flag = ~_unanimous(samples, addendum["k"] + 1)
    res["diagnostic"] = mechanism_diagnostic(task, flag, addendum)
    adjudicated = golds["adjudicated"][0]
    if not adjudicated.same_as_original:
        res["noise_control"] = noise_control(
            task, adjudicated, dict(zip(task.ids, flag.tolist(), strict=True)), addendum
        )
    res["seconds"] = time.perf_counter() - t0
    log(f"  {name}: {res['seconds']:.1f}s")
    return res


# ── verdict ──────────────────────────────────────────────────────────────


def beats_c4(task_res: dict, addendum: dict) -> bool:
    """C5 beats C4 by >= 2 points under both cost models, with a paired LB > 0."""
    pr = task_res["paired"]
    margin = addendum["verdict"]["beat_c4_pts"]
    return (
        all(pr[f"diff_{k}"] >= margin for k in replay.COST_KINDS) and pr["diff_lb"] > 0
    )


def verdict_addendum(
    results: dict, addendum: dict = PREREG_ADDENDUM, prereg: dict | None = None
) -> dict:
    """The pre-registered decision (design section 4), on the original gold of intent."""
    prereg = prereg or effective_prereg(addendum)
    tasks = results["tasks"]
    if "intent" not in tasks:
        raise ValueError("the verdict is decided on intent; run it")
    failed = results.get("v123", {})
    passes = {
        t: {p: replay.policy_passes(r["confirmatory"][p], prereg) for p in POLICIES}
        for t, r in tasks.items()
    }
    ok = {t: {p: v["pass"] for p, v in d.items()} for t, d in passes.items()}
    out = {"passes": ok, "detail": passes, "flags": [], "reasons": []}
    for t in tasks:
        if failed.get(t):
            out["flags"].append(f"{t}: sample checks failed: {'; '.join(failed[t])}")
    if failed.get("intent"):
        return {
            **out,
            "verdict": "INCONCLUSIVE",
            "scope": None,
            "reasons": [
                "V1-V3 fail on intent: the new samples may not be Haiku's own; "
                "new data needs Addendum 2"
            ],
        }
    intent = ok["intent"]
    if not any(intent.values()):
        return {
            **out,
            "verdict": "KILL_STANDS",
            "scope": None,
            "reasons": ["neither C4 nor C5 passes on intent"],
        }
    if intent["C4"]:
        if intent["C5"] and beats_c4(tasks["intent"], addendum):
            name, family = "BUILD_SC_PLUS_PAIR_TRIGGER", "C5"
            out["reasons"].append("C5 passes and beats C4 by the margin")
        else:
            name, family = "BUILD_SC_TRIGGER", "C4"
            out["reasons"].append("C4 passes; C5 does not both pass and beat it")
    else:
        name, family = "BUILD_SC_PLUS_PAIR_TRIGGER", "C5"
        out["reasons"].append("only C5 passes on intent")
    # Rule 4: a general floor needs faithfulness too (its data must be sound).
    other = [
        t for t in ("faithfulness",) if t in ok and ok[t][family] and not failed.get(t)
    ]
    scope = "general_floor" if other else "classification_only_single_task"
    diagnostic = tasks["intent"]["diagnostic"]["class"]
    if diagnostic == "not_useful":
        out["flags"].append(
            "a policy passes but the mechanism diagnostic says not useful"
        )
    for pol in POLICIES:
        if ok["intent"][pol] and tasks["intent"]["confirmatory"][pol].get("sign_flip"):
            out["flags"].append(f"{pol}: Delta A changes sign on the adjudicated gold")
    noise = tasks["intent"].get("noise_control", {})
    if noise and not noise["finds_real_errors"]:
        out["flags"].append(
            "a policy passes but the flag catches under 40% of the rows still wrong "
            "on the adjudicated gold (group A)"
        )
    if noise.get("tracks_ambiguity"):
        out["flags"].append(
            "the flag tracks ambiguity (B and C flagged 20+ points above A)"
        )
    return {**out, "verdict": name, "scope": scope, "scope_tasks": ["intent", *other]}


# ── outputs ──────────────────────────────────────────────────────────────


def _git_state() -> dict:
    return replay._git_state()


def run_study(
    tasks: dict[str, dict[str, tuple[replay.Task, SampleSet]]],
    checks: Checks,
    addendum: dict = PREREG_ADDENDUM,
    prereg: dict | None = None,
    quick: bool = False,
    log=print,
) -> dict:
    """Run the whole study on loaded tasks and samples and return the results dict."""
    prereg = prereg or effective_prereg(addendum)
    t0 = time.perf_counter()
    n_draws = prereg["bootstrap_quick"] if quick else prereg["bootstrap"]
    out: dict = {"tasks": {}}
    for name in addendum["run_tasks"]:
        if name in tasks:
            out["tasks"][name] = run_task(tasks[name], addendum, prereg, n_draws, log)
    out["v123"] = checks.v123
    out["verdict"] = verdict_addendum(out, addendum, prereg)
    out["checks"] = {
        "lines": checks.lines,
        "files": checks.files,
        "selected": checks.selected,
        "v3": checks.v3,
        "c1_reproduction": checks.c1_reproduction,
        "disclosure": checks.disclosure,
    }
    measured = {
        t: sum(f["cost"] for f in files.values() if f["used"])
        for t, files in checks.files.items()
    }
    out["measured_cost_usd"] = {**measured, "total": sum(measured.values())}
    out["meta"] = {
        "design": addendum["design"],
        "design_sha256": sha256_file(ADDENDUM_DESIGN)
        if ADDENDUM_DESIGN.exists()
        else None,
        "parent_design_sha256": replay.design_hash(),
        "parent_prereg_sha256": replay.prereg_hash(),
        "prereg_addendum_sha256": prereg_addendum_hash(addendum),
        "prereg_addendum": addendum,
        "code_sha256": sha256_file(Path(__file__)),
        "git": _git_state(),
        "sample_files": {
            t: {
                s: {
                    "sha256": f["sha256"],
                    "cost_sha256": f["cost_sha256"],
                    "used": f["used"],
                }
                for s, f in files.items()
            }
            for t, files in checks.files.items()
        },
        "quick": quick,
        "splits": prereg["splits"],
        "bootstrap": n_draws,
        "seed": prereg["seed"],
        "numpy": np.__version__,
        "seconds": time.perf_counter() - t0,
    }
    return out


def _pts(x, digits=1) -> str:
    return replay._pts(x, digits)


def _f(x, digits=3) -> str:
    return replay._f(x, digits)


def _ci(rate: dict) -> str:
    w = rate["wilson"]
    if rate["rate"] is None:
        return "n/a"
    return (
        f"{rate['flagged']}/{rate['n']} = {rate['rate']:.3f} [{w[0]:.3f}, {w[1]:.3f}]"
    )


def write_markdown(results: dict, path: Path) -> None:
    m, v = results["meta"], results["verdict"]
    lines = ["# Planner Addendum 1: results\n"]
    lines.append(
        f"Design: `{m['design']}` (sha256 `{m['design_sha256']}`). "
        f"PREREG_ADDENDUM sha256 `{m['prereg_addendum_sha256']}`. "
        f"Parent DESIGN `{m['parent_design_sha256']}`, PREREG `{m['parent_prereg_sha256']}`.  \n"
        f"Script sha256 `{m['code_sha256']}`; git `{m['git']['head']}`"
        f"{' (dirty)' if m['git']['dirty'] else ' (clean)'}.  \n"
        f"Mode: {'QUICK (smoke run, not citeable)' if m['quick'] else 'full'}; "
        f"S = {m['splits']} splits, B = {m['bootstrap']} bootstrap draws, "
        f"seed {m['seed']}; run time {m['seconds']:.1f}s.\n"
    )
    lines.append("## Verdict\n")
    lines.append(
        f"**{v['verdict']}**" + (f", scope: {v['scope']}" if v["scope"] else "")
    )
    lines.append("")
    lines.extend(f"- {r}" for r in v["reasons"])
    lines.extend(f"- FLAG: {f}" for f in v["flags"])
    lines.append("")
    lines.append("| Task | C4 | C5 |")
    lines.append("|---|---|---|")
    for t, d in v["passes"].items():
        lines.append(
            f"| {t} | "
            + " | ".join("pass" if d[p] else "fail" for p in POLICIES)
            + " |"
        )
    lines.append("")
    lines.append(
        "A pass yields an escalation trigger for Ondine's existing escalation step, "
        "not a Planner. The Planner KILL stands in every outcome.\n"
    )
    lines.append("## Sample checks\n")
    lines.append("```")
    lines.extend(results["checks"]["lines"])
    lines.append("```\n")
    lines.append("| Task | file | used | sha256 | measured cost |")
    lines.append("|---|---|---|---|---|")
    for t, files in results["checks"]["files"].items():
        for s, f in files.items():
            lines.append(
                f"| {t} | {s} | {f['used']} | `{f['sha256']}` | ${f['cost']:.4f} |"
            )
    cost = results["measured_cost_usd"]
    lines.append(f"\nMeasured sample cost, files used: ${cost['total']:.4f}.\n")
    for t, r in results["tasks"].items():
        lines.extend(_task_section(t, r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _task_section(t: str, r: dict) -> list[str]:
    out = [f"## {t}\n"]
    out.append(
        f"{r['n_rows']} rows. Metric: {r['metric_kind']}. s = {r['strong']}. "
        "Delta A and D2 in points; accuracy is the mean over splits with the 5-95% range. "
        f"Flag rates over all rows: C4 {_f(r['flags']['C4'])} ({r['flags']['C4_rows']} rows), "
        f"C5 {_f(r['flags']['C5'])} ({r['flags']['C5_rows']} rows).\n"
    )
    orig = r["golds"]["original"]
    out.append("### Confirmatory policies (original gold)\n")
    out.append(
        "| Policy | $/1k | % of s | accuracy | Delta A (LB) | D2 (LB) | routed | "
        "adjudicated acc | Delta A, length | D2, length |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for pol in POLICIES:
        s_, c = orig["confirmatory"][pol], r["confirmatory"][pol]
        out.append(
            f"| {pol} | {_f(s_['cost_flat']['mean'], 3)} "
            f"| {_f(100 * s_['pct_of_s_flat']['mean'], 0)}% "
            f"| {replay._range(s_['acc'])} "
            f"| {_pts(c['delta_flat'])} ({_pts(c['delta_lb'])}) "
            f"| {_pts(c['d2_flat'])} ({_pts(c['d2_lb'])}) "
            f"| {_f(s_['routed_share']['mean'], 3)} "
            f"| {_f(c.get('acc_adjudicated'))} "
            f"| {_pts(c['delta_length'])} | {_pts(c['d2_length'])} |"
        )
    out.append("")
    flips = [pol for pol in POLICIES if r["confirmatory"][pol].get("sign_flip")]
    if flips:
        out.append(
            "FLAG: Delta A changes sign on the adjudicated gold for "
            + ", ".join(flips)
            + " (not a kill).\n"
        )
    p = r["paired"]
    out.append(
        f"C5 against C4 (Delta A of C5 minus C4): flat {_pts(p['diff_flat'])}, "
        f"length {_pts(p['diff_length'])}, lower bound (flat) {_pts(p['diff_lb'])}.\n"
    )
    d = r["diagnostic"]
    out.append("### Mechanism diagnostic (flag = C4's trigger, all rows)\n")
    label = {"intent": ("R33", "R44", "FF")}.get(t, ("R_blind", "R_wrong", "FF"))
    out.append(f"- {label[0]}, flagged share of the blind-spot rows: {_ci(d['blind'])}")
    out.append(f"- {label[1]}, of Haiku's wrong rows: {_ci(d['wrong'])}")
    out.append(f"- {label[2]}, of Haiku's right rows: {_ci(d['right'])}")
    out.append(
        f"- Class: **{d['class']}**"
        + ("" if t == "intent" else " (descriptive only)")
        + "\n"
    )
    nc = r.get("noise_control")
    if nc:
        g = nc["groups"]
        out.append("### Label-noise control (adjudicated gold)\n")
        out.append(
            f"- Haiku wrong rows flagged: {_ci(nc['wrong'])}; right rows: {_ci(nc['right'])}"
        )
        out.append(f"- A (still wrong): {_ci(g['A'])}")
        out.append(f"- B (now right): {_ci(g['B'])}")
        out.append(f"- C (no defensible label, dropped): {_ci(g['C'])}")
        out.append(f"- B and C together: {_ci(g['B_or_C'])}")
        out.append(
            f"- flag(A) >= 0.40, finds real errors: {nc['finds_real_errors']}; "
            f"flag(B u C) - flag(A) = {_f(nc['gap_b_or_c_minus_a'])}, tracks ambiguity "
            f"(>= 0.20): {nc['tracks_ambiguity']}\n"
        )
    out.append("### Exploratory (not used for the verdict)\n")
    out.append("| Config | $/1k | accuracy | Delta A | D2 | routed |")
    out.append("|---|---|---|---|---|---|")
    for cid, g in orig["exploratory"].items():
        sm = g["summary"]
        out.append(
            f"| {cid} | {_f(sm['cost_flat']['mean'], 3)} | {_f(sm['acc']['mean'])} "
            f"| {_pts(sm['delta_flat']['mean'])} | {_pts(sm['d2_flat']['mean'])} "
            f"| {_f(sm['routed_share']['mean'], 3)} |"
        )
    out.append("")
    return out


def write_results(results: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "addendum-1.md", out_dir / "addendum-1.json"]
    write_markdown(results, paths[0])
    paths[1].write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="sample checks only")
    parser.add_argument("--quick", action="store_true", help="B = 100 smoke run")
    parser.add_argument("--out", type=Path, help="results directory")
    args = parser.parse_args(argv)

    parent = replay.check_integrity()
    if not parent.ok:
        print("\n".join(parent.lines))
        print(f"\nPARENT INTEGRITY FAILED ({len(parent.problems)} problems)")
        return 2
    print("parent integrity (replay.py --check): OK")
    checks = sample_checks()
    print("\n".join(checks.lines))
    if not checks.hard_ok:
        print(f"\nCHECK FAILED ({len(checks.problems)} problems, listed above as FAIL)")
        if any("missing" in p for p in checks.problems):
            print(
                "Draw the samples first: uv run python benchmarks/golden/label.py "
                "--task <task> --model haiku --sample <1-3> --route openrouter"
            )
        return 2
    if checks.v123:
        n = sum(len(v) for v in checks.v123.values())
        print(f"\nV1-V3 FAILED: {n} failures on {', '.join(sorted(checks.v123))}")
        if args.check:
            return 2
        if "intent" in checks.v123:
            print(
                "The intent verdict will be INCONCLUSIVE; the study runs to report it."
            )
    else:
        print("\nCHECK OK")
    if args.check:
        return 0

    tasks = {
        name: {
            g: (
                replay.load_task(name, g),
                load_samples(name, g, checks.selected[name]),
            )
            for g in replay.GOLDS
        }
        for name in PREREG_ADDENDUM["run_tasks"]
    }
    results = run_study(tasks, checks, quick=args.quick)
    out_dir = args.out or RESULTS / ("quick" if args.quick else "")
    for p in write_results(results, out_dir):
        print(f"wrote {p.relative_to(REPO) if p.is_relative_to(REPO) else p}")
    v = results["verdict"]
    print(f"VERDICT: {v['verdict']} ({v['scope']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
