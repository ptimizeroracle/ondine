"""Planner replay: does routing beat "use the cheapest model that is good enough"?

Pre-registered offline experiment (design: ``benchmarks/planner/DESIGN.md``).
It replays five models' recorded answers on the golden dev rows, so it makes no
API call. Every constant lives in ``PREREG``; the results embed its hash and the
hash of the design file. Do not change a constant after seeing a result.

    uv run python benchmarks/planner/replay.py --check    # integrity only
    uv run python benchmarks/planner/replay.py --quick    # B = 100, smoke run
    uv run python benchmarks/planner/replay.py            # S = 1000, B = 1000

``--check`` loads the data and proves the loading reproduces
``benchmarks/golden/data/baseline-report.json``. It never evaluates a policy.
The study (default and ``--quick``) writes to ``benchmarks/planner/results/``
(``results/quick/`` for ``--quick``). Exit code 2 means an integrity failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GOLDEN = REPO / "benchmarks" / "golden"
DATA = GOLDEN / "data"
DESIGN = REPO / "benchmarks" / "planner" / "DESIGN.md"
sys.path.insert(0, str(GOLDEN))
import score  # noqa: E402
from adjudicate import (  # noqa: E402
    attribute_errors,
    final_answer,
    norm,
    parse_json_object,
)

TASKS = ("intent", "faithfulness", "attributes")
MODELS = ("haiku", "luna", "sonnet", "sol", "opus")
GOLDS = ("original", "adjudicated")
POLICIES = ("C1", "C2", "C3")
COST_KINDS = ("flat", "length")
FAITHFULNESS_VALUES = ("consistent", "inconsistent")

#: Every constant of the design, in one place. Its sha256 goes into the results.
PREREG = {
    "design": "benchmarks/planner/DESIGN.md",
    "seed": 20261117,
    "tasks": list(TASKS),
    "models": list(MODELS),
    "cheap": "haiku",
    "partner": "luna",
    # Best single model on the original gold in baseline-report.json (design s 2).
    "strong": {"intent": "opus", "faithfulness": "sol", "attributes": "sonnet"},
    "metric": {
        "intent": "accuracy",
        "faithfulness": "balanced",
        "attributes": "micro",
    },
    "shrinkage_m": 4,
    "tau": 0.8,
    "n_labelled": 100,
    "splits": 1000,
    "bootstrap": 1000,
    "bootstrap_quick": 100,
    "bootstrap_splits_per_draw": 20,
    "alpha": 0.05,
    "n_confirmatory": 3,
    "hull_cost_multiple": 2.0,
    "grid": {
        "c1_strong": ["sonnet", "sol", "opus"],
        "c2_cheap": ["haiku", "luna"],
        "strong": ["sonnet", "sol", "opus"],
        "n_labelled": [50, 100],
        "tau": [0.6, 0.7, 0.8, 0.9],
    },
    "pass": {
        "lift_delta_pts": 3.0,
        "cut_d2_mean_pts": 0.0,
        "cut_d2_lb_pts": -2.0,
        "cut_delta_pts": 1.0,
    },
    "verdict": {"beat_c1_pts": 2.0},
    "gold_for_verdict": "original",
}


def prereg_hash(prereg: dict = PREREG) -> str:
    return hashlib.sha256(json.dumps(prereg, sort_keys=True).encode()).hexdigest()


def design_hash(path: Path = DESIGN) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


# ── loading, behind the guard ────────────────────────────────────────────


_DEV_FILES = ("dev.jsonl", "dev-adjudicated.jsonl")
#: Extra Haiku samples drawn for Addendum 1 (``haiku-s1`` to ``haiku-s9``).
_SAMPLE_STEM = re.compile(r"haiku-s[1-9]")


def guard_path(path: Path | str, data_dir: Path = DATA) -> Path:
    """Return `path` if the study may read it, else raise.

    Allowed: ``<task>/dev.jsonl``, ``<task>/dev-adjudicated.jsonl``,
    ``labels/<task>/<model>.jsonl`` and ``.cost.json`` (also for the extra
    Haiku samples ``haiku-s1`` to ``haiku-s9``), the intent label names, and
    the baseline report. The held-out split is never allowed.
    """
    resolved = Path(path).resolve()
    try:
        rel = resolved.relative_to(Path(data_dir).resolve())
    except ValueError:
        raise PermissionError(f"{path} is outside the golden data directory")
    parts = rel.parts
    if "holdout" in rel.as_posix().lower():
        raise PermissionError(f"{path}: the held-out split is off limits to this study")
    ok = (
        (len(parts) == 2 and parts[0] in TASKS and parts[1] in _DEV_FILES)
        or (
            len(parts) == 3
            and parts[0] == "labels"
            and parts[1] in TASKS
            and (
                parts[2].split(".")[0] in MODELS
                or _SAMPLE_STEM.fullmatch(parts[2].split(".")[0])
            )
            and parts[2].endswith((".jsonl", ".cost.json"))
        )
        or parts in (("intent", "labels.json"), ("baseline-report.json",))
    )
    if not ok:
        raise PermissionError(f"{path}: not one of the files this study may read")
    return resolved


def guard_selftest(data_dir: Path = DATA) -> list[str]:
    """Names of held-out or unlisted files the guard wrongly lets through."""
    leaked = []
    for bad in (
        data_dir / "intent" / "holdout.jsonl",
        data_dir / "labels" / "intent" / "holdout.jsonl",
        data_dir / "attributes" / "schemas.json",
    ):
        try:
            guard_path(bad, data_dir)
            leaked.append(bad.name)
        except PermissionError:
            pass
    return leaked


def read_jsonl(path: Path, data_dir: Path = DATA, missing_ok: bool = False) -> list:
    path = guard_path(path, data_dir)
    if missing_ok and not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_json(path: Path, data_dir: Path = DATA):
    return json.loads(guard_path(path, data_dir).read_text(encoding="utf-8"))


# ── the task container ───────────────────────────────────────────────────


@dataclass
class Task:
    """One task on one gold, as arrays.

    Rows are the gold file's rows. Per-field arrays are padded to ``nf``
    columns (``mask`` marks real fields); classification tasks have nf = 1.
    ``correct[m]`` is the primary right/wrong per field, ``value[m]`` the
    delivered value as a shared integer code, ``cell[m]`` the code the value
    router counts (the delivered value; for attributes the field cell).
    """

    name: str
    gold_kind: str
    metric_kind: str
    ids: list[str]
    mask: np.ndarray
    gold_class: np.ndarray
    value: dict[str, np.ndarray]
    cell: dict[str, np.ndarray]
    correct: dict[str, np.ndarray]
    strict: dict[str, np.ndarray]
    valid: dict[str, np.ndarray]
    u_flat: dict[str, float]
    u_len: dict[str, np.ndarray]
    n_cells: int
    models: tuple[str, ...] = MODELS
    same_as_original: bool = False
    _fit: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return self.mask.shape[0]

    @property
    def nf(self) -> int:
        return self.mask.shape[1]

    def unit(self, kind: str, model: str):
        """Cost of asking `model` on one row: a float (flat) or an array."""
        return self.u_flat[model] if kind == "flat" else self.u_len[model]

    def fit_tables(self, model: str):
        """Per-row counts the value router fits on, cached.

        Only fields whose value is valid count toward a value's precision
        (an invalid output has no delivered value); every labelled field
        counts toward the overall precision p-bar.
        """
        if model not in self._fit:
            fit = self.mask & self.valid[model][:, None]
            n_row = np.zeros((self.n, self.n_cells))
            r_row = np.zeros((self.n, self.n_cells))
            rows = np.arange(self.n)
            for f in range(self.nf):
                np.add.at(n_row, (rows, self.cell[model][:, f]), fit[:, f])
                np.add.at(
                    r_row,
                    (rows, self.cell[model][:, f]),
                    fit[:, f] & self.correct[model][:, f],
                )
            self._fit[model] = (
                n_row,
                r_row,
                (self.correct[model] & self.mask).sum(1).astype(float),
                self.mask.sum(1).astype(float),
            )
        return self._fit[model]


def make_task(
    name: str,
    gold_kind: str,
    metric_kind: str,
    ids: list[str],
    mask: np.ndarray,
    gold_class: np.ndarray,
    value: dict[str, np.ndarray],
    cell: dict[str, np.ndarray],
    correct: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    cost: dict[str, tuple[float, int]],
    lens: np.ndarray,
    len_total: float,
    strict: dict[str, np.ndarray] | None = None,
    n_cells: int | None = None,
    same_as_original: bool = False,
) -> Task:
    """Assemble a Task. `cost[m]` = (run cost in $, rows in the run)."""
    lens = np.asarray(lens, float)
    return Task(
        name=name,
        gold_kind=gold_kind,
        metric_kind=metric_kind,
        ids=list(ids),
        mask=mask,
        gold_class=np.asarray(gold_class, int),
        value=value,
        cell=cell,
        correct=correct,
        strict=strict if strict is not None else correct,
        valid=valid,
        u_flat={m: cost[m][0] / cost[m][1] for m in cost},
        u_len={m: cost[m][0] * lens / len_total for m in cost},
        n_cells=int(
            n_cells if n_cells is not None else max(c.max() for c in cell.values()) + 1
        ),
        models=tuple(cost),
        same_as_original=same_as_original,
    )


def _input_chars(task: str, row: dict) -> int:
    """Characters of the prompt inputs of one row (the columns label.py sends)."""
    if task == "intent":
        return len(row["text"])
    if task == "faithfulness":
        return len(row["doc"]) + len(row["summary"])
    return (
        len(row["category"])
        + len(row["title"])
        + len(row["description"])
        + len(", ".join(sorted(row["target"])))
    )


def load_task(task: str, gold: str = "original", data_dir: Path = DATA) -> Task:
    """Load one task on one gold. Reads only what ``guard_path`` allows."""
    dev = read_jsonl(data_dir / task / "dev.jsonl", data_dir)
    adjudicated = read_jsonl(
        data_dir / task / "dev-adjudicated.jsonl", data_dir, missing_ok=True
    )
    # score.py falls back to the original gold when there is no adjudicated file.
    rows = dev if gold == "original" else (adjudicated or dev)
    answers, cost = {}, {}
    for m in MODELS:
        answers[m] = {
            r["id"]: r["answer"]
            for r in read_jsonl(data_dir / "labels" / task / f"{m}.jsonl", data_dir)
        }
        c = read_json(data_dir / "labels" / task / f"{m}.cost.json", data_dir)
        cost[m] = (float(c["cost"]), int(c["rows"]))
    lens = np.array([_input_chars(task, r) for r in rows], float)
    len_total = float(sum(_input_chars(task, r) for r in dev))
    common = dict(
        gold_kind=gold,
        ids=[r["id"] for r in rows],
        cost=cost,
        lens=lens,
        len_total=len_total,
        same_as_original=gold == "adjudicated" and not adjudicated,
    )
    if task == "attributes":
        return _load_attributes(rows, answers, common)
    return _load_classification(task, rows, answers, common, data_dir)


def _load_classification(task, rows, answers, common, data_dir) -> Task:
    if task == "intent":
        allowed = {
            norm(x) for x in read_json(data_dir / "intent" / "labels.json", data_dir)
        }
    else:
        allowed = {norm(x) for x in FAITHFULNESS_VALUES}
    vocab: dict[str, int] = {}
    gold_norm = [norm(r["label"]) for r in rows]
    classes = sorted(set(gold_norm))
    value, correct, valid = {}, {}, {}
    for m in MODELS:
        # Same expression as score.score, so a null answer counts the same way.
        finals = [norm(final_answer(answers[m].get(r["id"]))) for r in rows]
        value[m] = np.array([vocab.setdefault(f, len(vocab)) for f in finals])[:, None]
        correct[m] = np.array([f == g for f, g in zip(finals, gold_norm, strict=True)])[
            :, None
        ]
        valid[m] = np.array([f in allowed for f in finals])
    return make_task(
        name=task,
        metric_kind=PREREG["metric"][task],
        mask=np.ones((len(rows), 1), bool),
        gold_class=np.array([classes.index(g) for g in gold_norm])
        if task == "faithfulness"
        else np.zeros(len(rows), int),
        value=value,
        cell={m: value[m] for m in MODELS},
        correct=correct,
        valid=valid,
        n_cells=len(vocab),
        **common,
    )


def _load_attributes(rows, answers, common) -> Task:
    n = len(rows)
    nf = max(len(r["target"]) for r in rows)
    mask = np.zeros((n, nf), bool)
    for i, r in enumerate(rows):
        mask[i, : len(r["target"])] = True
    vocab: dict[str, int] = {}
    cells: dict[tuple, int] = {}
    value = {m: np.full((n, nf), -1) for m in MODELS}
    cell = {m: np.zeros((n, nf), int) for m in MODELS}
    correct = {m: np.zeros((n, nf), bool) for m in MODELS}
    strict = {m: np.zeros((n, nf), bool) for m in MODELS}
    valid = {m: np.zeros(n, bool) for m in MODELS}
    for m in MODELS:
        for i, r in enumerate(rows):
            answer = answers[m].get(r["id"])
            parsed = parse_json_object(answer)
            valid[m][i] = parsed is not None
            wrong = set(attribute_errors(r["target"], answer))
            wrong_strict = set(score.strict_attribute_errors(r["target"], answer))
            for f, attribute in enumerate(r["target"]):
                correct[m][i, f] = attribute not in wrong
                strict[m][i, f] = attribute not in wrong_strict
                if parsed is None:
                    continue
                got = norm(parsed.get(attribute, "n/a"))
                value[m][i, f] = vocab.setdefault(got, len(vocab))
                key = (r["category"], attribute, got == "na")
                cell[m][i, f] = cells.setdefault(key, len(cells))
    return make_task(
        name="attributes",
        metric_kind=PREREG["metric"]["attributes"],
        mask=mask,
        gold_class=np.zeros(n, int),
        value=value,
        cell=cell,
        correct=correct,
        strict=strict,
        valid=valid,
        n_cells=len(cells),
        **common,
    )


# ── estimators ───────────────────────────────────────────────────────────


def shrunk_precision(codes_l, right_l, m, n_codes=None, weights=None):
    """Shrunk precision of each delivered value, from labelled rows.

    ``p(v) = (k_v + m * p_bar) / (n_v + m)``: k_v right, n_v seen, p_bar the
    overall precision on the labelled rows. A negative code is an output with
    no valid value: it counts toward p_bar only. Returns (table, p_bar);
    an unseen value gets p_bar.
    """
    codes = np.asarray(codes_l, int)
    right = np.asarray(right_l, bool)
    w = np.ones(len(codes)) if weights is None else np.asarray(weights, float)
    if n_codes is None:
        n_codes = int(codes.max()) + 1 if codes.size and codes.max() >= 0 else 0
    seen = codes >= 0
    n_v = np.bincount(codes[seen], weights=w[seen], minlength=n_codes)
    k_v = np.bincount(codes[seen], weights=(w * right)[seen], minlength=n_codes)
    p_bar = float((w * right).sum() / w.sum())
    return (k_v + m * p_bar) / (n_v + m), p_bar


def _fit_precision(task: Task, model: str, train_w: np.ndarray, m: float):
    """`shrunk_precision` for every split at once: (S, n_cells) and (S,)."""
    n_row, r_row, right_rows, field_rows = task.fit_tables(model)
    n_v = train_w @ n_row
    k_v = train_w @ r_row
    p_bar = (train_w @ right_rows) / (train_w @ field_rows)
    return (k_v + m * p_bar[:, None]) / (n_v + m), p_bar


def metric(task: Task, right, gold, weights):
    """The task's primary metric over weighted rows, one value per leading split.

    `right` is (S, n, nf) or (n, nf); `weights` is (S, n) or (n,). intent:
    accuracy. faithfulness: balanced accuracy over the gold classes present.
    attributes: lenient attribute-level accuracy, micro over fields.
    """
    w = np.asarray(weights, float)
    right = np.asarray(right, bool)
    if task.metric_kind == "micro":
        rows_right = (right & task.mask).sum(-1)
        return (w * rows_right).sum(-1) / (w * task.mask.sum(1)).sum(-1)
    hit = right[..., 0]
    if task.metric_kind == "accuracy":
        return (w * hit).sum(-1) / w.sum(-1)
    gold = np.asarray(gold, int)
    classes = np.eye(int(gold.max()) + 1)[gold]
    num = (w * hit) @ classes
    den = w @ classes
    present = den > 0
    recall = np.divide(num, den, out=np.zeros_like(num), where=present)
    return recall.sum(-1) / present.sum(-1)


def hull_value(xs: np.ndarray, ys: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Best reachable metric at expected cost `c`, per split.

    The single-model hull: the best of any one model that costs no more than
    `c`, and any mix of two models whose expected cost is `c` (the upper
    concave envelope, made non-decreasing). `xs`, `ys` are (S, k) cost and
    metric of the k models; `c` is (S,). Below the cheapest model the curve
    is held flat at the cheapest model's value, above the dearest at the
    best value: both generous to the comparator. Costs match to 1e-9
    relative, which can only raise the hull.
    """
    c = np.asarray(c, float)[:, None]
    c = np.maximum(c, xs.min(1, keepdims=True))
    tol = 1e-9 * np.maximum(np.abs(c), 1e-300)
    best = np.where(xs <= c + tol, ys, -np.inf).max(1)
    k = xs.shape[1]
    for i in range(k):
        for j in range(k):
            xi, xj, yi, yj = xs[:, i : i + 1], xs[:, j : j + 1], ys[:, i], ys[:, j]
            use = (xi < xj) & (xi <= c + tol)
            t = np.clip((c - xi) / np.where(use, xj - xi, 1.0), 0.0, 1.0)[:, 0]
            chord = yi + (yj - yi) * t
            best = np.maximum(best, np.where(use[:, 0], chord, -np.inf))
    return best


def hull(points):
    """The hull of (cost, metric) points as a function F(cost) -> metric."""
    pts = np.asarray(points, float)

    def f(c):
        c = np.atleast_1d(np.asarray(c, float))
        xs = np.broadcast_to(pts[:, 0], (len(c), len(pts)))
        ys = np.broadcast_to(pts[:, 1], (len(c), len(pts)))
        return hull_value(xs, ys, c)

    return f


# ── splits ───────────────────────────────────────────────────────────────


def split_rng(prereg: dict, task: str, gold: str, purpose: int, n_l: int):
    """A generator fixed by the design's seed and the stream's identity."""
    return np.random.default_rng(
        [
            prereg["seed"],
            prereg["tasks"].index(task) if task in prereg["tasks"] else 99,
            GOLDS.index(gold),
            purpose,
            n_l,
        ]
    )


def draw_splits(rng, n: int, n_l: int, n_splits: int) -> np.ndarray:
    """(n_splits, n) bool, True on `n_l` rows drawn uniformly without replacement."""
    r = rng.random((n_splits, n))
    idx = np.argpartition(r, n_l - 1, axis=1)[:, :n_l]
    train = np.zeros((n_splits, n), bool)
    np.put_along_axis(train, idx, True, axis=1)
    return train


@dataclass
class ModelPoints:
    """Each model's (cost, metric) on the test rows of each split."""

    y: np.ndarray
    x: dict[str, np.ndarray]


def model_points(task: Task, test_w: np.ndarray) -> ModelPoints:
    wsum = test_w.sum(1)
    y = np.stack(
        [
            metric(task, task.correct[m][None], task.gold_class, test_w)
            for m in task.models
        ],
        axis=1,
    )
    flat = np.tile([task.u_flat[m] for m in task.models], (len(test_w), 1))
    length = np.stack(
        [(test_w * task.u_len[m]).sum(1) / wsum for m in task.models], axis=1
    )
    return ModelPoints(y=y, x={"flat": flat, "length": length})


# ── policies: (task, train_w, test_w, params) -> (right, cost, routed) ───
#
# `train_w` and `test_w` are (S, n): the weight of each row on the labelled
# side and on the test side of each split (0/1 for plain splits, bootstrap
# counts otherwise). Policies return arrays over all n rows; callers weight
# them by `test_w`. `right` and `routed` are (S|1, n, nf) bool, `cost` maps a
# cost model to (S|1, n) $ (or a scalar). A policy reads labels only through
# `train_w`.


def _cost(task: Task, always: tuple, strong: str | None, row_routed) -> dict:
    out = {}
    for kind in COST_KINDS:
        base = sum(task.unit(kind, m) for m in always)
        out[kind] = base + (row_routed * task.unit(kind, strong) if strong else 0.0)
    return out


def _deliver(task: Task, routed, cheap: str, strong: str, params: dict):
    if params.get("gran") == "row":
        routed = routed.any(-1, keepdims=True) & task.mask
    right = np.where(routed, task.correct[strong], task.correct[cheap])
    return right, routed


def single(task, train_w, test_w, params):
    """Ask one model on every row."""
    m = params["model"]
    routed = np.zeros((1, task.n, task.nf), bool)
    cost = _cost(task, (m,), None, np.zeros((1, task.n)))
    return task.correct[m][None], cost, routed


def cascade_agree(task, train_w, test_w, params):
    """C1: ask the cheap pair; if they agree deliver the cheap answer, else ask s."""
    cheap, partner, strong = params["cheap"], params["partner"], params["strong"]
    agree = (
        (task.value[cheap] == task.value[partner])
        & task.valid[cheap][:, None]
        & task.valid[partner][:, None]
    )
    routed = (task.mask & ~agree)[None]
    right, routed = _deliver(task, routed, cheap, strong, params)
    cost = _cost(task, (cheap, partner), strong, routed.any(-1))
    return right, cost, routed


def _value_routed(task, train_w, params):
    cheap, tau, m = params["cheap"], params["tau"], params["m"]
    phat, _ = _fit_precision(task, cheap, train_w, m)
    low = (phat < tau)[:, task.cell[cheap]]  # same as comparing after the gather
    invalid = ~task.valid[cheap][:, None]
    return task.mask[None] & (low | invalid[None])


def value_router(task, train_w, test_w, params):
    """C2: ask the cheap model; route values whose shrunk precision is < tau."""
    cheap, strong = params["cheap"], params["strong"]
    routed = _value_routed(task, train_w, params)
    right, routed = _deliver(task, routed, cheap, strong, params)
    cost = _cost(task, (cheap,), strong, routed.any(-1))
    return right, cost, routed


def agree_or_value(task, train_w, test_w, params):
    """C3: route on disagreement of the cheap pair OR a low-precision value."""
    cheap, partner, strong = params["cheap"], params["partner"], params["strong"]
    agree = (
        (task.value[cheap] == task.value[partner])
        & task.valid[cheap][:, None]
        & task.valid[partner][:, None]
    )
    routed = _value_routed(task, train_w, params) | (task.mask & ~agree)[None]
    right, routed = _deliver(task, routed, cheap, strong, params)
    cost = _cost(task, (cheap, partner), strong, routed.any(-1))
    return right, cost, routed


def oracle_cheapest_right(task, train_w, test_w, params):
    """O1 (unreachable): per field the cheapest right model; pay the models used.

    A field no model gets right is delivered by the cheapest model. A row
    pays once for each distinct model its fields use.
    """
    order = sorted(task.models, key=lambda m: task.u_flat[m])
    chosen = np.zeros((task.n, task.nf), int)
    found = np.zeros((task.n, task.nf), bool)
    for i, m in enumerate(order):
        hit = task.correct[m] & ~found
        chosen[hit] = i
        found |= hit
    used = np.stack(
        [((chosen == i) & task.mask).any(1) for i in range(len(order))], axis=1
    )
    cost = {
        "flat": used @ np.array([task.u_flat[m] for m in order]),
        "length": (used * np.stack([task.u_len[m] for m in order], axis=1)).sum(1),
    }
    cost = {k: v[None] for k, v in cost.items()}
    return found[None], cost, ((chosen > 0) & task.mask)[None]


def oracle_in_sample(task, train_w, test_w, params):
    """O3 (unreachable): C2 with the precision fitted on every row, test rows too."""
    return value_router(task, np.ones_like(test_w), test_w, params)


POLICY_FNS = {
    "single": single,
    "C1": cascade_agree,
    "C2": value_router,
    "C3": agree_or_value,
    "O1": oracle_cheapest_right,
    "O3": oracle_in_sample,
}


# ── evaluation of an outcome ─────────────────────────────────────────────

#: Keys of an evaluation reported in points (x100), and costs in $ per 1k rows.
POINT_KEYS = ("delta_flat", "delta_length", "d2_flat", "d2_length")


def evaluate(
    task: Task, outcome, test_w, pts: ModelPoints, params: dict, strong: str, full=True
) -> dict[str, np.ndarray]:
    """Per-split metrics of a policy outcome on the test rows."""
    right, cost, routed = outcome
    wsum = test_w.sum(1)
    acc = metric(task, right, task.gold_class, test_w)
    out = {"acc": acc}
    s_idx = task.models.index(strong)
    for kind in COST_KINDS:
        c = (test_w * cost[kind]).sum(1) / wsum
        out[f"cost_{kind}"] = c
        out[f"hull_{kind}"] = hull_value(pts.x[kind], pts.y, c)
        out[f"delta_{kind}"] = acc - out[f"hull_{kind}"]
        hull2 = hull_value(pts.x[kind], pts.y, PREREG["hull_cost_multiple"] * c)
        out[f"d2_{kind}"] = acc - hull2
        out[f"pct_of_s_{kind}"] = c / pts.x[kind][:, s_idx]
    if not full:
        return out
    n_routed_fields = routed.sum(-1)
    total_fields = (test_w * task.mask.sum(1)).sum(1)
    out["routed_share"] = (test_w * n_routed_fields).sum(1) / total_fields
    out["routed_row_share"] = (test_w * (n_routed_fields > 0)).sum(1) / wsum
    cheap = params.get("cheap")
    if cheap in task.correct:
        wrong = (~task.correct[cheap]) & task.mask
        n_wrong = (routed & wrong).sum(-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["routed_cheap_wrong"] = (test_w * n_wrong).sum(1) / (
                test_w * n_routed_fields
            ).sum(1)
    success = (right | ~task.mask).all(-1)
    n_success = (test_w * success).sum(1)
    out["success_rate"] = n_success / wsum
    with np.errstate(invalid="ignore", divide="ignore"):
        out["cost_per_success_flat"] = (test_w * cost["flat"]).sum(1) / n_success
    if task.metric_kind == "micro" and cheap in task.strict and "strong" in params:
        right_strict = np.where(routed, task.strict[strong], task.strict[cheap])
        out["acc_strict"] = metric(task, right_strict, task.gold_class, test_w)
    return out


def _clean(x):
    """JSON-safe number (NaN and inf become None)."""
    x = float(x)
    return x if np.isfinite(x) else None


def summarise(ev: dict[str, np.ndarray]) -> dict:
    """Mean and 5-95% range over splits of each evaluated quantity."""
    out = {}
    for key, arr in ev.items():
        x = np.asarray(arr, float)
        scale = 100.0 if key in POINT_KEYS else 1.0
        if key.startswith("cost_") and not key.startswith("cost_per"):
            scale = 1000.0
        ok = x[np.isfinite(x)] * scale
        out[key] = (
            {
                "mean": _clean(ok.mean()),
                "p05": _clean(np.percentile(ok, 5)),
                "p95": _clean(np.percentile(ok, 95)),
            }
            if ok.size
            else {"mean": None, "p05": None, "p95": None}
        )
    return out


# ── cluster bootstrap ────────────────────────────────────────────────────


def bootstrap_batches(rng, n: int, n_l: int, n_draws: int, per_draw: int, chunk: int):
    """Yield (train_w, test_w, n_draws_in_chunk) for the cluster bootstrap.

    Each draw gives multinomial weights over the n unique ids (a row drawn
    twice has weight 2). Each draw then makes `per_draw` splits of the unique
    ids: `n_l` ids on the labelled side, the rest on the test side, so copies
    of a row are never on both sides. The weights apply on both sides.
    """
    per_chunk = max(1, chunk // per_draw)
    done = 0
    while done < n_draws:
        d = min(per_chunk, n_draws - done)
        w = rng.multinomial(n, np.full(n, 1.0 / n), size=d).astype(float)
        train = draw_splits(rng, n, n_l, d * per_draw)
        weights = np.repeat(w, per_draw, axis=0)
        yield weights * train, weights * ~train, d
        done += d


def cluster_bootstrap(task, configs, prereg, rng, n_draws, strong, chunk=1000):
    """Draw-level mean delta and D2 (flat cost), and paired deltas against C1.

    `configs` maps a name to (policy function, params). Every policy sees
    the same weights and splits in every draw. Returns name -> {"delta",
    "d2"} arrays of length `n_draws`, in points.
    """
    per_draw = prereg["bootstrap_splits_per_draw"]
    out = {name: {"delta": [], "d2": []} for name in configs}
    for train_w, test_w, d in bootstrap_batches(
        rng, task.n, prereg["n_labelled"], n_draws, per_draw, chunk
    ):
        pts = model_points(task, test_w)
        for name, (fn, params) in configs.items():
            ev = evaluate(
                task,
                fn(task, train_w, test_w, params),
                test_w,
                pts,
                params,
                strong,
                full=False,
            )
            out[name]["delta"].append(ev["delta_flat"].reshape(d, per_draw).mean(1))
            out[name]["d2"].append(ev["d2_flat"].reshape(d, per_draw).mean(1))
    return {
        name: {k: 100.0 * np.concatenate(v) for k, v in vals.items()}
        for name, vals in out.items()
    }


def lower_bound(draws: np.ndarray, prereg: dict) -> float:
    """One-sided Bonferroni lower bound: the alpha / 3 quantile of the draws."""
    return float(np.quantile(draws, prereg["alpha"] / prereg["n_confirmatory"]))


# ── the study ────────────────────────────────────────────────────────────


def confirmatory_params(task_name: str, prereg: dict) -> dict[str, dict]:
    cheap, partner = prereg["cheap"], prereg["partner"]
    s = prereg["strong"][task_name]
    base = {"cheap": cheap, "strong": s, "m": prereg["shrinkage_m"]}
    return {
        "C1": {**base, "partner": partner},
        "C2": {**base, "tau": prereg["tau"], "n_l": prereg["n_labelled"]},
        "C3": {
            **base,
            "partner": partner,
            "tau": prereg["tau"],
            "n_l": prereg["n_labelled"],
        },
    }


def config_id(policy: str, p: dict) -> str:
    gran = ":row" if p.get("gran") == "row" else ""
    if policy == "C1":
        return f"C1:{p['strong']}{gran}"
    return f"{policy}:{p['cheap']}:{p['strong']}:n{p['n_l']}:t{p['tau']}{gran}"


def grid_configs(task_name: str, prereg: dict) -> list[tuple[str, dict]]:
    """The exploratory grid (75 configs per task) plus, for attributes, the three
    whole-row variants of the confirmatory configs (design section 5)."""
    g, m = prereg["grid"], prereg["shrinkage_m"]
    cheap, partner = prereg["cheap"], prereg["partner"]
    n_conf = prereg["n_labelled"]
    out = []
    for s in g["c1_strong"]:
        out.append(("C1", {"cheap": cheap, "partner": partner, "strong": s, "m": m}))
    for c in g["c2_cheap"]:
        for s in g["strong"]:
            for n_l in g["n_labelled"]:
                for tau in g["tau"]:
                    out.append(
                        (
                            "C2",
                            {"cheap": c, "strong": s, "m": m, "tau": tau, "n_l": n_l},
                        )
                    )
    for s in g["strong"]:
        for n_l in g["n_labelled"]:
            for tau in g["tau"]:
                p = {"cheap": cheap, "partner": partner, "strong": s, "m": m}
                out.append(("C3", {**p, "tau": tau, "n_l": n_l}))
    if task_name == "attributes":
        for pol, p in confirmatory_params(task_name, prereg).items():
            out.append((pol, {**p, "gran": "row", "n_l": n_conf}))
    return [(pol, {"n_l": n_conf, **p}) for pol, p in out]


class _SplitCache:
    """The S splits and the model points for each labelled-set size, per task+gold."""

    def __init__(self, task: Task, prereg: dict):
        self.task, self.prereg, self._cache = task, prereg, {}

    def get(self, n_l: int):
        if n_l not in self._cache:
            t, pr = self.task, self.prereg
            train = draw_splits(
                split_rng(pr, t.name, t.gold_kind, 0, n_l), t.n, n_l, pr["splits"]
            )
            train_w, test_w = train.astype(float), (~train).astype(float)
            self._cache[n_l] = (train_w, test_w, model_points(t, test_w))
        return self._cache[n_l]


def run_splits(task, cache, policy, params, strong, full=True):
    train_w, test_w, pts = cache.get(params.get("n_l", cache.prereg["n_labelled"]))
    outcome = POLICY_FNS[policy](task, train_w, test_w, params)
    return evaluate(task, outcome, test_w, pts, params, strong, full)


def diagnostics(task: Task, cheap: str, partner: str) -> dict:
    """Shared-mistake rate of the cheap pair, over all rows (design section 8)."""
    agree = (
        (task.value[cheap] == task.value[partner])
        & task.valid[cheap][:, None]
        & task.valid[partner][:, None]
    )
    wrong_c = ~task.correct[cheap] & task.mask
    shared = agree & wrong_c & ~task.correct[partner]
    ones = np.ones(task.n)
    cap = metric(task, ~shared & task.mask, task.gold_class, ones)
    return {
        "units": int(task.mask.sum()),
        "cheap_wrong": int(wrong_c.sum()),
        "agree_and_wrong": int(shared.sum()),
        "agree_wrong_rate": _clean(shared.sum() / task.mask.sum()),
        "share_of_cheap_wrong": _clean(shared.sum() / max(wrong_c.sum(), 1)),
        "c1_ceiling": _clean(cap),
    }


def run_gold(task: Task, prereg: dict, full_grid: bool, log=print) -> dict:
    """Evaluate everything on one gold (point estimates over S splits)."""
    strong = prereg["strong"][task.name]
    cache = _SplitCache(task, prereg)
    conf = confirmatory_params(task.name, prereg)
    out: dict = {"n": task.n, "confirmatory": {}, "grid": {}, "oracle": {}}
    for pol, p in conf.items():
        out["confirmatory"][pol] = summarise(run_splits(task, cache, pol, p, strong))
    n_l = prereg["n_labelled"]
    train_w, test_w, pts = cache.get(n_l)
    o1 = evaluate(
        task, oracle_cheapest_right(task, train_w, test_w, {}), test_w, pts, {}, strong
    )
    out["oracle"]["O1"] = summarise(o1)
    o3_params = {**conf["C2"]}
    out["oracle"]["O3"] = summarise(run_splits(task, cache, "O3", o3_params, strong))
    out["oracle"]["O2_acc"] = out["oracle"]["O1"]["acc"]
    if full_grid:
        for pol, p in grid_configs(task.name, prereg):
            ev = run_splits(task, cache, pol, p, strong)
            out["grid"][config_id(pol, p)] = {
                "policy": pol,
                "params": {k: v for k, v in p.items() if k != "m"},
                "summary": summarise(ev),
            }
    return out


def run_task(golds: dict[str, Task], prereg: dict, n_draws: int, log=print) -> dict:
    task = golds["original"]
    name = task.name
    strong = prereg["strong"][name]
    t0 = time.perf_counter()
    res: dict = {
        "name": name,
        "n_rows": task.n,
        "strong": strong,
        "metric_kind": task.metric_kind,
    }
    full = np.ones(task.n)
    res["models"] = {
        m: {
            "cost_per_1k": 1000.0 * task.u_flat[m],
            "metric": _clean(metric(task, task.correct[m], task.gold_class, full)),
        }
        for m in task.models
    }
    pts_all = np.array(
        [[task.u_flat[m], res["models"][m]["metric"]] for m in task.models]
    )
    res["hull_points"] = pts_all.tolist()
    res["diagnostics"] = diagnostics(task, prereg["cheap"], prereg["partner"])
    res["golds"] = {}
    for gold, t in golds.items():
        if t.same_as_original:
            res["golds"][gold] = {"same_as_original": True}
            continue
        res["golds"][gold] = run_gold(t, prereg, full_grid=gold == "original")
    orig = res["golds"]["original"]

    # Cluster bootstrap on the original gold, flat cost, the 3 confirmatory policies.
    conf = confirmatory_params(name, prereg)
    configs = {pol: (POLICY_FNS[pol], p) for pol, p in conf.items()}
    draws = cluster_bootstrap(
        task,
        configs,
        prereg,
        split_rng(prereg, name, "original", 1, prereg["n_labelled"]),
        n_draws,
        strong,
    )
    paired = {}
    for pol in ("C2", "C3"):
        diff = draws[pol]["delta"] - draws["C1"]["delta"]
        paired[pol] = {
            "diff_lb": lower_bound(diff, prereg),
            "diff_boot_mean": float(diff.mean()),
        }
    res["bootstrap"] = {
        "n_draws": n_draws,
        "alpha_each": prereg["alpha"] / prereg["n_confirmatory"],
        "policies": {
            pol: {
                "delta_lb": lower_bound(d["delta"], prereg),
                "d2_lb": lower_bound(d["d2"], prereg),
                "delta_boot_mean": float(d["delta"].mean()),
                "d2_boot_mean": float(d["d2"].mean()),
            }
            for pol, d in draws.items()
        },
        "paired_vs_c1": paired,
    }
    res["confirmatory"], res["paired"] = _verdict_inputs(res, orig, prereg)
    res["plot"] = _plot_data(task, res, orig, prereg)
    res["seconds"] = time.perf_counter() - t0
    log(f"  {name}: {res['seconds']:.1f}s")
    return res


def _verdict_inputs(res: dict, orig: dict, prereg: dict):
    """The flat numbers `verdict` reads, in points and $ per row."""
    adj = res["golds"].get("adjudicated", {})
    conf, paired = {}, {}
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
    for pol in ("C2", "C3"):
        d = {}
        for kind in COST_KINDS:
            d[f"diff_{kind}"] = conf[pol][f"delta_{kind}"] - conf["C1"][f"delta_{kind}"]
        d["diff_lb"] = res["bootstrap"]["paired_vs_c1"][pol]["diff_lb"]
        paired[pol] = d
    return conf, paired


def _plot_data(task: Task, res: dict, orig: dict, prereg: dict) -> dict:
    pts = np.array(res["hull_points"])
    xs = np.geomspace(pts[:, 0].min(), pts[:, 0].max(), 120)
    f = hull(pts)(xs)
    summ = lambda s: [s["cost_flat"]["mean"], s["acc"]["mean"]]  # noqa: E731
    return {
        "models": {
            m: [1000 * x, y] for m, (x, y) in zip(task.models, pts, strict=True)
        },
        "hull": [[1000 * x, float(y)] for x, y in zip(xs, f, strict=True)],
        "grid": [
            {"id": k, **dict(zip(("cost", "acc"), summ(v["summary"]), strict=True))}
            for k, v in orig["grid"].items()
        ],
        "confirmatory": {pol: summ(s) for pol, s in orig["confirmatory"].items()},
        "o1": summ(orig["oracle"]["O1"]),
        "o2": orig["oracle"]["O1"]["acc"]["mean"],
    }


# ── verdict ──────────────────────────────────────────────────────────────


def policy_passes(entry: dict, prereg: dict) -> dict:
    """Apply the LIFT and CUT rules. A rule must hold under both cost models.

    The lower bounds come from the flat-cost bootstrap; the length-weighted
    cost model is judged on point estimates (design section 2).
    """
    p = prereg["pass"]

    def lift(kind):
        return entry[f"delta_{kind}"] >= p["lift_delta_pts"] and entry["delta_lb"] > 0

    def cut(kind):
        return (
            entry[f"d2_{kind}"] >= p["cut_d2_mean_pts"]
            and entry["d2_lb"] >= p["cut_d2_lb_pts"]
            and entry[f"delta_{kind}"] >= p["cut_delta_pts"]
        )

    lift_ok = all(lift(k) for k in COST_KINDS)
    cut_ok = all(cut(k) for k in COST_KINDS)
    return {"lift": lift_ok, "cut": cut_ok, "pass": lift_ok or cut_ok}


def beats_c1(task_res: dict, pol: str, prereg: dict) -> bool:
    """Rule 2's second branch: pol beats C1 by >= 2 points, paired LB > 0, at <= C1's cost."""
    c, c1, pr = (
        task_res["confirmatory"][pol],
        task_res["confirmatory"]["C1"],
        task_res["paired"][pol],
    )
    margin = prereg["verdict"]["beat_c1_pts"]
    return (
        all(pr[f"diff_{k}"] >= margin for k in COST_KINDS)
        and pr["diff_lb"] > 0
        and all(c[f"cost_{k}"] <= c1[f"cost_{k}"] for k in COST_KINDS)
    )


def verdict(results: dict, prereg: dict = PREREG) -> dict:
    """The pre-registered decision (design section 6), on the original gold."""
    tasks = results["tasks"]
    if "intent" not in tasks:
        raise ValueError("the verdict is decided on intent; run it")
    passes = {
        t: {p: policy_passes(r["confirmatory"][p], prereg) for p in POLICIES}
        for t, r in tasks.items()
    }
    ok = {t: {p: v["pass"] for p, v in d.items()} for t, d in passes.items()}
    intent = ok["intent"]
    reasons = []
    if not any(intent.values()):
        return {
            "verdict": "KILL",
            "scope": None,
            "passes": ok,
            "detail": passes,
            "reasons": ["no confirmatory policy passes on intent"],
        }
    ruler = []
    for pol in ("C2", "C3"):
        if not intent[pol]:
            continue
        if not intent["C1"]:
            ruler.append(pol)
            reasons.append(f"{pol} passes on intent and C1 fails")
        elif beats_c1(tasks["intent"], pol, prereg):
            ruler.append(pol)
            reasons.append(f"{pol} passes on intent and beats C1 by the margin")
    if ruler:
        name, family = "BUILD_RULER_FED_PLANNER", ("C2", "C3")
    else:
        name, family = "BUILD_DISAGREEMENT_CASCADE_ONLY", ("C1",)
        reasons.append(
            "C1 passes on intent and no Ruler-fed policy both passes and beats it"
        )
    other = [
        t
        for t in ("faithfulness", "attributes")
        if t in ok and any(ok[t][p] for p in family)
    ]
    scope = "general_floor" if other else "classification_only_single_task"
    return {
        "verdict": name,
        "scope": scope,
        "scope_tasks": ["intent", *other],
        "passes": ok,
        "detail": passes,
        "reasons": reasons,
    }


# ── integrity ────────────────────────────────────────────────────────────


@dataclass
class Integrity:
    ok: bool = True
    lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def say(self, text: str):
        self.lines.append(text)

    def fail(self, text: str):
        self.ok = False
        self.problems.append(text)
        self.lines.append(f"FAIL: {text}")


def _derived_score(task: Task, model: str) -> dict:
    """The score.score fields, recomputed from the task's arrays."""
    ones = np.ones(task.n)
    right = task.correct[model]
    if task.metric_kind == "micro":
        all_ok = int((right | ~task.mask).all(1).sum())
        return {
            "rows": task.n,
            "lenient_acc": float(metric(task, right, task.gold_class, ones)),
            "strict_acc": float(
                metric(task, task.strict[model], task.gold_class, ones)
            ),
            "rows_all_ok": all_ok,
            "successes": all_ok,
        }
    out = {
        "rows": task.n,
        "accuracy": float(right[:, 0].mean()),
        "successes": int(right[:, 0].sum()),
    }
    if task.metric_kind == "balanced":
        out["balanced_acc"] = float(metric(task, right, task.gold_class, ones))
    return out


def check_integrity(data_dir: Path = DATA, prereg: dict = PREREG) -> Integrity:
    """Load everything and prove the loading reproduces the baseline report."""
    rep = Integrity()
    rep.say(f"PREREG sha256 {prereg_hash(prereg)}")
    rep.say(f"design sha256 {design_hash() or 'MISSING'}")
    if design_hash() is None:
        rep.fail(f"design file {DESIGN} not found")
    for name in guard_selftest(data_dir):
        rep.fail(f"guard let {name} through")
    rep.say("loader guard: held-out and unlisted files refused: OK")
    try:
        baseline = read_json(data_dir / "baseline-report.json", data_dir)
    except FileNotFoundError:
        rep.fail("baseline-report.json missing")
        return rep
    worst_score = worst_matrix = 0.0
    pairs = 0
    for task_name in TASKS:
        try:
            tasks = {g: load_task(task_name, g, data_dir) for g in GOLDS}
        except FileNotFoundError as e:
            rep.fail(f"{task_name}: missing file {e.filename}")
            continue
        dev = read_jsonl(data_dir / task_name / "dev.jsonl", data_dir)
        adj = read_jsonl(
            data_dir / task_name / "dev-adjudicated.jsonl", data_dir, missing_ok=True
        )
        dev_ids = [r["id"] for r in dev]
        if len(set(dev_ids)) != len(dev_ids):
            rep.fail(f"{task_name}: duplicate dev ids")
        if not {r["id"] for r in adj} <= set(dev_ids):
            rep.fail(f"{task_name}: adjudicated ids not a subset of dev ids")
        rep.say(
            f"[{task_name}] dev {len(dev)} rows, adjudicated "
            f"{len(adj) if adj else 'none (falls back to original)'}"
        )
        for m in MODELS:
            labels = read_jsonl(
                data_dir / "labels" / task_name / f"{m}.jsonl", data_dir
            )
            ids = [r["id"] for r in labels]
            nulls = sum(r["answer"] is None for r in labels)
            cost = read_json(
                data_dir / "labels" / task_name / f"{m}.cost.json", data_dir
            )
            if len(set(ids)) != len(ids):
                rep.fail(f"{task_name}/{m}: duplicate ids in labels")
            if set(ids) != set(dev_ids):
                rep.fail(
                    f"{task_name}/{m}: labels cover {len(set(ids) & set(dev_ids))}"
                    f" of {len(dev_ids)} dev ids ({len(set(ids) - set(dev_ids))} extra)"
                )
            if cost["rows"] != len(dev_ids):
                rep.fail(f"{task_name}/{m}: cost file says {cost['rows']} rows")
            if abs(baseline[task_name][m]["cost_usd"] - cost["cost"]) > 1e-12:
                rep.fail(f"{task_name}/{m}: cost file differs from the baseline")
            rep.say(
                f"  {m:7s} {len(ids)} rows, {nulls} null answers, "
                f"{int((~tasks['original'].valid[m]).sum())} invalid values, "
                f"run cost ${cost['cost']:.4f}"
                + ("" if "source" in cost else "  (cost file has no 'source' key)")
            )
            answers = {r["id"]: r["answer"] for r in labels}
            for gold in GOLDS:
                rows = dev if gold == "original" else (adj or dev)
                got = score.score(task_name, rows, answers)
                want = baseline[task_name][m][gold]
                mine = _derived_score(tasks[gold], m)
                if set(got) != set(want):
                    rep.fail(f"{task_name}/{m}/{gold}: baseline keys differ")
                    continue
                pairs += 1
                for key in want:
                    worst_score = max(worst_score, abs(got[key] - want[key]))
                    if abs(got[key] - want[key]) > 1e-9:
                        rep.fail(
                            f"{task_name}/{m}/{gold}: score.score {key} "
                            f"{got[key]} != baseline {want[key]}"
                        )
                    worst_matrix = max(worst_matrix, abs(mine[key] - got[key]))
                    if abs(mine[key] - got[key]) > 1e-9:
                        rep.fail(
                            f"{task_name}/{m}/{gold}: task arrays give {key} "
                            f"{mine[key]} != score.score {got[key]}"
                        )
        _check_task_values(task_name, tasks["original"], data_dir, rep)
        best = max(
            MODELS,
            key=lambda m, t=task_name: baseline[t][m]["original"][
                {"intent": "accuracy", "faithfulness": "balanced_acc"}.get(
                    t, "lenient_acc"
                )
            ],
        )
        if best != prereg["strong"][task_name]:
            rep.fail(f"{task_name}: best single model is {best}, PREREG says s")
        rep.say(f"  best single model on original gold: {best} (PREREG s: OK)")
    rep.say(
        f"score.score vs baseline-report.json: {pairs} model x task x gold pairs, "
        f"max |diff| {worst_score:.1e}"
    )
    rep.say(f"task arrays vs score.score: max |diff| {worst_matrix:.1e}")
    return rep


def _check_task_values(name: str, task: Task, data_dir: Path, rep: Integrity):
    """Gold labels belong to the allowed value set; names are unique under norm."""
    if name == "attributes":
        return
    rows = read_jsonl(data_dir / name / "dev.jsonl", data_dir)
    if name == "intent":
        names = [
            norm(x) for x in read_json(data_dir / "intent" / "labels.json", data_dir)
        ]
    else:
        names = [norm(x) for x in FAITHFULNESS_VALUES]
    if len(set(names)) != len(names):
        rep.fail(f"{name}: two label names collide under norm()")
    if not {norm(r["label"]) for r in rows} <= set(names):
        rep.fail(f"{name}: a gold label is outside the allowed names")


# ── outputs ──────────────────────────────────────────────────────────────


def _pts(x, digits=1) -> str:
    return "n/a" if x is None else f"{x:+.{digits}f}"


def _f(x, digits=3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _range(s: dict, digits=3) -> str:
    return f"{_f(s['mean'], digits)} [{_f(s['p05'], digits)}, {_f(s['p95'], digits)}]"


def write_markdown(results: dict, path: Path) -> None:
    m, v = results["meta"], results["verdict"]
    L = []  # noqa: N806
    L.append("# Planner replay: results\n")
    L.append(
        f"Design: `{m['design']}` (sha256 `{m['design_sha256']}`). "
        f"PREREG sha256 `{m['prereg_sha256']}`.  \n"
        f"Mode: {'QUICK (smoke run, not citeable)' if m['quick'] else 'full'}; "
        f"S = {m['splits']} splits, B = {m['bootstrap']} bootstrap draws, "
        f"seed {m['seed']}; run time {m['seconds']:.1f}s.\n"
    )
    L.append("## Verdict\n")
    L.append(f"**{v['verdict']}**" + (f", scope: {v['scope']}" if v["scope"] else ""))
    L.append("")
    for r in v["reasons"]:
        L.append(f"- {r}")
    L.append("")
    L.append("| Task | C1 | C2 | C3 |")
    L.append("|---|---|---|---|")
    for t, d in v["passes"].items():
        L.append(
            f"| {t} | "
            + " | ".join("pass" if d[p] else "fail" for p in POLICIES)
            + " |"
        )
    L.append("")
    L.append("## Integrity\n")
    L.append("```")
    L.extend(results["integrity"])
    L.append("```\n")
    L.append(
        "Not verified by the design: the Opus cost files carry no `source` key, "
        "and Opus intent was run twice, so which run its cost records is unknown.\n"
    )
    for t, r in results["tasks"].items():
        L.extend(_task_section(t, r))
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _task_section(t: str, r: dict) -> list[str]:
    L = []  # noqa: N806
    s = r["strong"]
    L.append(f"## {t}\n")
    L.append(
        f"{r['n_rows']} rows. Metric: {r['metric_kind']}. s = {s}. "
        "Delta A and D2 in points; a policy's accuracy is the mean over splits "
        "with the 5-95% range.\n"
    )
    L.append("### Single models and the hull\n")
    L.append("| Model | $ per 1k rows | metric (all rows) |")
    L.append("|---|---|---|")
    for m, d in r["models"].items():
        L.append(f"| {m} | {d['cost_per_1k']:.3f} | {_f(d['metric'])} |")
    L.append("")
    orig = r["golds"]["original"]
    L.append("### Confirmatory policies (original gold)\n")
    L.append(
        "| Policy | $/1k | % of s | accuracy | Delta A (LB) | D2 (LB) | routed | "
        "routed, cheap wrong | $ per 1k successful rows | adjudicated acc | Delta A, length | D2, length |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for pol in POLICIES:
        s_ = orig["confirmatory"][pol]
        c = r["confirmatory"][pol]
        L.append(
            f"| {pol} | {_f(s_['cost_flat']['mean'], 3)} "
            f"| {_f(100 * s_['pct_of_s_flat']['mean'], 0)}% "
            f"| {_range(s_['acc'])} "
            f"| {_pts(c['delta_flat'])} ({_pts(c['delta_lb'])}) "
            f"| {_pts(c['d2_flat'])} ({_pts(c['d2_lb'])}) "
            f"| {_f(s_['routed_share']['mean'], 3)} "
            f"| {_f(s_['routed_cheap_wrong']['mean'], 3)} "
            f"| {_f(1000 * s_['cost_per_success_flat']['mean'], 2) if s_['cost_per_success_flat']['mean'] is not None else 'n/a'} "
            f"| {_f(c.get('acc_adjudicated'))} "
            f"| {_pts(c['delta_length'])} | {_pts(c['d2_length'])} |"
        )
    L.append("")
    flips = [p for p in POLICIES if r["confirmatory"][p].get("sign_flip")]
    if flips:
        L.append(
            "FLAG: Delta A changes sign on the adjudicated gold for "
            + ", ".join(flips)
            + " (not a kill).\n"
        )
    L.append("Paired against C1 (Delta A of the policy minus Delta A of C1):\n")
    L.append("| Policy | flat | length | LB (flat) |")
    L.append("|---|---|---|---|")
    for pol, d in r["paired"].items():
        L.append(
            f"| {pol} | {_pts(d['diff_flat'])} | {_pts(d['diff_length'])} | {_pts(d['diff_lb'])} |"
        )
    L.append("")
    if "acc_strict" in orig["confirmatory"]["C1"]:
        L.append(
            "Attributes, secondary metrics (strict attribute accuracy; rows with every attribute right):\n"
        )
        for pol in POLICIES:
            sc = orig["confirmatory"][pol]
            L.append(
                f"- {pol}: strict {_range(sc['acc_strict'])}, "
                f"all-right rows {_range(sc['success_rate'])}"
            )
        L.append("")
    L.append("### Diagnostics\n")
    d = r["diagnostics"]
    L.append(
        f"- Cheap pair agrees and is wrong on {d['agree_and_wrong']} of "
        f"{d['units']} units ({_f(d['agree_wrong_rate'])}), "
        f"{_f(d['share_of_cheap_wrong'])} of the cheap model's {d['cheap_wrong']} wrong units. "
        f"C1 accuracy ceiling if s were perfect: {_f(d['c1_ceiling'])}."
    )
    o1, o3 = orig["oracle"]["O1"], orig["oracle"]["O3"]
    c2 = orig["confirmatory"]["C2"]
    L.append(
        f"- O1 (cheapest right model per unit, unreachable): accuracy {_f(o1['acc']['mean'])}, "
        f"$/1k {_f(o1['cost_flat']['mean'], 3)}. O2 (any model right): {_f(o1['acc']['mean'])}."
    )
    L.append(
        f"- O3 (C2 fitted on all rows, unreachable): accuracy {_f(o3['acc']['mean'])}; "
        f"O3 minus C2 = {_f(100 * (o3['acc']['mean'] - c2['acc']['mean']), 2)} points "
        "(learning optimism)."
    )
    L.append("")
    L.append("### Exploratory grid (not used for the verdict)\n")
    L.append("| Config | $/1k | accuracy | Delta A | D2 | routed |")
    L.append("|---|---|---|---|---|---|")
    for cid, g in orig["grid"].items():
        sm = g["summary"]
        L.append(
            f"| {cid} | {_f(sm['cost_flat']['mean'], 3)} | {_f(sm['acc']['mean'])} "
            f"| {_pts(sm['delta_flat']['mean'])} | {_pts(sm['d2_flat']['mean'])} "
            f"| {_f(sm['routed_share']['mean'], 3)} |"
        )
    L.append("")
    return L


_SVG_STYLE = (
    "text{fill:#1f2328;font-family:system-ui,sans-serif;font-size:12px}"
    ".bg{fill:#fff}.grid{stroke:#d8dee4}.hull{stroke:#0969da;fill:none;stroke-width:2}"
    ".blue{fill:#0969da}.grey{fill:#8c959f;fill-opacity:.55}.note{fill:#57606a;font-size:11px}"
    ".red{fill:#cf222e;font-weight:600}.redmark{fill:#cf222e}"
    ".green{fill:#1a7f37}.greenline{stroke:#1a7f37;fill:none}"
    "@media (prefers-color-scheme: dark){"
    "text{fill:#e6edf3}.bg{fill:#0d1117}.grid{stroke:#30363d}.hull{stroke:#58a6ff}"
    ".blue{fill:#58a6ff}.grey{fill:#8b949e}.note{fill:#8b949e}"
    ".red{fill:#ff7b72}.redmark{fill:#ff7b72}.green{fill:#3fb950}.greenline{stroke:#3fb950}}"
)


def write_svg(task_result: dict, path: Path) -> None:
    """The cost-accuracy frontier of one task, as a hand-written SVG."""
    p = task_result["plot"]
    w, h, ml, mr, mt, mb = 760, 500, 70, 30, 50, 80
    pw, ph = w - ml - mr, h - mt - mb
    pts = (
        list(p["models"].values())
        + [[g["cost"], g["acc"]] for g in p["grid"]]
        + list(p["confirmatory"].values())
        + [p["o1"]]
        + p["hull"]
    )
    xs = [q[0] for q in pts if q[0] and q[0] > 0]
    ys = [q[1] for q in pts if q[1] is not None]
    lx0, lx1 = np.log10(min(xs) * 0.8), np.log10(max(xs) * 1.25)
    y0, y1 = min(ys) - 0.03, min(max(ys) + 0.03, 1.0)

    def px(x):
        return ml + (np.log10(x) - lx0) / (lx1 - lx0) * pw

    def py(y):
        return mt + (1 - (y - y0) / (y1 - y0)) * ph

    name, strong = escape(task_result["name"]), escape(task_result["strong"])
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
        f'height="{h}" role="img">',
        f"<title>Cost and accuracy frontier: {name}</title>",
        f"<style>{_SVG_STYLE}</style>",
        f'<rect class="bg" width="{w}" height="{h}"/>',
        f'<text x="{ml}" y="24" font-size="15" font-weight="600">{name}: cost against '
        f"{escape(task_result['metric_kind'])} (s = {strong})</text>",
    ]
    for tick in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100):
        if 10**lx0 <= tick <= 10**lx1:
            x = px(tick)
            o.append(
                f'<line class="grid" x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt + ph}"/>'
            )
            o.append(
                f'<text x="{x:.1f}" y="{mt + ph + 18}" text-anchor="middle">{tick:g}</text>'
            )
    step = 0.05 if y1 - y0 < 0.5 else 0.1
    t = np.ceil(y0 / step) * step
    while t <= y1:
        y = py(t)
        o.append(
            f'<line class="grid" x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}"/>'
        )
        o.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" text-anchor="end">{t:.2f}</text>')
        t += step
    o.append(
        f'<text x="{ml + pw / 2}" y="{mt + ph + 42}" text-anchor="middle">$ per 1k rows (log scale)</text>'
    )
    o.append(
        f'<text x="16" y="{mt + ph / 2}" transform="rotate(-90 16 {mt + ph / 2})" '
        'text-anchor="middle">metric</text>'
    )
    hull_pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in p["hull"])
    o.append(f'<polyline class="hull" points="{hull_pts}"/>')
    ceiling = py(p["o2"])
    o.append(
        f'<line class="greenline" x1="{ml}" y1="{ceiling:.1f}" x2="{ml + pw}" '
        f'y2="{ceiling:.1f}" stroke-dasharray="5 4"/>'
    )
    o.append(
        f'<text class="green" x="{ml + pw - 4}" y="{ceiling - 5:.1f}" text-anchor="end">'
        "O2: any model right (unreachable)</text>"
    )
    for g in p["grid"]:
        o.append(
            f'<circle class="grey" cx="{px(g["cost"]):.1f}" cy="{py(g["acc"]):.1f}" r="3">'
            f"<title>{escape(g['id'])}</title></circle>"
        )
    for model, (x, y) in p["models"].items():
        o.append(f'<circle class="blue" cx="{px(x):.1f}" cy="{py(y):.1f}" r="5"/>')
        o.append(
            f'<text x="{px(x) + 7:.1f}" y="{py(y) + 4:.1f}">{escape(model)}</text>'
        )
    for policy, (x, y) in p["confirmatory"].items():
        cx, cy = px(x), py(y)
        o.append(
            f'<path class="redmark" d="M{cx:.1f},{cy - 7:.1f} L{cx + 7:.1f},{cy:.1f} '
            f'L{cx:.1f},{cy + 7:.1f} L{cx - 7:.1f},{cy:.1f} Z"/>'
        )
        o.append(f'<text class="red" x="{cx + 9:.1f}" y="{cy - 6:.1f}">{policy}</text>')
    x, y = p["o1"]
    o.append(
        f'<circle class="greenline" cx="{px(x):.1f}" cy="{py(y):.1f}" r="5" stroke-width="2"/>'
    )
    o.append(f'<text class="green" x="{px(x) + 8:.1f}" y="{py(y) + 4:.1f}">O1</text>')
    o.append(
        f'<text class="note" x="{ml}" y="{mt + ph + 66}">blue: single models and their hull '
        "(all rows); red: confirmatory C1-C3; grey: exploratory grid; "
        "policies are means over splits.</text>"
    )
    o.append("</svg>")
    path.write_text("\n".join(o) + "\n", encoding="utf-8")


def run_study(
    tasks: dict[str, dict[str, Task]],
    prereg: dict = PREREG,
    quick: bool = False,
    integrity: list[str] | None = None,
    log=print,
) -> dict:
    """Run the whole study on loaded tasks and return the results dict."""
    t0 = time.perf_counter()
    n_draws = prereg["bootstrap_quick"] if quick else prereg["bootstrap"]
    out = {"tasks": {}}
    for name in prereg["tasks"]:
        if name in tasks:
            out["tasks"][name] = run_task(tasks[name], prereg, n_draws, log)
    out["verdict"] = verdict(out, prereg)
    out["integrity"] = integrity or []
    out["meta"] = {
        "design": prereg["design"],
        "design_sha256": design_hash(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git": _git_state(),
        "prereg_sha256": prereg_hash(prereg),
        "prereg": prereg,
        "quick": quick,
        "splits": prereg["splits"],
        "bootstrap": n_draws,
        "seed": prereg["seed"],
        "numpy": np.__version__,
        "seconds": time.perf_counter() - t0,
    }
    return out


def _git_state() -> dict:
    """The commit the run's code came from, and whether the tree had changes."""
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
        ).stdout.strip()

    dirty = git(
        "status", "--porcelain", "--", "benchmarks/planner", "benchmarks/golden"
    )
    return {"head": git("rev-parse", "HEAD"), "dirty": bool(dirty)}


def write_results(results: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "replay.md", out_dir / "replay.json"]
    write_markdown(results, paths[0])
    paths[1].write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for name, r in results["tasks"].items():
        svg = out_dir / f"frontier-{name}.svg"
        write_svg(r, svg)
        paths.append(svg)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="integrity check only")
    parser.add_argument("--quick", action="store_true", help="B = 100 smoke run")
    parser.add_argument("--out", type=Path, help="results directory")
    args = parser.parse_args(argv)

    rep = check_integrity()
    print("\n".join(rep.lines))
    if not rep.ok:
        print(f"\nINTEGRITY FAILED ({len(rep.problems)} problems)")
        return 2
    print("\nINTEGRITY OK")
    if args.check:
        return 0

    tasks = {t: {g: load_task(t, g) for g in GOLDS} for t in TASKS}
    results = run_study(tasks, PREREG, args.quick, rep.lines)
    out_dir = args.out or HERE / "results" / ("quick" if args.quick else "")
    for p in write_results(results, out_dir):
        print(f"wrote {p.relative_to(REPO) if p.is_relative_to(REPO) else p}")
    v = results["verdict"]
    print(f"VERDICT: {v['verdict']} ({v['scope']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
