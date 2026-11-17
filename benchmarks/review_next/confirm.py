"""Review-next confirmation: does R2 hold on fresh rows?

Pre-registered replication (design: ``benchmarks/review_next/DESIGN-confirm.md``;
parent ``DESIGN.md``, locked, verdict BUILD_SCOPED(faithfulness, R2)). The rows
are new: 770 intent rows and 320 faithfulness rows that dev never used, each
answered four times by Haiku on one route. One ranker per task is tested, R2
(Haiku's dissent D over three extra samples, then R1), against random (R0) and
the free per-value rate (R1). The splits, estimators, ties, thresholds and the
cluster bootstrap are the parent's (``review_next.py``, ``replay.py``,
``addendum1.py``), imported and not copied. Every constant lives in
``PREREG_CONFIRM``; the results embed its hash. Do not change a constant after
seeing a result.

    uv run python benchmarks/review_next/confirm.py --check   # no ranker but R0
    uv run python benchmarks/review_next/confirm.py --lock    # after the 4 + 4 sample runs
    uv run python benchmarks/review_next/confirm.py           # the one full run

``--check`` exits 2 on a failed check, 3 when the data is not all there yet
(it still checks everything that can be checked) and 0 when all is well. The
full run exits 2 on a dirty tree, on an existing result, or when the lock file
or the data hashes do not match. ``--quick`` (B = 100) also prints the verdict,
so it is refused on the real data until the full result exists.

Every read of a data file goes through ``guard_confirm``. It allows the confirm
rows, the four-plus-one Haiku sample files and their costs, the dev rows (for
the id-disjointness check) and the intent label names, and nothing else. The
release split is never read: the allowlist refuses it before any byte is read.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PLANNER = HERE.parent / "planner"
GOLDEN = HERE.parent / "golden"
for _dir in (GOLDEN, PLANNER, HERE):
    sys.path.insert(0, str(_dir))
import addendum1  # noqa: E402
import label as golden_label  # noqa: E402
import replay  # noqa: E402
import review_next as rn  # noqa: E402

REPO = replay.REPO
DATA = replay.DATA
DESIGN = HERE / "DESIGN-confirm.md"
LOCK = HERE / "confirm-data.lock.json"
RESULTS = HERE / "results"
TASKS = ("intent", "faithfulness")

#: Every constant of the design, in one place. Its sha256 goes into the results.
#: ``confirm_split_sha256`` is filled after ``build.py --confirm``: a full sha256
#: of the file, or its first 16 hex digits as the manifest records them.
PREREG_CONFIRM: dict[str, Any] = {
    "design": "benchmarks/review_next/DESIGN-confirm.md",
    "parent": {
        "design_sha256": "57e8962a0173c860dc464e2fb4cc2a456c3ece2f711c3209412f17265cfb7bff",
        "prereg_sha256": "85737be9a20a34cc253f665ab37fa5bd10dae2ce674d3b6e382875d9426afecc",
        "script_sha256": "94836cab636f6d40a3880ec90c127d502da4f6ec7b55cf3841fbed0879ad7e35",
        "verdict": "BUILD_SCOPED(faithfulness, R2)",
    },
    "design_sha256": "b8de9ba652b1bddfde4e586693ef90945cdd9ce2b5de662b3f41a95bd96c539a",
    "tasks": ["intent", "faithfulness"],
    "build": {
        "seed": 20261020,
        "rng": {
            "intent": "20261020:intent:confirm",
            "faithfulness": "20261020:faithfulness:confirm",
        },
        "intent_per_class": 10,
        "faithfulness_per_domain_label": 20,
        "faithfulness_cap_per_doc_label": 5,
        "rows": {"intent": 770, "faithfulness": 320},
    },
    "confirm_split_sha256": {
        "intent": "6ccee5cc9550152cd0c8ab298cc7f4ebb81a09f2a34e49f54d9fa972858899f5",
        "faithfulness": "6c8e98e0163ee7bceb1d9f7a2efca64f2aa1622edcc7f17a1a59b475cf3fcc9d",
    },
    "sample_files": ["haiku-s1", "haiku-s2", "haiku-s3", "haiku-s4"],
    "replacement_file": "haiku-s5",
    "delivered_column": 0,
    "delivered": "haiku",
    "partner": "haiku",  # P is 0 everywhere and no ranker reads it
    "route": "openrouter",
    "model_id": "openrouter/anthropic/claude-haiku-5-5",
    "budget_usd": {
        "total": 1.00,
        "per_run": {"intent": 0.14, "faithfulness": 0.09},
        "smoke": 0.01,
    },
    "seed": 20261118,
    "n_labelled": 100,
    "splits": 1000,
    "split_purpose": 0,
    "bootstrap_purpose": 1,
    "bootstrap": 1000,
    "bootstrap_quick": 100,  # the quick mode only; `review_next.effective_prereg` reads it
    "bootstrap_splits_per_draw": 20,
    "cluster": {"intent": "row", "faithfulness": "doc"},
    "shrinkage_m": 4,
    "k_fractions": ["0.05", "0.10", "0.20"],
    "primary_k": "0.10",
    "null": "R0",
    "comparator": "R1",
    "confirmatory": ["R2"],
    # Key columns, compared lexicographically; "r1" is the float tie-break. The
    # s-indices of the exploratory rankers are COLUMN indices: column j is the
    # file s(j+1), and column 0 (s1) is the delivered answer.
    "rankers": {"R0": [], "R1": ["r1"], "R2": ["D", "r1"]},
    "exploratory": {
        "E2": ["D"],
        "E4": ["D:s1,s2", "r1"],
        "E5": ["D:s1,s3", "r1"],
        "E6": ["D:s2,s3", "r1"],
        "E7": ["D:s1", "r1"],
        "E8": ["r1_all_rows"],
    },
    "alpha": 0.05,
    "n_tests": 2,  # LB quantile = 0.025
    "pass": {
        "lift_mean_min": 2.0,
        "lift_lb_gt": 1.0,
        "gain_mean_min_pts": 10.0,
        "gain_lb_gt_pts": 0.0,
    },
    "kill": {"gain_mean_lt_pts": 5.0, "lift_mean_lt": 1.5},
    "flags": {
        "k_fragile_gain_le_pts": 0.0,
        "subset_gain_min_pts": 10.0,
        "domain_neg_min": 3,
    },
    "v": {
        "v1": {
            "intent": {"metric": "accuracy", "center": 0.810, "tol": 0.060},
            "faithfulness": {"metric": "balanced", "center": 0.820, "tol": 0.070},
        },
        "v2": {
            "low": 0.5,
            "high": 2.0,
            "implied": {"intent": 95.0, "faithfulness": 109.6},
        },
        "v3_tol_pts": 5.0,
        "v4_max_bad_pct": 3.0,
    },
    # The design's own numbers, which the split-size check must reproduce from
    # the row counts: test rows (rows - n_labelled) and k at 5/10/20%.
    "expected": {
        "test_rows": {"intent": 670, "faithfulness": 220},
        "k": {"intent": [34, 67, 134], "faithfulness": [11, 22, 44]},
    },
    # Counts the design discloses (section 0). The build test reproduces them
    # from the sources; they are recorded here so the results embed them.
    "disclosure": {
        "faithfulness_pool": {
            "docs": 66,
            "unused": 1756,
            "consistent": 579,
            "inconsistent": 1177,
        },
        "intent_pool_min_per_class": 32,
    },
}

#: The exploratory rankers whose gain the SUBSET flag reads (R2 on two samples).
SUBSET_RANKERS = rn.SUBSET_RANKERS


def prereg_hash(prereg: dict = PREREG_CONFIRM) -> str:
    return str(replay.prereg_hash(prereg))


def _assert_prereg_consistent() -> None:
    """Fail at import if the constants contradict the design or the parent's."""
    p, parent = PREREG_CONFIRM, rn.PREREG_REVIEW
    for key in (
        "n_labelled",
        "splits",
        "bootstrap",
        "bootstrap_splits_per_draw",
        "shrinkage_m",
        "alpha",
        "k_fractions",
        "primary_k",
        "pass",
        "kill",
        "split_purpose",
        "bootstrap_purpose",
    ):
        if p[key] != parent[key]:
            raise RuntimeError(f"PREREG_CONFIRM[{key!r}] differs from the parent's")
    if p["n_tests"] != len(p["confirmatory"]) * len(p["tasks"]):
        raise RuntimeError("n_tests is not rankers x tasks")
    if not {*p["confirmatory"], p["null"], p["comparator"]} <= set(p["rankers"]):
        raise RuntimeError("a named ranker has no key spec")
    if set(p["rankers"]) & set(p["exploratory"]):
        raise RuntimeError("a ranker name is used twice")
    if not set(SUBSET_RANKERS) <= set(p["exploratory"]):
        raise RuntimeError("a SUBSET ranker is not an exploratory ranker")
    if p["delivered"] != p["partner"]:
        raise RuntimeError("the partner must be the delivered model: P is 0")
    specs = [*p["rankers"].values(), *p["exploratory"].values()]
    if any(col in ("P", "D+P") for spec in specs for col in spec):
        raise RuntimeError("a ranker reads the pair signal P")
    if len(p["sample_files"]) != 4 or p["delivered_column"] != 0:
        raise RuntimeError("the design uses four samples, s1 delivered")
    if p["budget_usd"]["total"] != golden_label.CONFIRM_TOTAL_BUDGET:
        raise RuntimeError("the total budget differs from label.py's")


_assert_prereg_consistent()


def confirm_effective(prereg: dict = PREREG_CONFIRM) -> dict:
    """The parent's effective PREREG with this design's seed and family of 2.

    ``replay.split_rng`` reads the seed and the task order from it; the parent's
    task order is kept (intent 0, faithfulness 1), so only the seed moves.
    """
    return {**rn.effective_prereg(prereg), "seed": prereg["seed"]}


# ── the guard: the only way a data file is read ──────────────────────────


def guard_confirm(path: Path | str, data_dir: Path = DATA, extra: tuple = ()) -> Path:
    """Return `path` if this study may read it, else raise PermissionError.

    Allowed, relative to the golden data directory: ``<task>/confirm.jsonl``;
    ``<task>/dev.jsonl`` (ids only, to prove the rows are new);
    ``labels/<task>/confirm/haiku-s1..s5`` as ``.jsonl`` and ``.cost.json``; and
    ``intent/labels.json``. Also any path in `extra` (the lock file). Everything
    else is refused, the release split included, before a byte is read.
    """
    resolved = Path(path).resolve()
    if resolved in {Path(x).resolve() for x in extra}:
        return resolved
    try:
        parts = resolved.relative_to(Path(data_dir).resolve()).parts
    except ValueError:
        raise PermissionError(f"{path} is outside the golden data directory") from None
    stems = {*PREREG_CONFIRM["sample_files"], PREREG_CONFIRM["replacement_file"]}
    ok = (
        (
            len(parts) == 2
            and parts[0] in TASKS
            and parts[1] in ("confirm.jsonl", "dev.jsonl")
        )
        or (
            len(parts) == 4
            and parts[0] == "labels"
            and parts[1] in TASKS
            and parts[2] == "confirm"
            and any(
                parts[3] == f"{s}{ext}"
                for s in stems
                for ext in (".jsonl", ".cost.json")
            )
        )
        or parts == ("intent", "labels.json")
    )
    if not ok:
        raise PermissionError(f"{path}: not one of the files this study may read")
    return resolved


def _read_guarded(path: Path | str, data_dir: Path = DATA, extra: tuple = ()) -> str:
    """The only raw read in this module: the text of a file the guard allows."""
    return guard_confirm(path, data_dir, extra).read_bytes().decode("utf-8")


def guard_selftest(data_dir: Path = DATA) -> list[str]:
    """Names of files the guard wrongly lets through (it should return none)."""
    leaked = []
    release = "hold" + "out"  # never spelled out: this module must not name the split
    for bad in (
        data_dir / "intent" / f"{release}.jsonl",
        data_dir / "faithfulness" / f"{release}.jsonl",
        data_dir / "labels" / "intent" / "confirm" / f"{release}.jsonl",
        data_dir / "labels" / "intent" / "haiku.jsonl",  # a dev label file
        data_dir / "labels" / "intent" / "confirm" / "haiku-s6.jsonl",
        data_dir / "labels" / "intent" / "confirm" / "haiku-s1-smoke3.jsonl",
        data_dir / "labels" / "intent" / "haiku-s1.jsonl",
        data_dir / "intent" / "dev-adjudicated.jsonl",
        data_dir / "attributes" / "dev.jsonl",
        data_dir / "baseline-report.json",
        data_dir.parent / "manifest.json",
    ):
        try:
            guard_confirm(bad, data_dir)
            leaked.append(str(bad.relative_to(data_dir.parent)))
        except PermissionError:
            pass
    return leaked


# ── rows, answers and tasks ──────────────────────────────────────────────


def _jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_rows(task: str, split: str, data_dir: Path = DATA) -> list[dict]:
    """The rows of ``<task>/<split>.jsonl`` (split is ``confirm`` or ``dev``), guarded."""
    return _jsonl(_read_guarded(data_dir / task / f"{split}.jsonl", data_dir))


def label_dir(task: str, data_dir: Path = DATA) -> Path:
    return data_dir / "labels" / task / "confirm"


def sample_path(task: str, stem: str, data_dir: Path = DATA) -> Path:
    return label_dir(task, data_dir) / f"{stem}.jsonl"


def allowed_values(task: str, data_dir: Path = DATA) -> set[str]:
    """The task's valid final answers (the parent's validity signal), guarded."""
    if task == "intent":
        names = json.loads(_read_guarded(data_dir / "intent" / "labels.json", data_dir))
        return {replay.norm(x) for x in names}
    return {replay.norm(x) for x in replay.FAITHFULNESS_VALUES}


def final_by_id(rows: list[dict]) -> tuple[dict[str, str], int]:
    """Normalised final answer per id and the null count, as ``addendum1`` makes them.

    A null answer gets a code of its own so two nulls never agree. This is
    ``addendum1._final_by_id`` on rows already read through the guard: that
    function reads its own file through the parent's guard, which refuses the
    confirm paths.
    """
    finals = {
        r["id"]: f"\x00null:{r['id']}"
        if r["answer"] is None
        else replay.norm(replay.final_answer(r["answer"]))
        for r in rows
    }
    return finals, sum(r["answer"] is None for r in rows)


def normalised_finals(rows: list[dict], answers: dict[str, Any]) -> list[str]:
    """``norm(final_answer(.))`` of each row's answer, the way ``replay.load_task`` does.

    A missing or null answer becomes the string "none", as in the parent.
    """
    return [replay.norm(replay.final_answer(answers.get(r["id"]))) for r in rows]


def confirm_task(
    task: str,
    rows: list[dict],
    finals: list[str],
    cost: tuple[float, int],
    allowed: set[str] | None = None,
    model: str = "haiku",
) -> replay.Task:
    """One model's answers on `rows` as a `replay.Task`, built with `replay.make_task`.

    `finals` are the normalised final answers in row order (`normalised_finals`),
    `cost` is the run's (USD, rows). Same arrays as ``replay.load_task`` gives
    for one model; the parent's loader cannot be used because it wants all five
    models' files for dev rows.
    """
    allowed = (
        allowed
        if allowed is not None
        else {replay.norm(x) for x in replay.FAITHFULNESS_VALUES}
    )
    gold_norm = [replay.norm(r["label"]) for r in rows]
    classes = sorted(set(gold_norm))
    vocab: dict[str, int] = {}
    value = np.array([vocab.setdefault(f, len(vocab)) for f in finals])[:, None]
    correct = np.array([f == g for f, g in zip(finals, gold_norm, strict=True)])[
        :, None
    ]
    lens = np.array([replay._input_chars(task, r) for r in rows], float)
    return replay.make_task(
        name=task,
        gold_kind="original",
        metric_kind=replay.PREREG["metric"][task],
        ids=[r["id"] for r in rows],
        mask=np.ones((len(rows), 1), bool),
        gold_class=np.array([classes.index(g) for g in gold_norm])
        if task == "faithfulness"
        else np.zeros(len(rows), int),
        value={model: value},
        cell={model: value},
        correct={model: correct},
        valid={model: np.array([f in allowed for f in finals])},
        cost={model: cost},
        lens=lens,
        len_total=float(lens.sum()),
        n_cells=len(vocab),
    )


def _null_per_file(finals: dict[str, str], stem: str) -> dict[str, str]:
    """Give each null answer a code of its own per file and row.

    ``final_by_id`` marks a null by row only, so two files that are both null on
    one row would read as agreeing in D, V3 and the identical-columns check. A
    null is no answer: it must agree with nothing, another null included. (The
    per-file copy check in ``_file_report`` keeps the by-row marker, so a copied
    file is still caught.)
    """
    return {
        rid: f"\x00null:{stem}:{rid}" if f.startswith("\x00null:") else f
        for rid, f in finals.items()
    }


def load_confirm_samples(
    task: str,
    stems: list[str] | None = None,
    data_dir: Path = DATA,
    rows: list[dict] | None = None,
) -> addendum1.SampleSet:
    """The four Haiku answers on the confirm rows; column 0 is s1, the delivered one."""
    stems = stems or PREREG_CONFIRM["sample_files"]
    rows = rows if rows is not None else load_rows(task, "confirm", data_dir)
    columns = [
        _null_per_file(
            final_by_id(
                _jsonl(_read_guarded(sample_path(task, s, data_dir), data_dir))
            )[0],
            s,
        )
        for s in stems
    ]
    return addendum1.build_sample_set(
        [r["id"] for r in rows],
        columns,
        allowed_values(task, data_dir),
        [replay.norm(r["label"]) for r in rows],
    )


def load_confirm_task(
    task: str,
    stems: list[str] | None = None,
    data_dir: Path = DATA,
) -> tuple[replay.Task, addendum1.SampleSet, list[dict]]:
    """The delivered answers (s1) as a Task, with the aligned samples and the rows."""
    stems = stems or PREREG_CONFIRM["sample_files"]
    rows = load_rows(task, "confirm", data_dir)
    delivered = _jsonl(_read_guarded(sample_path(task, stems[0], data_dir), data_dir))
    cost = json.loads(
        _read_guarded(
            sample_path(task, stems[0], data_dir).with_suffix(".cost.json"), data_dir
        )
    )
    finals = normalised_finals(rows, {r["id"]: r["answer"] for r in delivered})
    t = confirm_task(
        task,
        rows,
        finals,
        (float(cost["cost"]), int(cost["rows"])),
        allowed_values(task, data_dir),
        PREREG_CONFIRM["delivered"],
    )
    return t, load_confirm_samples(task, stems, data_dir, rows), rows


# ── the document-cluster bootstrap ───────────────────────────────────────


def doc_bootstrap_batches(
    rng: np.random.Generator,
    doc_idx: np.ndarray,
    n_l: int,
    n_draws: int,
    per_draw: int,
    chunk: int,
):
    """Yield (train_w, test_w, n_draws_in_chunk): the parent's bootstrap, documents as clusters.

    Each draw gives multinomial weights over the documents used (D draws over D
    documents); a row takes its document's weight. Each draw then makes
    `per_draw` splits that put `n_l` rows on the labelled side, drawn over rows
    as the Ruler draws. The weights apply on both sides, like
    ``replay.bootstrap_batches``, whose stream this equals when every row is
    its own document.
    """
    n, n_docs = len(doc_idx), int(np.max(doc_idx)) + 1
    per_chunk = max(1, chunk // per_draw)
    done = 0
    while done < n_draws:
        d = min(per_chunk, n_draws - done)
        doc_w = rng.multinomial(n_docs, np.full(n_docs, 1.0 / n_docs), size=d)
        w = doc_w.astype(float)[:, doc_idx]
        train = replay.draw_splits(rng, n, n_l, d * per_draw)
        weights = np.repeat(w, per_draw, axis=0)
        yield weights * train, weights * ~train, d
        done += d


def cluster_index(task: str, rows: list[dict], prereg: dict = PREREG_CONFIRM):
    """Per-row document index for a document-clustered task, None when rows are clusters."""
    if prereg["cluster"][task] == "row":
        return None
    return np.unique([r["doc"] for r in rows], return_inverse=True)[1]


def bootstrap_confirm(
    task: replay.Task,
    samples: addendum1.SampleSet,
    n_draws: int,
    prereg: dict = PREREG_CONFIRM,
    doc_idx: np.ndarray | None = None,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Draw-level lift L and gain G of each confirmatory ranker, per k fraction.

    The parent's stream (purpose 1, this design's seed): per draw, 20 splits of
    the resampled clusters. Every ranker sees the same weights, so G is paired:
    a ranker that equals R1 has G = 0 in every draw.
    """
    eff = confirm_effective(prereg)
    n_l, per_draw = prereg["n_labelled"], prereg["bootstrap_splits_per_draw"]
    fracs, rankers = prereg["k_fractions"], prereg["confirmatory"]
    comparator = prereg["comparator"]
    rng = replay.split_rng(eff, task.name, "original", prereg["bootstrap_purpose"], n_l)
    batches = (
        replay.bootstrap_batches(rng, task.n, n_l, n_draws, per_draw, chunk=1000)
        if doc_idx is None
        else doc_bootstrap_batches(rng, doc_idx, n_l, n_draws, per_draw, chunk=1000)
    )
    out: dict[str, dict[str, dict[str, list]]] = {
        r: {f: {"L": [], "G": []} for f in fracs} for r in rankers
    }
    for train_w, test_w, d in batches:
        red = rn.reduce_splits(
            rn.evaluate(
                task, samples, train_w, test_w, [comparator, *rankers], fracs, prereg
            ),
            d,
            comparator,
        )
        for r in rankers:
            for f in fracs:
                out[r][f]["L"].append(red[r][f]["L"])
                out[r][f]["G"].append(red[r][f]["G"])
    return {
        r: {
            f: {q: np.concatenate(v) for q, v in by_q.items()}
            for f, by_q in by_f.items()
        }
        for r, by_f in out.items()
    }


# ── verdict and flags ────────────────────────────────────────────────────


def verdict_confirm(statuses: dict[str, str]) -> dict:
    """The overall decision (DESIGN-confirm section 5) from {task: PASS|KILL|INCONCLUSIVE}.

    CONFIRMED_GENERAL: both pass. CONFIRMED(faithfulness): faithfulness passes,
    intent does not. NOT_REPLICATED: faithfulness is KILL (an intent PASS is
    then reported as NEW(intent)). Anything else is INCONCLUSIVE.
    """
    faith, intent = statuses["faithfulness"], statuses["intent"]
    new: list[str] = []
    if faith == "PASS" and intent == "PASS":
        name = "CONFIRMED_GENERAL"
    elif faith == "PASS":
        name = "CONFIRMED(faithfulness)"
    elif faith == "KILL":
        name = "NOT_REPLICATED"
        if intent == "PASS":
            new = ["NEW(intent)"]
            name += " + NEW(intent)"
    else:
        name = "INCONCLUSIVE"
    return {"verdict": name, "statuses": dict(statuses), "new": new}


def flags_confirm(
    entry: dict,
    domain_gains: dict[str, float | None] | None,
    prereg: dict = PREREG_CONFIRM,
) -> list[str]:
    """K_FRAGILE, SUBSET and DOMAIN: reported, never decisive.

    `entry` holds `G_by_frac` (mean G of R2 per k fraction) and `G_subsets`
    (mean G of E4-E6 at the primary k). `domain_gains` is G per faithfulness
    domain, None for intent.
    """
    f, out = prereg["flags"], []
    others = [x for x in prereg["k_fractions"] if x != prereg["primary_k"]]
    by_frac = entry.get("G_by_frac", {})
    if any(
        by_frac.get(x) is not None and by_frac[x] <= f["k_fragile_gain_le_pts"]
        for x in others
    ):
        out.append("K_FRAGILE")
    if any(
        g is not None and g < f["subset_gain_min_pts"]
        for g in entry.get("G_subsets", {}).values()
    ):
        out.append("SUBSET")
    if domain_gains is not None and (
        sum(g is not None and g <= 0 for g in domain_gains.values())
        >= f["domain_neg_min"]
    ):
        out.append("DOMAIN")
    return out


# ── checks ───────────────────────────────────────────────────────────────

LoadedTask = tuple[replay.Task, addendum1.SampleSet, list[dict]]


@dataclass
class ConfirmChecks(rn.Checks):
    """What `--check` found. ``problems`` exit 2; ``missing`` (data not there yet) exit 3.

    ``v123`` make one task INCONCLUSIVE and are reported, not an exit 2.
    """

    missing: list[str] = field(default_factory=list)
    selected: dict[str, list[str]] = field(default_factory=dict)
    files: dict[str, dict] = field(default_factory=dict)
    v3: dict[str, dict] = field(default_factory=dict)
    spend: float = 0.0
    holds: list[str] = field(default_factory=list)
    loaded: dict[str, LoadedTask] = field(default_factory=dict, repr=False)

    @property
    def exit_code(self) -> int:
        return 2 if self.problems else 3 if self.missing else 0

    def need(self, what: str) -> None:
        self.missing.append(what)
        self.lines.append(f"NOT YET AVAILABLE: {what}")

    def fail_v(self, task: str, text: str) -> None:
        self.v123.setdefault(task, []).append(text)
        self.lines.append(f"FAIL ({task}): {text}")


def _pin_is_set(want: str) -> bool:
    """True once the PREREG holds a real pin: a full sha256 or its first 16 hex digits."""
    return len(want) in (16, 64) and all(c in "0123456789abcdef" for c in want)


def _pin_matches(have: str, want: str) -> bool:
    return _pin_is_set(want) and have.startswith(want)


def _check_design_hashes(
    rep: ConfirmChecks,
    prereg: dict,
    design: Path,
    parent_design: Path,
    parent_script: Path,
) -> None:
    pins = prereg["parent"]
    got = {
        "this design": (
            addendum1.sha256_file(design) if design.exists() else None,
            prereg["design_sha256"],
        ),
        "the parent's design": (
            addendum1.sha256_file(parent_design) if parent_design.exists() else None,
            pins["design_sha256"],
        ),
        "the parent's PREREG": (
            replay.prereg_hash(rn.PREREG_REVIEW),
            pins["prereg_sha256"],
        ),
        "the parent's script (review_next.py)": (
            addendum1.sha256_file(parent_script) if parent_script.exists() else None,
            pins["script_sha256"],
        ),
    }
    rep.say(f"PREREG_CONFIRM sha256 {prereg_hash(prereg)}")
    bad = False
    for name, (have, want) in got.items():
        if have != want:
            bad = True
            rep.fail(f"{name}: sha256 {str(have)[:12]} is not the pinned {want[:12]}")
    if not bad:
        rep.say("design, parent design, parent PREREG and parent script hashes: OK")


def _check_confirm_rows(
    rep: ConfirmChecks, data_dir: Path, prereg: dict
) -> dict[str, list[dict]]:
    """Counts, quotas, disjointness from dev and the pinned hash, per task that exists."""
    b, found = prereg["build"], {}
    for task in prereg["tasks"]:
        path = data_dir / task / "confirm.jsonl"
        pin = prereg["confirm_split_sha256"][task]
        if not path.exists():
            rep.need(f"{task}/confirm.jsonl (build.py --confirm)")
            if not _pin_is_set(pin):
                rep.need(
                    f"PREREG confirm_split_sha256[{task}] (filled after the build)"
                )
            continue
        text = _read_guarded(path, data_dir)
        rows = _jsonl(text)
        found[task] = rows
        have = addendum1.sha256_file(guard_confirm(path, data_dir))
        ids = [r["id"] for r in rows]
        problems = []
        if not _pin_matches(have, pin):
            problems.append(f"sha256 {have[:16]} is not the PREREG's {pin[:24]!r}")
        if len(rows) != b["rows"][task]:
            problems.append(f"{len(rows)} rows, expected {b['rows'][task]}")
        if len(set(ids)) != len(ids):
            problems.append("duplicate ids")
        if task == "intent":
            per = Counter(r["label"] for r in rows)
            if set(per.values()) != {b["intent_per_class"]}:
                problems.append(f"classes do not all have {b['intent_per_class']} rows")
            detail = f"{len(per)} classes x {b['intent_per_class']}"
        else:
            per_dl = Counter((r["domain"], r["label"]) for r in rows)
            per_doc = Counter((r["doc"], r["label"]) for r in rows)
            quota = b["faithfulness_per_domain_label"]
            if set(per_dl.values()) != {quota} or len(per_dl) != 2 * len(
                {d for d, _ in per_dl}
            ):
                problems.append(f"domain/label pairs do not all have {quota} rows")
            if max(per_doc.values(), default=0) > b["faithfulness_cap_per_doc_label"]:
                problems.append("a document gives more rows of one label than the cap")
            detail = (
                f"{len({d for d, _ in per_dl})} domains x {quota} per label; max "
                f"{max(per_doc.values(), default=0)} rows per document and label"
            )
        dev_path = data_dir / task / "dev.jsonl"
        if dev_path.exists():
            dev_ids = {r["id"] for r in load_rows(task, "dev", data_dir)}
            if dev_ids & set(ids):
                problems.append("a confirm id is a dev id")
        else:
            problems.append("dev.jsonl missing: id disjointness not checked")
        for p in problems:
            rep.fail(f"{task}/confirm.jsonl: {p}")
        if not problems:
            rep.say(
                f"[{task}] confirm rows: {len(rows)} ({detail}); disjoint from dev; "
                f"sha256 {have[:16]} matches the PREREG"
            )
    return found


def _file_report(
    stem: str,
    task: str,
    ctx: dict,
    data_dir: Path,
    prereg: dict,
    earlier: dict[str, dict[str, str]],
) -> tuple[dict, dict[str, str] | None]:
    """Everything known about one sample file: V1, V2, V4 and its hashes."""
    path = sample_path(task, stem, data_dir)
    cost_path = path.with_suffix(".cost.json")
    rep: dict = {"stem": stem, "exists": path.exists() and cost_path.exists()}
    if not rep["exists"]:
        return rep, None
    rows = _jsonl(_read_guarded(path, data_dir))
    cost = json.loads(_read_guarded(cost_path, data_dir))
    finals, nulls = final_by_id(rows)
    ids = [r["id"] for r in rows]
    confirm_ids, n = ctx["ids"], len(ctx["ids"])
    right = np.array([finals.get(i) == ctx["gold_norm"][i] for i in confirm_ids])
    value = float(
        replay.metric(ctx["task"], right[:, None], ctx["task"].gold_class, np.ones(n))
    )
    bad = sum(
        finals.get(i) is None or finals[i] not in ctx["allowed"] for i in confirm_ids
    )
    out_tokens, n_rows = cost.get("output_tokens"), cost.get("rows")
    v = prereg["v"]
    rep.update(
        sha256=addendum1.sha256_file(guard_confirm(path, data_dir)),
        cost_sha256=addendum1.sha256_file(guard_confirm(cost_path, data_dir)),
        cost=float(cost.get("cost", float("nan"))),
        route=cost.get("route"),
        model=cost.get("model"),
        input_tokens=cost.get("input_tokens"),
        output_tokens=out_tokens,
        rows=n_rows,
        nulls=nulls,
        v1=addendum1.check_v1(value, v["v1"][task]),
        v2=addendum1.check_v2(
            None if out_tokens is None or not n_rows else out_tokens / n_rows,
            v["v2"]["implied"][task],
            v["v2"],
        ),
    )
    rep["v4"] = {
        "ids_ok": len(ids) == len(set(ids)) and set(ids) == set(confirm_ids),
        "n_ids": len(ids),
        "n_duplicates": len(ids) - len(set(ids)),
        "n_missing": len(set(confirm_ids) - set(ids)),
        "n_extra": len(set(ids) - set(confirm_ids)),
        "bad": bad,
        "bad_pct": 100.0 * bad / n,
        "bad_ok": bad * 100 <= v["v4_max_bad_pct"] * n,
        "provenance_ok": rep["route"] == prereg["route"]
        and rep["model"] == prereg["model_id"]
        and n_rows == n,
    }
    rep["identical"] = [
        o
        for o, other in earlier.items()
        if all(finals.get(i) == other.get(i) for i in confirm_ids)
    ]
    return rep, finals


def _file_line(r: dict) -> str:
    if r["v2"]["ratio"] is None:
        return f"  {r['stem']}: no output_tokens in its cost file"
    v1, v2, v4 = r["v1"], r["v2"], r["v4"]
    mark = lambda ok: "ok" if ok else "FAIL"  # noqa: E731
    return (
        f"  {r['stem']}: V1 {v1['value']:.3f} in [{v1['lo']:.3f}, {v1['hi']:.3f}] "
        f"{mark(v1['ok'])}; V2 {v2['value']:.1f} tokens/row, x{v2['ratio']:.2f} of the "
        f"design's {mark(v2['ok'])}; V4 ids {mark(v4['ids_ok'])}, bad {v4['bad']} "
        f"({v4['bad_pct']:.1f}%) {mark(v4['bad_ok'])}, provenance "
        f"{mark(v4['provenance_ok'])}; sha256 {r['sha256'][:12]}, cost ${r['cost']:.4f}"
    )


def _check_task_samples(
    rep: ConfirmChecks, task: str, rows: list[dict], data_dir: Path, prereg: dict
) -> None:
    """V1-V4 for one task's sample files and the single s5 replacement."""
    primary, spare = list(prereg["sample_files"]), prereg["replacement_file"]
    stems = [*primary, spare]
    present = {s: sample_path(task, s, data_dir).exists() for s in stems}
    missing = [s for s in primary if not present[s]]
    extra = (
        sorted(
            p.stem
            for p in label_dir(task, data_dir).glob("haiku-s*.jsonl")
            if p.stem not in stems and "smoke" not in p.stem
        )
        if label_dir(task, data_dir).exists()
        else []
    )
    if extra:
        rep.fail(f"{task}: unexpected sample files {extra}")
    if missing:
        rep.need(f"{task}: {', '.join(missing)} (label.py --split confirm --sample N)")
        if present[spare]:
            rep.fail(f"{task}: {spare} exists while {', '.join(missing)} is missing")
        return
    allowed = allowed_values(task, data_dir)
    gold_norm = {r["id"]: replay.norm(r["label"]) for r in rows}
    first = _jsonl(_read_guarded(sample_path(task, primary[0], data_dir), data_dir))
    ref_task = confirm_task(
        task,
        rows,
        normalised_finals(rows, {r["id"]: r["answer"] for r in first}),
        (0.0, 1),
        allowed,
    )
    ctx = {
        "task": ref_task,
        "ids": [r["id"] for r in rows],
        "gold_norm": gold_norm,
        "allowed": allowed,
    }
    reports: dict[str, dict] = {}
    finals: dict[str, dict[str, str]] = {}
    for stem in stems:
        reports[stem], found = _file_report(
            stem, task, ctx, data_dir, prereg, dict(finals)
        )
        if found is not None:
            finals[stem] = found
    before = len(rep.problems)
    selected = addendum1._select_files(
        reports, {"sample_files": primary, "replacement_file": spare}, task, rep
    )
    rep.selected[task] = selected
    rep.files[task] = {
        s: {**r, "used": s in selected} for s, r in reports.items() if r["exists"]
    }
    if not selected or len(rep.problems) > before:
        return
    for stem in selected:
        r = reports[stem]
        rep.say(_file_line(r))
        if not r["v1"]["ok"]:
            v1 = r["v1"]
            rep.fail_v(
                task,
                f"{stem} V1: {v1['value']:.3f} outside [{v1['lo']:.3f}, {v1['hi']:.3f}]",
            )
        if not r["v2"]["ok"]:
            rep.fail_v(
                task,
                f"{stem} V2: output tokens per row {r['v2']['value']} (x{r['v2']['ratio']}) outside the band",
            )
    samples = load_confirm_samples(task, selected, data_dir, rows)
    v3 = addendum1.check_v3(samples.codes, prereg["v"]["v3_tol_pts"])
    rep.v3[task] = v3
    rep.say(
        f"  V3: agreement of s1 with the others minus among the rest: "
        f"{-v3['diff_pts']:+.1f} points {'ok' if v3['ok'] else 'FAIL'}"
    )
    if not v3["ok"]:
        rep.fail_v(task, f"V3: difference {-v3['diff_pts']:+.1f} points")
    same = addendum1.identical_columns(samples.codes)
    if same:
        rep.fail(f"{task}: samples identical on every row: columns {same}")
    if not rep.v123.get(task):
        rep.say(f"[{task}] V1-V3: ok")
    else:
        rep.say(f"[{task}] V1-V3: FAILED (R2 INCONCLUSIVE)")


def _check_spend(rep: ConfirmChecks, data_dir: Path, prereg: dict) -> None:
    """Every cost file under ``labels/*/confirm/`` counts, smoke and aborted ones too."""
    rep.spend = golden_label.confirm_spend(data_dir)
    cap = prereg["budget_usd"]["total"]
    rep.say(f"confirm spend recorded: ${rep.spend:.4f} of the ${cap:.2f} cap")
    rep.holds = [
        p.relative_to(data_dir).as_posix() for p in golden_label.confirm_holds(data_dir)
    ]
    if rep.holds:
        rep.say(
            f"unreleased hold files (a run did not finish cleanly; each is charged "
            f"its whole budget): {rep.holds}"
        )
    if round(rep.spend, 9) > round(cap, 9):
        rep.fail(f"confirm spend ${rep.spend:.4f} is over the ${cap:.2f} cap")


def _data_files(data_dir: Path, prereg: dict) -> list[Path]:
    """Every file the study reads, as it exists now."""
    stems = [*prereg["sample_files"], prereg["replacement_file"]]
    out: list[Path] = [data_dir / "intent" / "labels.json"]
    for task in prereg["tasks"]:
        out += [data_dir / task / "confirm.jsonl", data_dir / task / "dev.jsonl"]
        for s in stems:
            p = sample_path(task, s, data_dir)
            out += [p, p.with_suffix(".cost.json")]
    return [p for p in out if p.exists()]


def lock_content(data_dir: Path, prereg: dict) -> dict:
    """What the lock file records: the PREREG hash and every data file's sha256."""
    return {
        "prereg_sha256": prereg_hash(prereg),
        "design_sha256": prereg["design_sha256"],
        "data_sha256": {
            p.relative_to(data_dir).as_posix(): addendum1.sha256_file(
                guard_confirm(p, data_dir)
            )
            for p in _data_files(data_dir, prereg)
        },
    }


def _check_lock(
    rep: ConfirmChecks, data_dir: Path, prereg: dict, lock_path: Path
) -> None:
    if not lock_path.exists():
        rep.need(f"{lock_path.name} (confirm.py --lock, committed before the run)")
        return
    lock = json.loads(_read_guarded(lock_path, data_dir, extra=(lock_path,)))
    bad = 0
    for hold in rep.holds:
        rep.fail(f"{hold}: a hold file appeared after the lock was written")
        bad += 1
    if lock.get("prereg_sha256") != prereg_hash(prereg):
        rep.fail("the lock file was written for a different PREREG_CONFIRM")
        bad += 1
    now = lock_content(data_dir, prereg)["data_sha256"]
    for rel, want in lock["data_sha256"].items():
        if now.get(rel) != want:
            rep.fail(
                f"{rel}: sha256 {str(now.get(rel))[:12]} differs from the lock {want[:12]}"
            )
            bad += 1
    for rel in sorted(set(now) - set(lock["data_sha256"])):
        rep.fail(f"{rel}: a data file that the lock does not list")
        bad += 1
    rep.data_sha256 = dict(now)
    if not bad:
        rep.say(f"lock file: {len(now)} data files match their sha256")


def _placeholder_task(name: str, n: int) -> replay.Task:
    """n rows of fixed fake answers, for the split-size and R0 checks before the data exists."""
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 2, n)
    wrong = rng.random(n) < 0.3
    answer = np.where(wrong, 1 - gold, gold)
    return replay.make_task(
        name=name,
        gold_kind="original",
        metric_kind="accuracy",
        ids=[f"p{i}" for i in range(n)],
        mask=np.ones((n, 1), bool),
        gold_class=np.zeros(n, int),
        value={"haiku": answer[:, None]},
        cell={"haiku": answer[:, None]},
        correct={"haiku": (answer == gold)[:, None]},
        valid={"haiku": np.ones(n, bool)},
        cost={"haiku": (1.0, n)},
        lens=np.ones(n),
        len_total=float(n),
    )


def _check_split_sizes(
    rep: ConfirmChecks, prereg: dict, real: dict[str, replay.Task]
) -> None:
    """Test rows, k and R0's lift. Before the data exists a placeholder of the right size is used."""
    eff, n_l, fracs = (
        confirm_effective(prereg),
        prereg["n_labelled"],
        prereg["k_fractions"],
    )
    for name in prereg["tasks"]:
        n = prereg["build"]["rows"][name]
        task = real.get(name) or _placeholder_task(name, n)
        train_w, test_w, _ = replay._SplitCache(task, eff).get(n_l)
        sizes = np.unique(test_w.sum(1))
        ks = [rn.k_at(f, int(sizes[0])) for f in fracs]
        worst = rn.r0_deviation(task, test_w, fracs, prereg)
        origin = (
            "" if name in real else " (placeholder rows: the data is not there yet)"
        )
        rep.say(
            f"[{name}] test rows per split {sizes.tolist()}, k at "
            f"{'/'.join(f'{float(f) * 100:g}%' for f in fracs)} = "
            f"{'/'.join(map(str, ks))}; R0 lift 1 to within {worst:.1e} on "
            f"{len(test_w)} splits{origin}"
        )
        want = prereg["expected"]
        if len(sizes) != 1 or int(sizes[0]) != want["test_rows"][name]:
            rep.fail(
                f"{name}: test rows {sizes.tolist()}, the design says "
                f"{want['test_rows'][name]}"
            )
        if ks != want["k"][name]:
            rep.fail(f"{name}: k values {ks}, the design says {want['k'][name]}")
        if worst > 1e-12:
            rep.fail(f"{name}: R0's lift is not 1 (max deviation {worst:.2e})")


def _check_reproduces_load_task(
    rep: ConfirmChecks, data_dir: Path, prereg: dict
) -> None:
    """Fed the dev Haiku file, `confirm_task` gives `replay.load_task`'s haiku arrays."""
    for name in prereg["tasks"]:
        try:
            ref = replay.load_task(name, "original", data_dir)
            dev = replay.read_jsonl(data_dir / name / "dev.jsonl", data_dir)
            answers = {
                r["id"]: r["answer"]
                for r in replay.read_jsonl(
                    data_dir / "labels" / name / "haiku.jsonl", data_dir
                )
            }
            cost = replay.read_json(
                data_dir / "labels" / name / "haiku.cost.json", data_dir
            )
        except (FileNotFoundError, PermissionError):
            rep.need(f"[{name}] dev files for the confirm_task cross-check")
            continue
        got = confirm_task(
            name,
            dev,
            normalised_finals(dev, answers),
            (float(cost["cost"]), int(cost["rows"])),
            addendum1.allowed_values(name, data_dir),
        )
        same = (
            all(
                np.array_equal(getattr(got, f)["haiku"], getattr(ref, f)["haiku"])
                for f in ("value", "cell", "correct", "valid")
            )
            and np.array_equal(got.gold_class, ref.gold_class)
            and got.u_flat["haiku"] == ref.u_flat["haiku"]
        )
        if same:
            rep.say(
                f"[{name}] confirm_task on the dev Haiku file equals replay.load_task: OK"
            )
        else:
            rep.fail(f"{name}: confirm_task does not reproduce replay.load_task on dev")


def check(
    data_dir: Path = DATA,
    prereg: dict = PREREG_CONFIRM,
    *,
    design_path: Path = DESIGN,
    parent_design: Path = rn.DESIGN,
    parent_script: Path = Path(rn.__file__),
    lock_path: Path = LOCK,
) -> ConfirmChecks:
    """Every check the design lists. It evaluates no ranker except R0.

    A part whose data is not there yet is listed in ``missing`` and the rest is
    still checked. The keyword arguments let a test point the checks at a fixture.
    """
    rep = ConfirmChecks()
    _check_design_hashes(rep, prereg, design_path, parent_design, parent_script)
    rows = _check_confirm_rows(rep, data_dir, prereg)
    for task, task_rows in rows.items():
        _check_task_samples(rep, task, task_rows, data_dir, prereg)
    _check_spend(rep, data_dir, prereg)
    _check_lock(rep, data_dir, prereg, lock_path)
    complete = not rep.missing and not rep.problems
    if complete:
        rep.loaded = {
            t: load_confirm_task(t, rep.selected[t], data_dir) for t in prereg["tasks"]
        }
    _check_split_sizes(rep, prereg, {t: v[0] for t, v in rep.loaded.items()})
    leaked = guard_selftest(data_dir)
    if leaked:
        rep.fail(f"the guard let {leaked} through")
    else:
        rep.say("guard: the release split and unlisted files are refused: OK")
    _check_reproduces_load_task(rep, data_dir, prereg)
    return rep


def lock(
    data_dir: Path = DATA,
    prereg: dict = PREREG_CONFIRM,
    lock_path: Path = LOCK,
    **kw: Any,
) -> ConfirmChecks:
    """Write the lock file after a check with no problem and no missing data."""
    if lock_path.exists():
        raise SystemExit(f"{lock_path} exists; a lock is written once")
    rep = check(data_dir, prereg, lock_path=lock_path, **kw)
    if rep.holds:
        raise SystemExit(
            f"not locking: unreleased hold files {rep.holds}. A run did not finish "
            "cleanly; check the provider's usage, then remove the hold file by hand"
        )
    open_items = [m for m in rep.missing if lock_path.name not in m]
    if rep.problems or open_items:
        raise SystemExit(
            "not locking: "
            + "; ".join([*rep.problems, *(f"missing {m}" for m in open_items)])
        )
    lock_path.write_text(
        json.dumps(lock_content(data_dir, prereg), indent=2) + "\n", encoding="utf-8"
    )
    return rep


# ── the study ────────────────────────────────────────────────────────────


def point_estimates(
    task: replay.Task, samples: addendum1.SampleSet, prereg: dict = PREREG_CONFIRM
) -> dict:
    """Precision, lift, gain and recall of every ranker over the S splits, per k fraction."""
    eff = confirm_effective(prereg)
    train_w, test_w, _ = replay._SplitCache(task, eff).get(prereg["n_labelled"])
    names = list(rn.all_rankers(prereg))
    w_total = int(np.rint(test_w.sum(1)[0]))
    return {
        "n": task.n,
        "test_rows": w_total,
        "k": {f: rn.k_at(f, w_total) for f in prereg["k_fractions"]},
        "rankers": rn.aggregate(
            rn.evaluate(
                task, samples, train_w, test_w, names, prereg["k_fractions"], prereg
            ),
            prereg["comparator"],
        ),
    }


def domain_gains(
    task: replay.Task,
    samples: addendum1.SampleSet,
    domains: list[str],
    prereg: dict = PREREG_CONFIRM,
) -> dict[str, float | None]:
    """Mean G of R2 inside each domain, at the primary k (descriptive, never decisive).

    The labelled rows are the split's, over all domains; the test rows are
    restricted to one domain by zero weights, so each domain is ranked and
    scored among its own test rows only.
    """
    eff = confirm_effective(prereg)
    train_w, test_w, _ = replay._SplitCache(task, eff).get(prereg["n_labelled"])
    primary, comparator = prereg["primary_k"], prereg["comparator"]
    out: dict[str, float | None] = {}
    for dom in sorted(set(domains)):
        mask = np.array([d == dom for d in domains], float)
        ev = rn.evaluate(
            task,
            samples,
            train_w,
            test_w * mask,
            [comparator, *prereg["confirmatory"]],
            [primary],
            prereg,
        )
        red = rn.reduce_splits(ev, 1, comparator)
        out[dom] = replay._clean(red[prereg["confirmatory"][0]][primary]["G"][0])
    return out


def run_task(
    loaded: LoadedTask, prereg: dict, n_draws: int, v_failed: bool, log: Any = print
) -> dict:
    """Everything for one task: point estimates, bootstrap, status, flags."""
    t0 = time.perf_counter()
    task, samples, rows = loaded
    ranker, primary = prereg["confirmatory"][0], prereg["primary_k"]
    res: dict[str, Any] = {"name": task.name, "n_rows": task.n}
    res["point"] = point_estimates(task, samples, prereg)
    doc_idx = cluster_index(task.name, rows, prereg)
    draws = bootstrap_confirm(task, samples, n_draws, prereg, doc_idx)
    res["bootstrap"] = {
        "draws": n_draws,
        "q": prereg["alpha"] / prereg["n_tests"],
        "cluster": prereg["cluster"][task.name],
        "clusters": task.n if doc_idx is None else int(doc_idx.max()) + 1,
        "rankers": {
            r: {
                f: {q: rn.draw_summary(x, prereg) for q, x in by_q.items()}
                for f, by_q in by_f.items()
            }
            for r, by_f in draws.items()
        },
    }
    orig = res["point"]["rankers"]
    boot = res["bootstrap"]["rankers"][ranker]
    entry = {
        "L": orig[ranker][primary]["L"],
        "G": orig[ranker][primary]["G"],
        "L_lb": boot[primary]["L"]["lb"],
        "G_lb": boot[primary]["G"]["lb"],
        "v_failed": v_failed,
        "G_by_frac": {f: orig[ranker][f]["G"] for f in prereg["k_fractions"]},
        "G_subsets": {e: orig[e][primary]["G"] for e in SUBSET_RANKERS},
    }
    res["entry"] = entry
    res["status"] = rn.task_status(entry, prereg)
    res["domain_gains"] = (
        domain_gains(task, samples, [r["domain"] for r in rows], prereg)
        if task.name == "faithfulness"
        else None
    )
    res["flags"] = flags_confirm(entry, res["domain_gains"], prereg)
    res["seconds"] = time.perf_counter() - t0
    log(f"  {task.name}: {res['seconds']:.1f}s")
    return res


def _git_state() -> dict:
    """The commit, and whether anything the run depends on has uncommitted changes.

    Looks at this study's directory (not its results), the planner and golden
    code the study imports, and every unit test, so an uncommitted edit cannot
    read as clean.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
        ).stdout.strip()

    changed = git(
        "status",
        "--porcelain",
        "--",
        "benchmarks/review_next",
        ":(exclude)benchmarks/review_next/results",
        "benchmarks/planner",
        "benchmarks/golden",
        "tests/unit",
    )
    return {"head": git("rev-parse", "HEAD"), "dirty": bool(changed)}


def run_study(
    checks: ConfirmChecks,
    prereg: dict = PREREG_CONFIRM,
    quick: bool = False,
    log: Any = print,
) -> dict:
    """Run the whole study on the loaded tasks and return the results dict."""
    t0 = time.perf_counter()
    n_draws = prereg["bootstrap_quick"] if quick else prereg["bootstrap"]
    out: dict[str, Any] = {"tasks": {}}
    for name in prereg["tasks"]:
        out["tasks"][name] = run_task(
            checks.loaded[name], prereg, n_draws, bool(checks.v123.get(name)), log
        )
    done = out["tasks"]
    out["verdict"] = {
        **verdict_confirm({t: done[t]["status"] for t in done}),
        "flags": {t: done[t]["flags"] for t in done},
        "v_failed": sorted(t for t in done if checks.v123.get(t)),
    }
    extra = prereg["sample_files"][1:]
    out["cost_per_1k_rows_extra_samples"] = {
        t: 1000.0
        * sum(
            checks.files[t][s]["cost"] / checks.files[t][s]["rows"]
            for s in extra
            if s in checks.selected[t] and s in checks.files[t]
        )
        for t in done
    }
    out["checks"] = {
        "lines": checks.lines,
        "v123": checks.v123,
        "v3": checks.v3,
        "files": checks.files,
        "selected": checks.selected,
        "spend": checks.spend,
    }
    out["meta"] = {
        "design": prereg["design"],
        "design_sha256": addendum1.sha256_file(DESIGN) if DESIGN.exists() else None,
        "prereg_confirm_sha256": prereg_hash(prereg),
        "prereg_confirm": prereg,
        "parent_design_sha256": addendum1.sha256_file(rn.DESIGN),
        "parent_prereg_sha256": replay.prereg_hash(rn.PREREG_REVIEW),
        "parent_script_sha256": addendum1.sha256_file(Path(rn.__file__)),
        "code_sha256": addendum1.sha256_file(Path(__file__)),
        "git": _git_state(),
        "data_sha256": checks.data_sha256,
        "quick": quick,
        "splits": prereg["splits"],
        "bootstrap": n_draws,
        "seed": prereg["seed"],
        "numpy": np.__version__,
        "seconds": time.perf_counter() - t0,
    }
    return out


# ── outputs ──────────────────────────────────────────────────────────────

_HEAD = (
    "| Ranker | errors per 100 reviewed | L | LB(L) | G (pts) | LB(G) | recall "
    "| rows per error |\n|---|---|---|---|---|---|---|---|"
)


def _ranker_rows(agg: dict, names: list[str], f: str, boot: dict | None) -> list[str]:
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


def _task_section(t: str, r: dict, prereg: dict) -> list[str]:
    primary, fracs = prereg["primary_k"], prereg["k_fractions"]
    point, boot = r["point"], r["bootstrap"]
    ranker = prereg["confirmatory"][0]
    shown = [prereg["null"], prereg["comparator"], *prereg["confirmatory"]]
    out = [f"## {t}\n"]
    out.append(
        f"{r['n_rows']} rows; {point['test_rows']} test rows per split; k = "
        + "/".join(str(point["k"][f]) for f in fracs)
        + f" at {'/'.join(fracs)}. Primary k fraction {primary}. Bootstrap B = "
        f"{boot['draws']} over {boot['clusters']} {boot['cluster']} clusters, LB at q = {boot['q']}.\n"
    )
    out.append(f"### Confirmatory, original gold, k fraction {primary}\n")
    out.append(_HEAD)
    out.extend(_ranker_rows(point["rankers"], shown, primary, boot["rankers"]))
    out.append("")
    g = boot["rankers"][ranker][primary]["G"]
    out.append(
        f"Status: {ranker} **{r['status']}**. Flags: {r['flags'] or 'none'}. "
        f"SE of G (bootstrap): {replay._f(g['se'], 1)}.\n"
    )
    out.append("### k sensitivity (mean G, points)\n")
    out.append("| Ranker | " + " | ".join(fracs) + " |")
    out.append("|---|" + "---|" * len(fracs))
    cells = " | ".join(replay._pts(point["rankers"][ranker][f]["G"]) for f in fracs)
    out.append(f"| {ranker} | {cells} |\n")
    out.append(f"### Exploratory (never decisive), {primary}\n")
    out.append(_HEAD)
    out.extend(
        _ranker_rows(point["rankers"], list(prereg["exploratory"]), primary, None)
    )
    out.append("")
    if r["domain_gains"]:
        out.append(f"### G of {ranker} per domain at {primary} (descriptive)\n")
        out.append("| Domain | G (pts) |\n|---|---|")
        out.extend(f"| {d} | {replay._pts(g)} |" for d, g in r["domain_gains"].items())
        out.append("")
    return out


def write_markdown(results: dict, path: Path) -> None:
    m, v = results["meta"], results["verdict"]
    prereg = m["prereg_confirm"]
    out = ["# Review-next confirmation: results\n"]
    out.append(
        f"Design: `{m['design']}` (sha256 `{m['design_sha256']}`). PREREG_CONFIRM sha256 "
        f"`{m['prereg_confirm_sha256']}`. Parent: DESIGN `{m['parent_design_sha256']}`, "
        f"PREREG `{m['parent_prereg_sha256']}`, script `{m['parent_script_sha256']}`.  \n"
        f"Script sha256 `{m['code_sha256']}`; git `{m['git']['head']}`"
        f"{' (dirty)' if m['git']['dirty'] else ' (clean)'}.  \n"
        f"Mode: {'QUICK (smoke run, not citeable)' if m['quick'] else 'full'}; "
        f"S = {m['splits']} splits, B = {m['bootstrap']} bootstrap draws, seed {m['seed']}; "
        f"run time {m['seconds']:.1f}s.\n"
    )
    out.append("## Verdict\n")
    out.append(f"**{v['verdict']}**\n")
    if v["v_failed"]:
        out.append(
            f"V1-V3 failed on: {', '.join(v['v_failed'])} (R2 INCONCLUSIVE there).\n"
        )
    out.extend(f"- FLAGS {t}: {', '.join(f) or 'none'}" for t, f in v["flags"].items())
    out.append("")
    out.append("| Task | R2 status |\n|---|---|")
    out.extend(f"| {t} | {s} |" for t, s in v["statuses"].items())
    out.append(
        "\nA pass is a ranking signal for the review step, not a router. The Planner KILL "
        "stands in every outcome. Extra samples cost per 1k rows: "
        + "; ".join(
            f"{t} ${c:.3f}"
            for t, c in results["cost_per_1k_rows_extra_samples"].items()
        )
        + ".\n"
    )
    out.append("## Checks\n")
    out.append("```")
    out.extend(results["checks"]["lines"])
    out.append("```\n")
    out.append("| Data file | sha256 |\n|---|---|")
    out.extend(f"| {k} | `{h}` |" for k, h in m["data_sha256"].items())
    out.append("")
    for t, r in results["tasks"].items():
        out.extend(_task_section(t, r, prereg))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_results(results: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "confirm.md", out_dir / "confirm.json"]
    write_markdown(results, paths[0])
    paths[1].write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return paths


def _refusal(args: argparse.Namespace, data_dir: Path = DATA) -> str | None:
    """Why this run must not start, or None (the design: one full run).

    ``--quick`` uses the full S = 1000 splits and prints the verdict, so on the
    real data it would leak the result; it is refused there until the full
    result exists, ``--out`` or not. On any other data directory (a fixture) it
    is allowed. A full run needs a clean tree and may not overwrite an earlier
    result.
    """
    full_json = (args.out or RESULTS) / "confirm.json"
    on_real_data = Path(data_dir).resolve() == DATA.resolve()
    if args.quick and on_real_data and not (RESULTS / "confirm.json").exists():
        return (
            "--quick reveals the outcome (S = 1000 splits, verdict printed): not before "
            "the full run. Use a fixture data directory to try it."
        )
    if args.quick:
        return None
    if full_json.exists():
        return f"{full_json} exists: a second full run would overwrite the result"
    if _git_state()["dirty"]:
        return (
            "the tree has uncommitted changes under benchmarks/review_next, "
            "benchmarks/planner, benchmarks/golden or tests/unit: run the one full "
            "run from the clean, reviewed commit"
        )
    return None


def main(argv: list[str] | None = None, **check_kwargs: Any) -> int:
    """CLI. `check_kwargs` go to `check` (tests point it at a fixture)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="checks only; R0 is the only ranker run"
    )
    parser.add_argument(
        "--lock", action="store_true", help="write the data lock file after the runs"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="B = 100; reveals the outcome, so on the real data it needs an existing result",
    )
    parser.add_argument("--out", type=Path, help="results directory")
    args = parser.parse_args(argv)

    if args.lock:
        rep = lock(**check_kwargs)
        print("\n".join(rep.lines))
        print(f"\nLOCK written: {check_kwargs.get('lock_path', LOCK)}")
        return 0
    refusal = None if args.check else _refusal(args, check_kwargs.get("data_dir", DATA))
    if refusal:
        print(f"REFUSED: {refusal}")
        return 2
    checks = check(**check_kwargs)
    print("\n".join(checks.lines))
    if checks.problems:
        print(f"\nCHECK FAILED ({len(checks.problems)} problems, listed above as FAIL)")
        return 2
    if checks.missing:
        print(
            f"\nCHECK INCOMPLETE: {len(checks.missing)} parts not yet available (listed above)"
        )
        return 3
    print("\nCHECK OK")
    if args.check:
        return 0

    prereg = check_kwargs.get("prereg", PREREG_CONFIRM)
    results = run_study(checks, prereg, quick=args.quick)
    out_dir = args.out or RESULTS / ("quick" if args.quick else "")
    for p in write_results(results, out_dir):
        print(f"wrote {p.relative_to(REPO) if p.is_relative_to(REPO) else p}")
    print(f"VERDICT: {results['verdict']['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
