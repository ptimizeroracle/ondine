"""benchmarks/review_next/confirm.py on synthetic data.

The confirmation is pre-registered: nobody may look at a ranker's precision, lift
or gain on the real confirm rows before the code is reviewed, and those rows
have no model answers yet. Every test therefore builds a task, samples or a
golden-layout fixture whose right answer follows from how it is constructed.
Nothing reads the real data directory, nothing calls a model. Each test names
the regression it would catch.
"""

from __future__ import annotations

import builtins
import dataclasses
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from test_planner_replay import (  # noqa: E402
    INTENT_NAMES,
    STRONG,  # noqa: F401  (kept for fixtures that need the parent's strong map)
    make_layout,
    write_jsonl,
)
from test_review_next import planted  # noqa: E402

_ROOT = _HERE.parents[1]
_MODULE = _ROOT / "benchmarks" / "review_next" / "confirm.py"
_spec = importlib.util.spec_from_file_location("review_next_confirm", _MODULE)
cf = importlib.util.module_from_spec(_spec)
sys.modules["review_next_confirm"] = cf
_spec.loader.exec_module(cf)
rn = cf.rn
replay = cf.replay
addendum1 = cf.addendum1
label = cf.golden_label

P = cf.PREREG_CONFIRM
PRIMARY = P["primary_k"]
FRACS = P["k_fractions"]


def small(**over):
    """PREREG_CONFIRM reduced for speed: 60 splits, 16 bootstrap draws of 5 splits."""
    return {
        **P,
        "splits": 60,
        "bootstrap": 16,
        "bootstrap_quick": 16,
        "bootstrap_splits_per_draw": 5,
        **over,
    }


# ── the guard ────────────────────────────────────────────────────────────

D = Path("/data")  # a path that need not exist: the guard works on names


class TestGuard:
    @pytest.mark.parametrize(
        "rel",
        [
            "intent/confirm.jsonl",
            "faithfulness/confirm.jsonl",
            "intent/dev.jsonl",
            "faithfulness/dev.jsonl",
            "intent/labels.json",
            "labels/intent/confirm/haiku-s1.jsonl",
            "labels/intent/confirm/haiku-s5.cost.json",
            "labels/faithfulness/confirm/haiku-s3.jsonl",
            "labels/faithfulness/confirm/haiku-s4.cost.json",
        ],
    )
    def test_the_listed_files_are_allowed(self, rel):
        # Catches an allowlist that is too tight to run the study.
        assert cf.guard_confirm(D / rel, D) == (D / rel).resolve()

    @pytest.mark.parametrize(
        "rel",
        [
            "intent/holdout.jsonl",
            "faithfulness/Holdout.jsonl",
            "labels/intent/confirm/holdout.jsonl",
            "labels/intent/holdout/haiku-s1.jsonl",
            "labels/intent/haiku.jsonl",  # a dev label file
            "labels/intent/haiku-s1.jsonl",  # a dev sample
            "labels/intent/confirm/haiku-s6.jsonl",
            "labels/intent/confirm/haiku-s0.jsonl",
            "labels/intent/confirm/haiku-s1-smoke3.jsonl",
            "labels/intent/confirm/haiku-s1-aborted.cost.json",
            "labels/intent/confirm/luna.jsonl",
            "labels/attributes/confirm/haiku-s1.jsonl",
            "intent/dev-adjudicated.jsonl",
            "attributes/dev.jsonl",
            "attributes/confirm.jsonl",
            "attributes/schemas.json",
            "faithfulness/labels.json",
            "baseline-report.json",
            "labels/intent/confirm/../haiku.jsonl",
            "intent/confirm/../holdout.jsonl",
            "intent/confirm.jsonl.bak",
        ],
    )
    def test_everything_else_is_refused_before_any_read(self, rel):
        # Catches a loosened guard: the release split, dev labels, an unlisted
        # sample index, smoke and aborted files and traversal all stay out.
        with pytest.raises(PermissionError):
            cf.guard_confirm(D / rel, D)

    def test_a_path_outside_the_data_directory_is_refused(self):
        with pytest.raises(PermissionError, match="outside"):
            cf.guard_confirm("/etc/passwd", D)
        with pytest.raises(PermissionError):
            cf.guard_confirm(D.parent / "other" / "intent" / "confirm.jsonl", D)

    def test_the_extra_paths_allow_exactly_the_lock_file(self, tmp_path):
        lock = tmp_path / "x.lock.json"

        assert cf.guard_confirm(lock, D, extra=(lock,)) == lock.resolve()
        with pytest.raises(PermissionError):
            cf.guard_confirm(tmp_path / "y.json", D, extra=(lock,))

    def test_the_selftest_names_a_leak(self, monkeypatch):
        # Catches a selftest that cannot fail: with a guard that allows all it must report.
        assert cf.guard_selftest(D) == []
        monkeypatch.setattr(cf, "guard_confirm", lambda p, d=D, extra=(): Path(p))

        assert len(cf.guard_selftest(D)) >= 10


class TestSourceHygiene:
    def test_the_release_word_never_appears_and_there_is_one_guarded_raw_read(self):
        # Catches a second way to read a file, and any mention of the release split.
        text = _MODULE.read_text()

        assert "holdout" not in text.lower()
        for raw in ("open(", "read_text(", "json.load("):
            assert raw not in text, f"confirm.py reads files itself: {raw}"
        assert text.count("read_bytes(") == 1
        assert re.search(r"guard_confirm\([^)]*\)\.read_bytes\(\)", text)
        body = text[text.index("def _read_guarded") :].split("\n\n\n")[0]
        assert "read_bytes(" in body

    def test_no_path_with_the_release_word_is_opened_during_a_check_or_a_run(
        self, full, monkeypatch, tmp_path
    ):
        # The fixture holds a garbage release file: record every read attempt and
        # prove none names it, in a check and in a whole run.
        seen: list[str] = []
        real_bytes, real_text, real_open = (
            Path.read_bytes,
            Path.read_text,
            builtins.open,
        )

        def spy_bytes(self, *a, **k):
            seen.append(str(self))
            return real_bytes(self, *a, **k)

        def spy_text(self, *a, **k):
            seen.append(str(self))
            return real_text(self, *a, **k)

        def spy_open(file, *a, **k):
            seen.append(str(file))
            return real_open(file, *a, **k)

        monkeypatch.setattr(Path, "read_bytes", spy_bytes)
        monkeypatch.setattr(Path, "read_text", spy_text)
        monkeypatch.setattr(builtins, "open", spy_open)
        monkeypatch.setattr(cf, "_git_state", lambda: CLEAN)
        monkeypatch.setattr(cf, "RESULTS", tmp_path / "results")
        release = full.data / "intent" / ("hold" + "out.jsonl")
        assert release.exists()  # the fixture has one to tempt the loader

        assert cf.main(["--out", str(tmp_path / "o")], **full.kwargs) == 0

        assert seen
        assert not [p for p in seen if "hold" + "out" in p.lower()]


# ── tasks and samples ────────────────────────────────────────────────────

# The layout fixture: make_layout's dev rows and five models' labels, plus
# confirm rows and four (five) Haiku sample files for intent and faithfulness.

CONF_INTENT = [
    {"id": f"ci{k}", "text": f"confirm msg {k}", "label": INTENT_NAMES[k % 3]}
    for k in range(30)
]
FAITH_DOMAINS = ("d1", "d2")
CONF_FAITH = [
    {
        "id": f"cf-{dom}-{lab[0]}{k}",
        "domain": dom,
        "doc": f"{dom}-doc{'ABCA'[k]}",
        "summary": f"summary {dom} {lab} {k}",
        "label": lab,
    }
    for dom in FAITH_DOMAINS
    for lab in ("consistent", "inconsistent")
    for k in range(4)
]
ROWS = {"intent": CONF_INTENT, "faithfulness": CONF_FAITH}
STEMS = P["sample_files"]
SPARE = P["replacement_file"]
#: Row positions each file gets wrong: files differ, so none is a copy of another.
FLIPS = {
    "haiku-s1": (0,),
    "haiku-s2": (0, 5),
    "haiku-s3": (1,),
    "haiku-s4": (2,),
    "haiku-s5": (3,),
}
TOKENS_PER_ROW = 100


def wrong_answer(task: str, label_: str) -> str:
    if task == "intent":
        return "lost_card" if label_ != "lost_card" else "card_arrival"
    return "inconsistent" if label_ == "consistent" else "consistent"


def answers(task: str, stem: str):
    flips = FLIPS[stem]
    return [
        {
            "id": r["id"],
            "answer": wrong_answer(task, r["label"]) if k in flips else r["label"],
        }
        for k, r in enumerate(ROWS[task])
    ]


def cost_record(task: str, **over):
    n = len(ROWS[task])
    return {
        "route": P["route"],
        "model": P["model_id"],
        "cost": 0.01,
        "tokens": 5000,
        "rows": n,
        "input_tokens": 4000,
        "output_tokens": TOKENS_PER_ROW * n,
        "served_models": [],
        "split": "confirm",
        **over,
    }


def write_sample(
    data: Path, task: str, stem: str, answer_rows=None, **cost_over
) -> None:
    folder = data / "labels" / task / "confirm"
    write_jsonl(folder / f"{stem}.jsonl", answer_rows or answers(task, stem))
    (folder / f"{stem}.cost.json").write_text(
        json.dumps(cost_record(task, **cost_over))
    )


def build_layout(root: Path, samples=STEMS) -> Path:
    data = make_layout(root)
    for task in cf.TASKS:
        write_jsonl(data / task / "confirm.jsonl", ROWS[task])
        for stem in samples:
            write_sample(data, task, stem)
    return data


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_prereg(data: Path, **over) -> dict:
    """PREREG_CONFIRM for the 30 + 16 row fixture, its pins taken from the files."""
    return {
        **P,
        "n_labelled": 8,
        "splits": 20,
        "bootstrap": 8,
        "bootstrap_quick": 8,
        "bootstrap_splits_per_draw": 2,
        "build": {
            **P["build"],
            "rows": {"intent": 30, "faithfulness": 16},
            "intent_per_class": 10,
            "faithfulness_per_domain_label": 4,
            "faithfulness_cap_per_doc_label": 2,
        },
        "confirm_split_sha256": {t: sha(data / t / "confirm.jsonl") for t in cf.TASKS},
        "expected": {
            "test_rows": {"intent": 22, "faithfulness": 8},
            "k": {
                "intent": [rn.k_at(f, 22) for f in FRACS],
                "faithfulness": [rn.k_at(f, 8) for f in FRACS],
            },
        },
        "v": {
            "v1": {
                "intent": {"metric": "accuracy", "center": 0.9, "tol": 0.5},
                "faithfulness": {"metric": "balanced", "center": 0.9, "tol": 0.5},
            },
            "v2": {
                "low": 0.5,
                "high": 2.0,
                "implied": {"intent": 100.0, "faithfulness": 100.0},
            },
            "v3_tol_pts": 100.0,
            "v4_max_bad_pct": 3.0,
        },
        **over,
    }


CLEAN = {"head": "abc", "dirty": False}


class Fx:
    """A fixture layout, its PREREG and the keyword arguments for `check`."""

    def __init__(self, root: Path, samples=STEMS):
        self.root = root
        self.data = build_layout(root, samples)
        self.lock = root / "confirm-data.lock.json"
        self.prereg = fixture_prereg(self.data)

    @property
    def kwargs(self) -> dict:
        return {"data_dir": self.data, "prereg": self.prereg, "lock_path": self.lock}

    def check(self, **over):
        return cf.check(**{**self.kwargs, **over})

    def refreeze(self) -> None:
        """Re-pin the confirm files after a test edits them."""
        self.prereg = fixture_prereg(self.data)

    def sample(self, task: str, stem: str) -> Path:
        return self.data / "labels" / task / "confirm" / f"{stem}.jsonl"


@pytest.fixture
def fx(tmp_path):
    return Fx(tmp_path)


@pytest.fixture
def full(tmp_path):
    """A complete fixture with its lock file written."""
    f = Fx(tmp_path)
    cf.lock(**f.kwargs)
    return f


class TestConfirmTask:
    def test_it_reproduces_load_tasks_haiku_arrays_on_the_dev_rows(self, tmp_path):
        # Catches a confirm_task whose codes, validity, correctness, classes or
        # cost drift from the parent's loader: every number would then differ.
        data = make_layout(tmp_path)
        for name in cf.TASKS:
            ref = replay.load_task(name, "original", data)
            dev = replay.read_jsonl(data / name / "dev.jsonl", data)
            ans = {
                r["id"]: r["answer"]
                for r in replay.read_jsonl(data / "labels" / name / "haiku.jsonl", data)
            }
            cost = replay.read_json(data / "labels" / name / "haiku.cost.json", data)

            got = cf.confirm_task(
                name,
                dev,
                cf.normalised_finals(dev, ans),
                (float(cost["cost"]), int(cost["rows"])),
                addendum1.allowed_values(name, data),
            )

            for field in ("value", "cell", "correct", "valid"):
                np.testing.assert_array_equal(
                    getattr(got, field)["haiku"], getattr(ref, field)["haiku"]
                )
            np.testing.assert_array_equal(got.gold_class, ref.gold_class)
            assert got.u_flat["haiku"] == ref.u_flat["haiku"]
            assert got.metric_kind == ref.metric_kind
            assert got.ids == ref.ids
            np.testing.assert_allclose(got.u_len["haiku"], ref.u_len["haiku"])

    def test_it_holds_one_model_only(self, tmp_path):
        data = make_layout(tmp_path)
        dev = replay.read_jsonl(data / "intent" / "dev.jsonl", data)

        t = cf.confirm_task("intent", dev, ["x"] * len(dev), (1.0, len(dev)), {"x"})

        assert t.models == ("haiku",)
        assert list(t.correct) == ["haiku"]

    def test_a_null_answer_is_none_and_invalid(self, tmp_path):
        # Catches a null answer scored as a valid value (the parent maps it to "none").
        rows = [
            {"id": "a", "text": "t", "label": "top_up"},
            {"id": "b", "text": "t", "label": "top_up"},
        ]
        finals = cf.normalised_finals(rows, {"a": "top_up", "b": None})

        task = cf.confirm_task("intent", rows, finals, (1.0, 2), {"topup"})

        assert finals == ["topup", "none"]
        assert task.valid["haiku"].tolist() == [True, False]
        assert task.correct["haiku"][:, 0].tolist() == [True, False]


class TestSamples:
    def test_column_j_is_file_s_j_plus_1_and_column_0_is_the_delivered_answer(self, fx):
        # Catches a column order that scrambles D and the E4-E7 subsets: column j
        # differs from column 0 exactly where file s(j+1) and s1 disagree.
        samples = cf.load_confirm_samples("intent", data_dir=fx.data)

        assert samples.ids == [r["id"] for r in CONF_INTENT]
        assert samples.n_columns == 4
        for j, stem in enumerate(STEMS):
            differs = [
                (k in FLIPS["haiku-s1"]) != (k in FLIPS[stem]) for k in range(30)
            ]
            np.testing.assert_array_equal(
                samples.codes[:, j] != samples.codes[:, 0], differs
            )

    def test_a_null_answer_gets_its_own_code_and_counts_as_dissent(self, fx):
        # Catches a null read as a real answer: two nulls must never agree.
        rows = answers("intent", "haiku-s2")
        rows[4] = {"id": rows[4]["id"], "answer": None}
        write_jsonl(fx.sample("intent", "haiku-s2"), rows)

        samples = cf.load_confirm_samples("intent", data_dir=fx.data)

        assert samples.codes[4, 1] != samples.codes[4, 0]
        assert not samples.valid[4, 1]
        assert rn.dissent(samples)[4] >= 1

    def test_the_delivered_task_is_s1_and_aligned_with_the_samples(self, fx):
        # Catches the delivered answer taken from another file: s1 gets row 0 wrong only.
        task, samples, rows = cf.load_confirm_task("faithfulness", data_dir=fx.data)

        assert task.ids == samples.ids == [r["id"] for r in rows]
        np.testing.assert_array_equal(task.value["haiku"][:, 0], samples.codes[:, 0])
        assert (~task.correct["haiku"][:, 0]).tolist() == [True] + [False] * 15

    def test_the_replacement_file_takes_the_failed_files_column(self, fx):
        # Catches a replacement appended as a fifth column: D would count 4 samples.
        write_sample(fx.data, "intent", "haiku-s5")
        samples = cf.load_confirm_samples(
            "intent", ["haiku-s1", "haiku-s2", "haiku-s5", "haiku-s4"], fx.data
        )

        assert samples.n_columns == 4
        differs = [
            (k in FLIPS["haiku-s1"]) != (k in FLIPS["haiku-s5"]) for k in range(30)
        ]
        np.testing.assert_array_equal(
            samples.codes[:, 2] != samples.codes[:, 0], differs
        )


class TestSignals:
    def loaded(self, fx):
        task, samples, _ = cf.load_confirm_task("intent", data_dir=fx.data)
        return task, samples

    def test_d_counts_the_files_s2_to_s4_that_differ_from_s1(self, fx):
        # Catches D computed against the mode, or over a wrong column range.
        task, samples = self.loaded(fx)

        d = rn.dissent(samples)

        want = [
            sum(((k in FLIPS["haiku-s1"]) != (k in FLIPS[s])) for s in STEMS[1:])
            for k in range(30)
        ]
        assert d.tolist() == want

    def test_exploratory_subsets_read_column_indices_not_file_numbers(self, fx):
        # E4 = "D:s1,s2" is columns 1 and 2, i.e. the files s2 and s3.
        task, samples = self.loaded(fx)
        train_w = np.ones((1, task.n))

        keys = rn.ranker_keys("E4", task, samples, train_w, P)

        np.testing.assert_array_equal(keys[0, :, 0], rn.dissent(samples, [1, 2]))
        s2_s3 = [
            sum(((k in FLIPS["haiku-s1"]) != (k in FLIPS[s])) for s in STEMS[1:3])
            for k in range(30)
        ]
        assert rn.dissent(samples, [1, 2]).tolist() == s2_s3

    def test_the_partner_is_the_delivered_model_so_p_is_zero_and_no_ranker_reads_it(
        self, fx
    ):
        # Catches a Luna call sneaking back in: P = 0 on valid rows, no key has a P column.
        task, samples = self.loaded(fx)

        assert P["partner"] == P["delivered"] == "haiku"
        assert not rn.pair_disagree(task, "haiku", "haiku").any()
        specs = [*P["rankers"].values(), *P["exploratory"].values()]
        assert all(col not in ("P", "D+P") for spec in specs for col in spec)
        keys = rn.ranker_keys("R2", task, samples, np.ones((1, task.n)), P)
        assert keys.shape[-1] == 2  # D and r1 only

    def test_r2_is_d_then_r1(self, fx):
        task, samples = self.loaded(fx)
        train_w = np.zeros((1, task.n))
        train_w[0, :8] = 1

        keys = rn.ranker_keys("R2", task, samples, train_w, P)

        np.testing.assert_array_equal(keys[0, :, 0], rn.dissent(samples))
        np.testing.assert_allclose(
            keys[0, :, 1], rn.ranker_keys("R1", task, samples, train_w, P)[0, :, 0]
        )


# ── the document-cluster bootstrap ───────────────────────────────────────

DOC_IDX = np.repeat(np.arange(5), [3, 1, 4, 2, 2])  # 12 rows, 5 documents


def doc_batches(seed=0, n_draws=7, per_draw=3, chunk=10, n_l=4, doc_idx=DOC_IDX):
    return list(
        cf.doc_bootstrap_batches(
            np.random.default_rng(seed), doc_idx, n_l, n_draws, per_draw, chunk
        )
    )


class TestDocBootstrap:
    def test_rows_of_one_document_share_a_weight_and_weights_sum_to_the_documents(
        self,
    ):
        # Catches row-level weights (the parent's), which would understate the
        # variance of a task whose rows cluster in documents.
        for train_w, test_w, _ in doc_batches():
            w = train_w + test_w  # each row is on one side only
            for k in range(5):
                block = w[:, k == DOC_IDX]
                assert (block == block[:, :1]).all()
            first_rows = [int(np.flatnonzero(k == DOC_IDX)[0]) for k in range(5)]
            np.testing.assert_array_equal(w[:, first_rows].sum(1), 5.0)

    def test_a_document_can_have_rows_on_both_sides_because_labelled_rows_are_drawn_over_rows(
        self,
    ):
        # Catches labelled rows drawn over documents: the Ruler labels rows.
        both = False
        for train_w, test_w, _ in doc_batches(n_draws=40):
            assert ((train_w > 0).sum(1) <= 4).all()
            for k in range(5):
                rows = k == DOC_IDX
                both |= bool(
                    ((train_w[:, rows] > 0).any(1) & (test_w[:, rows] > 0).any(1)).any()
                )
        assert both

    def test_each_split_has_n_l_labelled_rows_before_weights(self):
        # With all-ones weights the split sizes are visible: use 1 document per row.
        for train_w, test_w, _ in doc_batches(doc_idx=np.arange(12), n_l=4):
            assert ((train_w > 0).sum(1) <= 4).all()
            assert (test_w.sum(1) + train_w.sum(1) == 12).all()

    def test_with_every_row_its_own_document_it_is_the_parents_stream(self):
        # Catches a different draw order: rows-as-clusters must equal replay.bootstrap_batches.
        own = np.arange(12)
        mine = doc_batches(seed=3, doc_idx=own)
        theirs = list(
            replay.bootstrap_batches(np.random.default_rng(3), 12, 4, 7, 3, 10)
        )

        assert len(mine) == len(theirs)
        for a, b in zip(mine, theirs, strict=True):
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
            assert a[2] == b[2]

    def test_the_splits_of_a_draw_share_its_weights_and_the_chunks_cover_all_draws(
        self,
    ):
        # 7 draws, chunk 10 // 3 = 3 draws per chunk: 3 + 3 + 1.
        batches = doc_batches()

        assert [b[2] for b in batches] == [3, 3, 1]
        for train_w, test_w, d in batches:
            w = (train_w + test_w).reshape(d, 3, -1)
            # The union of a draw's splits can hide a row's weight on one side only;
            # the per-document weight is what must agree across its three splits.
            first = [int(np.flatnonzero(k == DOC_IDX)[0]) for k in range(5)]
            per_split = w[:, :, first]
            assert (per_split == per_split[:, :1, :]).all()

    def test_it_is_deterministic_in_the_generator(self):
        a, b = doc_batches(seed=9), doc_batches(seed=9)

        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x[0], y[0])
        assert not np.array_equal(a[0][0], doc_batches(seed=10)[0][0])

    def test_cluster_index_is_none_for_rows_and_dense_for_documents(self):
        rows = [{"doc": d} for d in ["b", "a", "b", "c"]]

        assert cf.cluster_index("intent", rows) is None
        assert cf.cluster_index("faithfulness", rows).tolist() == [1, 0, 1, 2]


class TestBootstrapConfirm:
    def test_a_ranker_equal_to_r1_has_gain_exactly_zero_in_every_draw(self):
        # Catches an unpaired bootstrap: R2 with no dissent orders by r1 alone, so
        # G is 0 in every draw, for row clusters and for document clusters.
        task, samples, _ = planted("none")
        for doc_idx in (None, np.arange(task.n) // 3):
            draws = cf.bootstrap_confirm(task, samples, 4, small(), doc_idx)

            for f in FRACS:
                assert (draws["R2"][f]["G"] == 0.0).all()

    def test_it_returns_l_and_g_of_r2_only_at_every_k(self):
        # Catches R4 (or any other ranker) left in the confirmatory family.
        task, samples, _ = planted("errors")

        draws = cf.bootstrap_confirm(task, samples, 4, small())

        assert list(draws) == ["R2"]
        for f in FRACS:
            assert set(draws["R2"][f]) == {"L", "G"}
            assert draws["R2"][f]["G"].shape == (4,)

    def test_the_lower_bound_is_the_0025_quantile(self):
        # Catches the parent's 0.0125 (family of 4) or 0.0167 (family of 3).
        draws = np.linspace(-5.0, 10.0, 401)

        lb = rn.draw_summary(draws, P)["lb"]

        assert lb == pytest.approx(np.quantile(draws, 0.025))
        assert lb != pytest.approx(np.quantile(draws, 0.0125))
        assert lb != pytest.approx(np.quantile(draws, 0.05 / 3))
        assert P["n_tests"] == 2

    def test_row_clusters_use_the_confirm_seed_on_the_parents_stream(self, monkeypatch):
        # Pins purpose 1, gold "original", n_l = 100, 20 splits per draw, chunk 1000
        # and this design's seed 20261118 (not the parent's 20261117).
        task, samples, _ = planted("errors")
        real = replay.bootstrap_batches
        seen: list = []

        def spy(rng, n, n_l, n_draws, per_draw, chunk):
            seen.append((n, n_l, n_draws, per_draw, chunk))
            for batch in real(rng, n, n_l, n_draws, per_draw, chunk):
                seen.append(batch)
                yield batch

        monkeypatch.setattr(replay, "bootstrap_batches", spy)
        cf.bootstrap_confirm(task, samples, 4, P)
        monkeypatch.setattr(replay, "bootstrap_batches", real)

        def stream(seed):
            return list(
                real(
                    replay.split_rng(
                        {**replay.PREREG, "seed": seed}, task.name, "original", 1, 100
                    ),
                    task.n,
                    100,
                    4,
                    20,
                    1000,
                )
            )

        assert seen[0] == (task.n, 100, 4, 20, 1000)
        for got, ref in zip(seen[1:], stream(20261118), strict=True):
            np.testing.assert_array_equal(got[0], ref[0])
            np.testing.assert_array_equal(got[1], ref[1])
        assert not np.array_equal(seen[1][0], stream(20261117)[0][0])

    def test_document_clusters_use_the_same_stream_through_the_doc_function(
        self, monkeypatch
    ):
        task, samples, _ = planted("errors")
        doc_idx = np.arange(task.n) // 3
        seen: list = []
        real = cf.doc_bootstrap_batches

        def spy(rng, *args, **kwargs):
            for batch in real(rng, *args, **kwargs):
                seen.append(batch)
                yield batch

        monkeypatch.setattr(cf, "doc_bootstrap_batches", spy)
        cf.bootstrap_confirm(task, samples, 4, P, doc_idx)
        want = list(
            real(
                replay.split_rng(
                    {**replay.PREREG, "seed": 20261118}, task.name, "original", 1, 100
                ),
                doc_idx,
                100,
                4,
                20,
                1000,
            )
        )

        assert len(seen) == len(want)
        for a, b in zip(seen, want, strict=True):
            np.testing.assert_array_equal(a[0], b[0])

    def test_the_draws_are_reproducible(self):
        task, samples, _ = planted("errors")

        a = cf.bootstrap_confirm(task, samples, 4, small())
        b = cf.bootstrap_confirm(task, samples, 4, small())

        np.testing.assert_array_equal(a["R2"][PRIMARY]["G"], b["R2"][PRIMARY]["G"])


# ── the study, verdict and flags ─────────────────────────────────────────


def loaded_planted(signal: str, name: str = "intent"):
    task, samples, _ = planted(signal)
    rows = [
        {"id": rid, "doc": f"doc{k // 3}", "domain": f"d{k % 4}"}
        for k, rid in enumerate(task.ids)
    ]
    return dataclasses.replace(task, name=name), samples, rows


def run_planted(signal: str, name: str = "intent", v_failed: bool = False):
    return cf.run_task(
        loaded_planted(signal, name), small(), 16, v_failed, log=lambda *_: None
    )


class TestRunTask:
    def test_a_planted_signal_passes_and_a_dissent_without_information_is_killed(self):
        # Catches a status read from the wrong ranker, k or bound. "errors": D marks
        # every error, so R2 passes. "none": D is constant, R2 reduces to R1, G = 0.
        good = run_planted("errors")
        bad = run_planted("none")

        assert good["status"] == "PASS"
        assert good["entry"]["G"] > 50
        assert bad["status"] == "KILL"
        assert bad["entry"]["G"] == pytest.approx(0.0, abs=1e-9)
        assert bad["entry"]["L"] < 1.5

    def test_a_dissent_independent_of_the_errors_never_passes_and_gains_nothing_on_average(
        self,
    ):
        # One dataset gives a lucky or unlucky gain; over many datasets the mean is
        # near 0 and none of them may PASS.
        results = []
        for seed in range(8):
            task, samples, _ = planted("independent", seed=seed)
            rows = [{"id": i, "doc": f"x{k // 3}"} for k, i in enumerate(task.ids)]
            results.append(
                cf.run_task(
                    (task, samples, rows), small(splits=200), 16, False, lambda *_: None
                )
            )

        assert "PASS" not in [r["status"] for r in results]
        assert abs(np.mean([r["entry"]["G"] for r in results])) < 5.0

    def test_the_entry_reads_r2_at_the_primary_k_and_its_bootstrap_bounds(self):
        res = run_planted("errors")

        orig = res["point"]["rankers"]["R2"][PRIMARY]
        boot = res["bootstrap"]["rankers"]["R2"][PRIMARY]
        assert res["entry"]["L"] == orig["L"]
        assert res["entry"]["G"] == orig["G"]
        assert res["entry"]["L_lb"] == boot["L"]["lb"]
        assert res["entry"]["G_lb"] == boot["G"]["lb"]
        assert res["entry"]["G_subsets"].keys() == {"E4", "E5", "E6"}

    def test_a_v_failure_forces_inconclusive_even_when_the_numbers_pass(self):
        assert run_planted("errors", v_failed=True)["status"] == "INCONCLUSIVE"

    def test_faithfulness_clusters_on_documents_and_reports_domain_gains(self):
        # Catches the row bootstrap on a task whose rows share documents.
        res = run_planted("errors", name="faithfulness")

        assert res["bootstrap"]["cluster"] == "doc"
        assert res["bootstrap"]["clusters"] == 77  # 231 rows, 3 per document
        assert set(res["domain_gains"]) == {"d0", "d1", "d2", "d3"}
        assert all(g > 0 for g in res["domain_gains"].values())

    def test_intent_clusters_on_rows_and_has_no_domain_gains(self):
        res = run_planted("errors")

        assert res["bootstrap"]["cluster"] == "row"
        assert res["bootstrap"]["clusters"] == 231
        assert res["domain_gains"] is None

    def test_the_point_table_has_every_ranker_and_the_exploratory_six(self):
        res = run_planted("errors")

        assert set(res["point"]["rankers"]) == {"R0", "R1", "R2", *P["exploratory"]}
        assert len(P["exploratory"]) == 6
        assert res["point"]["k"][PRIMARY] == rn.k_at(PRIMARY, res["point"]["test_rows"])

    def test_the_lift_of_random_is_one_and_gain_of_r1_is_zero(self):
        # Catches a comparator that is not R1.
        res = run_planted("errors")

        assert res["point"]["rankers"]["R0"][PRIMARY]["L"] == pytest.approx(1.0)
        assert res["point"]["rankers"]["R1"][PRIMARY]["G"] == pytest.approx(0.0)


class TestVerdict:
    @pytest.mark.parametrize(
        ("faith", "intent", "want"),
        [
            ("PASS", "PASS", "CONFIRMED_GENERAL"),
            ("PASS", "KILL", "CONFIRMED(faithfulness)"),
            ("PASS", "INCONCLUSIVE", "CONFIRMED(faithfulness)"),
            ("KILL", "PASS", "NOT_REPLICATED + NEW(intent)"),
            ("KILL", "KILL", "NOT_REPLICATED"),
            ("KILL", "INCONCLUSIVE", "NOT_REPLICATED"),
            ("INCONCLUSIVE", "PASS", "INCONCLUSIVE"),
            ("INCONCLUSIVE", "KILL", "INCONCLUSIVE"),
            ("INCONCLUSIVE", "INCONCLUSIVE", "INCONCLUSIVE"),
        ],
    )
    def test_the_truth_table_of_design_section_5(self, faith, intent, want):
        # Catches a swapped branch: faithfulness KILL plus intent PASS is
        # NOT_REPLICATED with NEW(intent), never CONFIRMED_GENERAL.
        got = cf.verdict_confirm({"faithfulness": faith, "intent": intent})

        assert got["verdict"] == want
        assert got["new"] == (["NEW(intent)"] if "NEW" in want else [])

    def test_boundaries_g_10_passes_and_9_99_does_not_g_5_is_not_kill(self):
        # Thresholds compared exactly, as in the parent.
        base = {"L": 2.0, "L_lb": 1.01, "G": 10.0, "G_lb": 0.01, "v_failed": False}

        assert rn.task_status(base, P) == "PASS"
        assert rn.task_status({**base, "G": 9.99}, P) == "INCONCLUSIVE"
        assert rn.task_status({**base, "G": 4.99}, P) == "KILL"
        assert rn.task_status({**base, "G": 5.0, "L": 1.49}, P) == "KILL"
        assert rn.task_status({**base, "G_lb": 0.0}, P) == "INCONCLUSIVE"
        assert rn.task_status({**base, "L_lb": 1.0}, P) == "INCONCLUSIVE"


class TestFlags:
    entry = {
        "G_by_frac": {"0.05": 12.0, "0.10": 20.0, "0.20": 15.0},
        "G_subsets": {"E4": 12.0, "E5": 30.0, "E6": 10.0},
    }

    def test_none_when_every_k_gains_and_every_subset_holds(self):
        assert cf.flags_confirm(self.entry, None, P) == []

    @pytest.mark.parametrize("frac", ["0.05", "0.20"])
    def test_k_fragile_fires_at_5_or_20_percent_at_gain_zero_but_not_at_10(self, frac):
        entry = {**self.entry, "G_by_frac": {**self.entry["G_by_frac"], frac: 0.0}}
        assert "K_FRAGILE" in cf.flags_confirm(entry, None, P)
        assert "K_FRAGILE" not in cf.flags_confirm(
            {**self.entry, "G_by_frac": {**self.entry["G_by_frac"], "0.10": -3.0}},
            None,
            P,
        )

    def test_subset_fires_below_10_points_in_any_of_e4_to_e6(self):
        entry = {**self.entry, "G_subsets": {"E4": 12.0, "E5": 9.99, "E6": 30.0}}

        assert "SUBSET" in cf.flags_confirm(entry, None, P)
        assert "SUBSET" not in cf.flags_confirm(
            {**self.entry, "G_subsets": {"E4": 10.0}}, None, P
        )

    def test_domain_fires_at_three_domains_with_gain_at_most_zero(self):
        # Catches a threshold at 2 or 4; None (undefined) does not count.
        gains = {f"d{i}": 5.0 for i in range(8)}

        assert "DOMAIN" not in cf.flags_confirm(
            self.entry, {**gains, "d0": 0.0, "d1": -1.0, "d2": None}, P
        )
        assert "DOMAIN" in cf.flags_confirm(
            self.entry, {**gains, "d0": 0.0, "d1": -1.0, "d2": -0.1}, P
        )
        assert "DOMAIN" not in cf.flags_confirm(self.entry, None, P)

    def test_domain_gains_are_zero_when_r2_is_r1(self):
        # Catches a per-domain G that mixes in other domains' test rows.
        task, samples, rows = loaded_planted("none", "faithfulness")

        gains = cf.domain_gains(task, samples, [r["domain"] for r in rows], small())

        assert set(gains.values()) == {0.0}

    def test_domain_gains_come_from_each_domains_own_test_rows(self):
        # Catches a per-domain G computed over every domain's rows: dissent marks the
        # errors in domain d0 only, so d0 gains and the others gain exactly nothing.
        task, samples, rows = loaded_planted("errors", "faithfulness")
        domains = [r["domain"] for r in rows]
        in_d0 = np.array([d == "d0" for d in domains])
        codes = samples.codes.copy()
        codes[~in_d0, 1:] = codes[~in_d0, :1]
        quiet = dataclasses.replace(samples, codes=codes)

        gains = cf.domain_gains(task, quiet, domains, small())

        assert gains["d0"] > 20
        assert {gains["d1"], gains["d2"], gains["d3"]} == {0.0}

    def test_flags_never_change_the_status(self):
        res = run_planted("errors", name="faithfulness")

        assert res["status"] == "PASS"
        entry = {**res["entry"], "G_by_frac": {"0.05": -1.0, "0.10": 20.0, "0.20": 1.0}}
        assert "K_FRAGILE" in cf.flags_confirm(entry, None, P)
        assert rn.task_status(entry, P) == "PASS"


# ── the pins ─────────────────────────────────────────────────────────────


class TestPreregPins:
    def test_the_parent_pins_are_the_real_parent_files(self):
        # Catches a parent edit after this design was locked (or a wrong pin).
        assert addendum1.sha256_file(rn.DESIGN) == P["parent"]["design_sha256"]
        assert addendum1.sha256_file(Path(rn.__file__)) == P["parent"]["script_sha256"]
        assert replay.prereg_hash(rn.PREREG_REVIEW) == P["parent"]["prereg_sha256"]

    def test_this_designs_own_hash_is_its_files_hash(self):
        assert addendum1.sha256_file(cf.DESIGN) == P["design_sha256"]

    def test_budgets_are_the_ones_label_py_enforces(self):
        # Catches the two caps drifting apart: the PREREG would claim a cap label.py ignores.
        assert P["budget_usd"]["total"] == label.CONFIRM_TOTAL_BUDGET
        assert P["budget_usd"]["per_run"] == label.CONFIRM_RUN_BUDGET
        assert P["budget_usd"]["smoke"] == label.CONFIRM_SMOKE_BUDGET
        assert P["model_id"] == label.OPENROUTER_MODEL
        assert P["route"] == "openrouter"

    def test_the_validity_and_verdict_constants_are_the_designs(self):
        # Catches a quiet edit of a threshold in PREREG_CONFIRM: sections 5 and 6
        # of DESIGN-confirm.md fix every one of these numbers.
        assert P["v"] == {
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
        }
        assert P["pass"] == {
            "lift_mean_min": 2.0,
            "lift_lb_gt": 1.0,
            "gain_mean_min_pts": 10.0,
            "gain_lb_gt_pts": 0.0,
        }
        assert P["kill"] == {"gain_mean_lt_pts": 5.0, "lift_mean_lt": 1.5}
        assert P["flags"] == {
            "k_fragile_gain_le_pts": 0.0,
            "subset_gain_min_pts": 10.0,
            "domain_neg_min": 3,
        }
        assert (P["splits"], P["bootstrap"], P["bootstrap_splits_per_draw"]) == (
            1000,
            1000,
            20,
        )
        assert (P["shrinkage_m"], P["alpha"], P["n_tests"]) == (4, 0.05, 2)
        assert P["budget_usd"] == {
            "total": 1.00,
            "per_run": {"intent": 0.14, "faithfulness": 0.09},
            "smoke": 0.01,
        }

    def test_the_build_recipe_names_the_builders_generators(self):
        build_path = _ROOT / "benchmarks" / "golden" / "build.py"
        spec = importlib.util.spec_from_file_location(
            "golden_build_for_pins", build_path
        )
        build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build)

        assert P["build"]["rng"] == build.CONFIRM_RNG
        assert P["build"]["seed"] == build.SEED
        assert P["build"]["intent_per_class"] == build.CONFIRM_INTENT_PER_CLASS
        assert (
            P["build"]["faithfulness_per_domain_label"]
            == build.CONFIRM_FAITHFULNESS_PER_DOMAIN_LABEL
        )
        assert (
            P["build"]["faithfulness_cap_per_doc_label"]
            == build.CONFIRM_FAITHFULNESS_CAP_PER_DOC_LABEL
        )
        assert P["build"]["rows"] == {
            "intent": 77 * build.CONFIRM_INTENT_PER_CLASS,
            "faithfulness": 8 * 2 * build.CONFIRM_FAITHFULNESS_PER_DOMAIN_LABEL,
        }

    def test_the_design_numbers_follow_from_the_row_counts(self):
        # 770 - 100 = 670 and 320 - 100 = 220 test rows; k = 34/67/134 and 11/22/44.
        for task, rows in P["build"]["rows"].items():
            test_rows = rows - P["n_labelled"]
            assert test_rows == P["expected"]["test_rows"][task]
            assert [rn.k_at(f, test_rows) for f in FRACS] == P["expected"]["k"][task]

    def test_the_seed_is_this_designs_and_the_parents_task_order_is_kept(self):
        eff = cf.confirm_effective()

        assert eff["seed"] == 20261118 != rn.effective_prereg(P)["seed"]
        assert eff["tasks"].index("intent") == 0
        assert eff["tasks"].index("faithfulness") == 1
        assert (eff["splits"], eff["n_labelled"], eff["n_confirmatory"]) == (
            1000,
            100,
            2,
        )

    def test_the_confirmatory_family_is_r2_only_and_the_exploratory_set_is_six(self):
        assert P["confirmatory"] == ["R2"]
        assert list(P["exploratory"]) == ["E2", "E4", "E5", "E6", "E7", "E8"]
        assert P["cluster"] == {"intent": "row", "faithfulness": "doc"}
        assert P["sample_files"] == ["haiku-s1", "haiku-s2", "haiku-s3", "haiku-s4"]

    def test_the_splits_are_drawn_from_the_confirm_seed(self):
        # Catches the parent's seed leaking into the splits.
        task, _, _ = planted("errors")
        eff = cf.confirm_effective(small())

        train_w, _, _ = replay._SplitCache(task, eff).get(100)
        want = replay.draw_splits(
            replay.split_rng(
                {**replay.PREREG, "seed": 20261118}, "intent", "original", 0, 100
            ),
            task.n,
            100,
            60,
        )

        np.testing.assert_array_equal(train_w, want.astype(float))


# ── --check ──────────────────────────────────────────────────────────────


def text(rep) -> str:
    return "\n".join(rep.lines)


def drop_confirm_data(fx: Fx) -> None:
    """The state before anything is built: no confirm rows, no samples, no lock."""
    for t in cf.TASKS:
        (fx.data / t / "confirm.jsonl").unlink()
        for p in (fx.data / "labels" / t / "confirm").iterdir():
            p.unlink()
    fx.lock.unlink(missing_ok=True)


class TestCheck:
    def test_a_complete_locked_fixture_passes_and_prints_every_part(self, full, capsys):
        code = cf.main(["--check"], **full.kwargs)

        out = capsys.readouterr().out
        assert code == 0
        assert "CHECK OK" in out
        for part in (
            "design, parent design, parent PREREG and parent script hashes: OK",
            "[intent] confirm rows: 30 (3 classes x 10)",
            "[faithfulness] confirm rows: 16 (2 domains x 4 per label; max 2 rows per document and label)",
            "[intent] V1-V3: ok",
            "[faithfulness] V1-V3: ok",
            "lock file: ",
            "[intent] test rows per split [22.0]",
            "[faithfulness] test rows per split [8.0]",
            "R0 lift 1 to within",
            "guard: the release split and unlisted files are refused: OK",
            "[intent] confirm_task on the dev Haiku file equals replay.load_task: OK",
            "confirm spend recorded: $0.0800 of the $1.00 cap",
        ):
            assert part in out, part

    def test_before_the_data_exists_it_exits_3_lists_the_gaps_and_checks_the_rest(
        self, fx, capsys
    ):
        # Catches an "ok" with nothing there (exit 0), and a check that stops at the
        # first gap instead of checking what it can.
        drop_confirm_data(fx)
        pins = {t: "<filled by the lead>" for t in cf.TASKS}

        code = cf.main(
            ["--check"],
            **{**fx.kwargs, "prereg": {**fx.prereg, "confirm_split_sha256": pins}},
        )

        out = capsys.readouterr().out
        assert code == 3
        assert "CHECK INCOMPLETE" in out
        for gap in (
            "intent/confirm.jsonl",
            "faithfulness/confirm.jsonl",
            "PREREG confirm_split_sha256[intent]",
            "PREREG confirm_split_sha256[faithfulness]",
            "confirm-data.lock.json",
        ):
            assert f"NOT YET AVAILABLE: {gap}" in out or any(
                gap in line and line.startswith("NOT YET") for line in out.splitlines()
            ), gap
        assert "R0 lift 1 to within" in out
        assert "placeholder rows" in out
        assert "guard: " in out
        assert "equals replay.load_task: OK" in out
        assert "FAIL" not in out

    def test_a_partial_set_of_samples_is_incomplete_and_names_the_missing_files(
        self, fx
    ):
        for stem in ("haiku-s3", "haiku-s4"):
            fx.sample("intent", stem).unlink()
            fx.sample("intent", stem).with_suffix(".cost.json").unlink()

        rep = fx.check()

        assert rep.exit_code == 3
        assert any("intent: haiku-s3, haiku-s4" in m for m in rep.missing)
        assert not rep.problems

    def test_a_replacement_file_beside_a_missing_one_is_a_problem(self, fx):
        fx.sample("intent", "haiku-s3").unlink()
        write_sample(fx.data, "intent", "haiku-s5")

        rep = fx.check()

        assert any(
            "haiku-s5 exists while haiku-s3 is missing" in p for p in rep.problems
        )

    def test_it_never_reports_ok_while_anything_is_missing(self, fx):
        # Catches exit 0 on an unlocked fixture: the lock is part of "all there".
        rep = fx.check()

        assert rep.exit_code == 3
        assert [m for m in rep.missing] == [
            "confirm-data.lock.json (confirm.py --lock, committed before the run)"
        ]

    def test_check_mode_never_runs_a_ranker_or_the_bootstrap(self, full, monkeypatch):
        # evaluate, ranker_keys, aggregate, the bootstrap and the study all raise.
        def boom(*_a, **_k):
            raise AssertionError("check mode computed a result")

        for name in ("evaluate", "ranker_keys", "aggregate"):
            monkeypatch.setattr(rn, name, boom)
        for name in ("bootstrap_confirm", "run_study", "run_task", "point_estimates"):
            monkeypatch.setattr(cf, name, boom)

        assert full.check().exit_code == 0

    def test_the_guard_selftest_failing_is_a_hard_problem(self, full, monkeypatch):
        monkeypatch.setattr(cf, "guard_selftest", lambda _d: ["intent/leak"])

        assert any("the guard let" in p for p in full.check().problems)

    def test_confirm_task_drifting_from_load_task_is_a_hard_problem(
        self, full, monkeypatch
    ):
        real = replay.load_task

        def flipped(name, gold="original", data_dir=None):
            t = real(name, gold, data_dir)
            t.correct["haiku"] = ~t.correct["haiku"]
            return t

        monkeypatch.setattr(replay, "load_task", flipped)

        assert any(
            "does not reproduce replay.load_task" in p for p in full.check().problems
        )

    def test_the_test_rows_and_k_must_be_the_designs(self, fx):
        bad = {
            **fx.prereg,
            "expected": {
                **fx.prereg["expected"],
                "k": {"intent": [1, 2, 3], "faithfulness": [1, 1, 1]},
            },
        }

        rep = cf.check(**{**fx.kwargs, "prereg": bad})

        assert any("k values" in p for p in rep.problems)


class TestCheckHashes:
    def test_this_designs_file_hash_must_match(self, fx, tmp_path):
        # Catches an edit to the locked design after the PREREG pinned it.
        altered = tmp_path / "DESIGN-confirm.md"
        altered.write_bytes(cf.DESIGN.read_bytes() + b" ")

        rep = fx.check(design_path=altered)

        assert any(p.startswith("this design:") for p in rep.problems)

    @pytest.mark.parametrize(
        ("pin", "name"),
        [
            ("design_sha256", "the parent's design"),
            ("prereg_sha256", "the parent's PREREG"),
            ("script_sha256", "the parent's script"),
        ],
    )
    def test_each_parent_pin_must_match_its_file(self, fx, pin, name):
        # Catches an edit to review_next.py (or its design or PREREG) after this
        # design pinned it: the parent's recorded hashes would no longer hold.
        bad = {**fx.prereg, "parent": {**fx.prereg["parent"], pin: "0" * 64}}

        rep = cf.check(**{**fx.kwargs, "prereg": bad})

        assert any(p.startswith(name) for p in rep.problems), rep.problems
        assert rep.exit_code == 2

    def test_a_missing_design_file_is_a_problem_not_a_crash(self, fx, tmp_path):
        rep = fx.check(design_path=tmp_path / "nope.md")

        assert any(p.startswith("this design:") for p in rep.problems)


def edit_rows(fx: Fx, task: str, edit) -> None:
    path = fx.data / task / "confirm.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    edit(rows)
    write_jsonl(path, rows)
    fx.refreeze()


class TestCheckRows:
    def test_the_pinned_hash_must_match_the_file(self, fx):
        # Catches a confirm.jsonl that changed after its hash was pinned.
        path = fx.data / "intent" / "confirm.jsonl"
        path.write_text(path.read_text() + "\n")

        rep = fx.check()

        assert any("intent/confirm.jsonl: sha256" in p for p in rep.problems)

    def test_an_unfilled_pin_with_the_file_present_is_a_problem(self, fx):
        bad = {
            **fx.prereg,
            "confirm_split_sha256": {"intent": "<x>", "faithfulness": "<x>"},
        }

        rep = cf.check(**{**fx.kwargs, "prereg": bad})

        assert sum("is not the PREREG's" in p for p in rep.problems) == 2

    def test_a_full_sha256_and_its_16_hex_prefix_both_match(self, fx):
        full_pin = {t: sha(fx.data / t / "confirm.jsonl") for t in cf.TASKS}
        short = {t: v[:16] for t, v in full_pin.items()}

        for pins in (full_pin, short):
            rep = cf.check(
                **{**fx.kwargs, "prereg": {**fx.prereg, "confirm_split_sha256": pins}}
            )
            assert not any("is not the PREREG's" in p for p in rep.problems)

    def test_the_row_count_is_the_designs(self, fx):
        edit_rows(fx, "intent", lambda rows: rows.pop())

        assert any("29 rows, expected 30" in p for p in fx.check().problems)

    def test_every_class_has_exactly_its_quota(self, fx):
        def edit(rows):
            rows[0]["label"] = rows[1]["label"]  # one class 11, another 9

        edit_rows(fx, "intent", edit)

        assert any("classes do not all have 10 rows" in p for p in fx.check().problems)

    def test_a_confirm_id_that_is_a_dev_id_is_a_problem(self, fx):
        # Catches overlap with dev: the rows must be new.
        edit_rows(fx, "intent", lambda rows: rows[0].update(id="i0"))

        assert any("a confirm id is a dev id" in p for p in fx.check().problems)

    def test_duplicate_ids_are_a_problem(self, fx):
        edit_rows(fx, "faithfulness", lambda rows: rows[1].update(id=rows[0]["id"]))

        assert any("duplicate ids" in p for p in fx.check().problems)

    def test_a_domain_label_quota_miss_is_a_problem(self, fx):
        edit_rows(fx, "faithfulness", lambda rows: rows[0].update(label="inconsistent"))

        assert any(
            "domain/label pairs do not all have 4 rows" in p
            for p in fx.check().problems
        )

    def test_a_document_above_the_cap_for_one_label_is_a_problem(self, fx):
        # Catches a missing cap: 3 rows of one label from one document (cap 2).
        def edit(rows):
            for r in rows:
                if r["domain"] == "d1" and r["label"] == "consistent":
                    r["doc"] = "d1-docA"

        edit_rows(fx, "faithfulness", edit)

        assert any(
            "more rows of one label than the cap" in p for p in fx.check().problems
        )

    def test_the_intact_rows_report_counts_and_disjointness(self, fx):
        assert any("disjoint from dev" in line for line in fx.check().lines)


class TestCheckSamples:
    def test_one_file_failing_only_v4_is_replaced_by_s5_in_its_column(self, fx):
        # A row missing from s3 is a V4 failure only (V1 and V2 hold).
        write_jsonl(fx.sample("intent", "haiku-s3"), answers("intent", "haiku-s3")[:-1])
        write_sample(fx.data, "intent", "haiku-s5")

        rep = fx.check()

        assert not rep.problems, rep.problems
        assert rep.selected["intent"] == [
            "haiku-s1",
            "haiku-s2",
            "haiku-s5",
            "haiku-s4",
        ]
        assert "haiku-s3 fails only V4" in text(rep)
        assert rep.files["intent"]["haiku-s5"]["used"]
        assert not rep.files["intent"]["haiku-s3"]["used"]

    def test_a_v4_failure_without_s5_asks_for_it(self, fx):
        write_jsonl(fx.sample("intent", "haiku-s3"), answers("intent", "haiku-s3")[:-1])

        rep = fx.check()

        assert any(
            "the design allows one replacement: draw haiku-s5" in p
            for p in rep.problems
        )

    def test_s5_when_no_file_fails_is_refused(self, fx):
        # Catches a free redraw: a replacement is only for a V4 failure.
        write_sample(fx.data, "intent", "haiku-s5")

        assert any("nothing may be replaced" in p for p in fx.check().problems)

    def test_two_v4_failures_cannot_be_replaced(self, fx):
        for stem in ("haiku-s2", "haiku-s3"):
            write_jsonl(fx.sample("intent", stem), answers("intent", stem)[:-1])
        write_sample(fx.data, "intent", "haiku-s5")

        assert any("cannot be replaced" in p for p in fx.check().problems)

    def test_one_invalid_answer_in_thirty_rows_is_over_3_percent(self, fx):
        # 1 / 30 = 3.3% > 3%: a V4 failure (the bad-row rule is exact).
        rows = answers("intent", "haiku-s2")
        rows[7] = {"id": rows[7]["id"], "answer": "not an intent at all"}
        write_jsonl(fx.sample("intent", "haiku-s2"), rows)

        assert any(
            "fails V4" in p and "null or invalid" in p for p in fx.check().problems
        )

    @pytest.mark.parametrize(
        "over",
        [{"route": "direct"}, {"model": "anthropic/claude-haiku-5-5"}, {"rows": 29}],
    )
    def test_route_model_and_rows_are_fixed(self, fx, over):
        # Catches a sample drawn on another route or model, or on other rows.
        write_sample(fx.data, "intent", "haiku-s2", **over)

        assert any(
            "fails V4" in p and "route/model/rows" in p for p in fx.check().problems
        )

    def test_two_identical_files_fail_v4(self, fx):
        # Catches a copied file posing as a fresh draw.
        write_jsonl(fx.sample("intent", "haiku-s3"), answers("intent", "haiku-s2"))

        assert any(
            "identical on every row to ['haiku-s2']" in p for p in fx.check().problems
        )

    def test_an_unexpected_sample_file_is_a_problem_but_a_smoke_file_is_not(self, fx):
        write_sample(
            fx.data, "intent", "haiku-s1-smoke3", answers("intent", "haiku-s1")
        )
        assert not fx.check().problems

        write_sample(fx.data, "intent", "haiku-s6", answers("intent", "haiku-s1"))
        assert any(
            "unexpected sample files ['haiku-s6']" in p for p in fx.check().problems
        )

    def test_v1_v2_and_v3_failures_are_reported_per_task_and_are_not_an_exit_2(
        self, fx
    ):
        # As in the parent: the study may still run, and that task is INCONCLUSIVE.
        tight = {
            **fx.prereg["v"],
            "v1": {
                "intent": {"metric": "accuracy", "center": 0.5, "tol": 0.01},
                "faithfulness": fx.prereg["v"]["v1"]["faithfulness"],
            },
            "v2": {
                **fx.prereg["v"]["v2"],
                "implied": {"intent": 1000.0, "faithfulness": 100.0},
            },
            "v3_tol_pts": 0.0,
        }

        rep = cf.check(**{**fx.kwargs, "prereg": {**fx.prereg, "v": tight}})

        assert not rep.problems
        assert {"intent", "faithfulness"} == set(rep.v123)
        assert any("V1" in x for x in rep.v123["intent"])
        assert any("V2" in x for x in rep.v123["intent"])
        assert any("V3" in x for x in rep.v123["faithfulness"])
        assert "[intent] V1-V3: FAILED" in text(rep)

    def test_v3_is_the_difference_between_s1_and_the_others_and_the_rest(self, fx):
        rep = fx.check()

        v3 = rep.v3["intent"]
        assert v3["diff_pts"] == pytest.approx(
            100 * (v3["among_new"] - v3["with_original"])
        )
        assert "V3: agreement of s1 with the others minus among the rest" in text(rep)

    def test_the_file_lines_carry_v1_v2_v4_and_the_cost(self, fx):
        out = text(fx.check())

        assert re.search(
            r"haiku-s1: V1 0\.\d+ in \[.*\] ok; V2 100\.0 tokens/row, x1\.00", out
        )
        assert "provenance ok" in out
        assert "cost $0.0100" in out


class TestCheckSpend:
    def test_smoke_and_aborted_runs_count_toward_the_cap(self, fx):
        # Catches an undercount: the cap is on all new spend.
        for task, name, cost in (
            ("intent", "haiku-s1-smoke3", 0.46),
            ("faithfulness", "haiku-s2-aborted", 0.46),
        ):
            (fx.data / "labels" / task / "confirm" / f"{name}.cost.json").write_text(
                json.dumps({"cost": cost})
            )

        rep = fx.check()

        assert rep.spend == pytest.approx(
            1.00
        )  # 0.08 + 0.92: equal to the cap is allowed
        assert not any("spend" in p for p in rep.problems)

        (
            fx.data / "labels" / "intent" / "confirm" / "haiku-s1-smoke3.cost.json"
        ).write_text(json.dumps({"cost": 0.4601}))
        assert any("over the $1.00 cap" in p for p in fx.check().problems)

    def test_dev_cost_files_do_not_count(self, fx):
        (fx.data / "labels" / "intent" / "haiku-s1.cost.json").write_text(
            json.dumps({"cost": 5.0})
        )

        assert fx.check().spend == pytest.approx(0.08)


# ── the lock ─────────────────────────────────────────────────────────────


class TestLock:
    def test_it_records_the_prereg_and_the_hash_of_every_data_file(self, full):
        lock = json.loads(full.lock.read_text())

        assert lock["prereg_sha256"] == replay.prereg_hash(full.prereg)
        files = lock["data_sha256"]
        assert "intent/confirm.jsonl" in files
        assert "labels/faithfulness/confirm/haiku-s4.cost.json" in files
        assert files["intent/confirm.jsonl"] == sha(
            full.data / "intent" / "confirm.jsonl"
        )
        assert len(files) == 1 + 2 * (
            2 + 2 * 4
        )  # labels.json; per task confirm, dev, 4 x (jsonl + cost)
        assert not [k for k in files if "hold" + "out" in k.lower()]

    def test_a_written_lock_passes_the_check(self, full):
        rep = full.check()

        assert rep.exit_code == 0
        assert "lock file: 21 data files match their sha256" in text(rep)
        assert len(rep.data_sha256) == 21

    def test_it_is_written_once(self, full):
        with pytest.raises(SystemExit, match="written once"):
            cf.lock(**full.kwargs)

    def test_it_refuses_while_data_is_missing_or_a_check_fails(self, fx, tmp_path):
        fx.sample("intent", "haiku-s4").unlink()
        with pytest.raises(SystemExit, match="not locking"):
            cf.lock(**fx.kwargs)
        assert not fx.lock.exists()

    def test_it_refuses_on_a_hard_problem(self, fx):
        path = fx.data / "intent" / "confirm.jsonl"
        path.write_text(path.read_text() + "\n")  # no longer the pinned bytes

        with pytest.raises(SystemExit, match="not locking"):
            cf.lock(**fx.kwargs)

    def test_an_edit_to_a_sample_after_the_lock_fails_the_check(self, full):
        # Catches a quiet change of an answer file after it was locked.
        rows = answers("intent", "haiku-s2")
        rows[9] = {"id": rows[9]["id"], "answer": "top_up"}
        write_jsonl(full.sample("intent", "haiku-s2"), rows)

        rep = full.check()

        assert rep.exit_code == 2
        assert any(
            "labels/intent/confirm/haiku-s2.jsonl: sha256" in p
            and "differs from the lock" in p
            for p in rep.problems
        )

    def test_a_data_file_added_after_the_lock_fails_the_check(self, full):
        # Catches a late s5 draw that the lock does not list.
        write_sample(full.data, "intent", "haiku-s5")

        rep = full.check()

        assert any("a data file that the lock does not list" in p for p in rep.problems)

    def test_a_changed_prereg_after_the_lock_fails_the_check(self, full):
        rep = cf.check(
            **{**full.kwargs, "prereg": {**full.prereg, "primary_k": "0.05"}}
        )

        assert any("different PREREG_CONFIRM" in p for p in rep.problems)


# ── run guards and outputs ───────────────────────────────────────────────


@pytest.fixture
def guarded(full, tmp_path, monkeypatch):
    """The locked fixture, a scratch results root and a clean tree."""
    monkeypatch.setattr(cf, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(cf, "_git_state", lambda: CLEAN)
    return full.kwargs


class TestRunGuards:
    def test_quick_is_refused_before_anything_runs_unless_out_or_a_result_exists(
        self, guarded, tmp_path, monkeypatch, capsys
    ):
        def boom(*_a, **_k):
            raise AssertionError("a refused run computed something")

        with monkeypatch.context() as m:
            m.setattr(cf, "check", boom)
            m.setattr(cf, "run_study", boom)
            code = cf.main(["--quick"], **{**guarded, "data_dir": cf.DATA})
        out = capsys.readouterr().out

        assert code == 2
        assert "REFUSED" in out
        assert "reveals the outcome" in out
        assert not (tmp_path / "results").exists()

        scratch = tmp_path / "scratch"
        assert cf.main(["--quick", "--out", str(scratch)], **guarded) == 0
        assert (scratch / "confirm.json").exists()

        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "confirm.json").write_text("{}")
        assert cf.main(["--quick"], **guarded) == 0
        assert (tmp_path / "results" / "quick" / "confirm.json").exists()

    def test_check_is_never_refused(self, guarded, monkeypatch):
        monkeypatch.setattr(cf, "_git_state", lambda: {"head": "x", "dirty": True})

        assert cf.main(["--check"], **guarded) == 0

    def test_a_full_run_on_a_dirty_tree_exits_2_and_a_clean_one_writes_the_result(
        self, guarded, tmp_path, monkeypatch, capsys
    ):
        out_dir = tmp_path / "full"
        monkeypatch.setattr(cf, "_git_state", lambda: {"head": "x", "dirty": True})

        dirty = cf.main(["--out", str(out_dir)], **guarded)

        assert dirty == 2
        assert "uncommitted changes" in capsys.readouterr().out
        assert not out_dir.exists()

        monkeypatch.setattr(cf, "_git_state", lambda: CLEAN)
        assert cf.main(["--out", str(out_dir)], **guarded) == 0
        meta = json.loads((out_dir / "confirm.json").read_text())["meta"]
        assert meta["git"] == CLEAN
        assert meta["quick"] is False
        assert meta["bootstrap"] == 8  # the fixture's B; the real one is 1000

    def test_a_second_full_run_may_not_overwrite_the_result(
        self, guarded, tmp_path, capsys
    ):
        out_dir = tmp_path / "full"
        assert cf.main(["--out", str(out_dir)], **guarded) == 0
        first = (out_dir / "confirm.json").read_text()

        code = cf.main(["--out", str(out_dir)], **guarded)

        assert code == 2
        assert "would overwrite" in capsys.readouterr().out
        assert (out_dir / "confirm.json").read_text() == first
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "confirm.json").write_text("{}")
        assert cf.main([], **guarded) == 2

    def test_a_missing_lock_stops_the_run_with_the_incomplete_code(
        self, guarded, tmp_path, capsys
    ):
        # Catches a full run without a committed lock: the data hashes would be unproven.
        guarded["lock_path"].unlink()

        code = cf.main(["--out", str(tmp_path / "full")], **guarded)

        assert code == 3
        assert "CHECK INCOMPLETE" in capsys.readouterr().out
        assert not (tmp_path / "full").exists()

    def test_a_lock_mismatch_stops_the_run_with_exit_2(self, guarded, tmp_path, capsys):
        path = (
            guarded["data_dir"] / "labels" / "intent" / "confirm" / "haiku-s1.cost.json"
        )
        path.write_text(path.read_text().replace('"cost": 0.01', '"cost": 0.02'))

        code = cf.main(["--out", str(tmp_path / "full")], **guarded)

        assert code == 2
        assert "differs from the lock" in capsys.readouterr().out
        assert not (tmp_path / "full").exists()

    def test_the_dirty_flag_sees_the_studys_own_files_but_not_its_results(
        self, tmp_path, monkeypatch
    ):
        def git(*args):
            subprocess.run(
                ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
                cwd=tmp_path,
                check=True,
                capture_output=True,
            )

        tracked = (
            "benchmarks/review_next/confirm.py",
            "benchmarks/review_next/results/x.json",
            "benchmarks/planner/replay.py",
            "benchmarks/golden/build.py",
            "tests/unit/test_any.py",
            "docs/other.md",
        )
        for rel in tracked:
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / rel).write_text("a\n")
        git("init", "-q")
        git("add", "-A")
        git("commit", "-q", "-m", "init")
        monkeypatch.setattr(cf, "REPO", tmp_path)

        assert cf._git_state()["dirty"] is False
        (tmp_path / "benchmarks/review_next/results/x.json").write_text("changed\n")
        (tmp_path / "docs/other.md").write_text("changed\n")
        assert cf._git_state()["dirty"] is False  # results and unrelated files
        for rel in tracked[:1] + tracked[2:5]:
            (tmp_path / rel).write_text("changed\n")
            assert cf._git_state()["dirty"] is True, rel
            git("checkout", "--", rel)
            assert cf._git_state()["dirty"] is False
        (tmp_path / "benchmarks/review_next/untracked.py").write_text("x\n")
        assert cf._git_state()["dirty"] is True  # an untracked file in the study dir


class TestOutputs:
    def run(self, guarded, tmp_path):
        out_dir = tmp_path / "full"
        assert cf.main(["--out", str(out_dir)], **guarded) == 0
        return json.loads((out_dir / "confirm.json").read_text()), out_dir

    def test_the_results_embed_the_hashes_the_run_depended_on(self, guarded, tmp_path):
        results, _ = self.run(guarded, tmp_path)

        m = results["meta"]
        assert (
            m["design_sha256"] == addendum1.sha256_file(cf.DESIGN) == P["design_sha256"]
        )
        assert m["prereg_confirm_sha256"] == replay.prereg_hash(guarded["prereg"])
        assert m["parent_design_sha256"] == P["parent"]["design_sha256"]
        assert m["parent_prereg_sha256"] == P["parent"]["prereg_sha256"]
        assert m["parent_script_sha256"] == P["parent"]["script_sha256"]
        assert m["code_sha256"] == addendum1.sha256_file(_MODULE)
        assert m["seed"] == 20261118
        assert len(m["data_sha256"]) == 21
        assert results["checks"]["spend"] == pytest.approx(0.08)

    def test_the_verdict_follows_the_two_statuses(self, guarded, tmp_path):
        results, _ = self.run(guarded, tmp_path)

        v = results["verdict"]
        assert v["statuses"] == {t: results["tasks"][t]["status"] for t in cf.TASKS}
        assert v["verdict"] == cf.verdict_confirm(v["statuses"])["verdict"]
        assert set(results["tasks"]) == {"intent", "faithfulness"}

    def test_the_markdown_reports_the_verdict_both_tasks_the_exploratory_and_domains(
        self, guarded, tmp_path
    ):
        _, out_dir = self.run(guarded, tmp_path)

        md = (out_dir / "confirm.md").read_text()
        for heading in ("## Verdict", "## intent", "## faithfulness"):
            assert heading in md
        assert "Exploratory (never decisive)" in md
        assert "| E8 |" in md
        assert "| R2 |" in md
        assert "G of R2 per domain" in md
        assert "over 6 doc clusters" in md  # faithfulness: 2 domains x 3 documents
        assert "over 30 row clusters" in md  # intent

    def test_the_faithfulness_bootstrap_is_over_documents(self, guarded, tmp_path):
        results, _ = self.run(guarded, tmp_path)

        b = results["tasks"]["faithfulness"]["bootstrap"]
        assert (b["cluster"], b["clusters"]) == ("doc", 6)  # 2 domains x 3 documents
        assert results["tasks"]["intent"]["bootstrap"]["cluster"] == "row"

    def test_the_cost_of_the_extra_samples_per_1k_rows_is_reported(
        self, guarded, tmp_path
    ):
        results, _ = self.run(guarded, tmp_path)

        # three extra files at $0.01 per run of 30 rows: 3 * 0.01 / 30 * 1000
        assert results["cost_per_1k_rows_extra_samples"]["intent"] == pytest.approx(1.0)

    def test_the_run_uses_the_selected_files_including_a_replacement(
        self, tmp_path, monkeypatch
    ):
        fx = Fx(tmp_path)
        write_jsonl(fx.sample("intent", "haiku-s3"), answers("intent", "haiku-s3")[:-1])
        write_sample(fx.data, "intent", "haiku-s5")
        cf.lock(**fx.kwargs)
        monkeypatch.setattr(cf, "_git_state", lambda: CLEAN)
        monkeypatch.setattr(cf, "RESULTS", tmp_path / "results")

        assert cf.main(["--out", str(tmp_path / "o")], **fx.kwargs) == 0

        results = json.loads((tmp_path / "o" / "confirm.json").read_text())
        assert results["checks"]["selected"]["intent"] == [
            "haiku-s1",
            "haiku-s2",
            "haiku-s5",
            "haiku-s4",
        ]


# ── review round 2 ───────────────────────────────────────────────────────


class TestQuickOnRealData:
    def test_quick_is_refused_on_the_real_data_dir_even_with_out(
        self, tmp_path, monkeypatch, capsys
    ):
        # Catches --quick --out printing the verdict before the full run: the README
        # says it is refused until results/confirm.json exists.
        def boom(*_a, **_k):
            raise AssertionError("a refused run computed something")

        monkeypatch.setattr(cf, "RESULTS", tmp_path / "results")
        monkeypatch.setattr(cf, "check", boom)
        monkeypatch.setattr(cf, "run_study", boom)

        code = cf.main(["--quick", "--out", str(tmp_path / "o")], data_dir=cf.DATA)

        assert code == 2
        assert "REFUSED" in capsys.readouterr().out
        assert not (tmp_path / "o").exists()

    def test_the_real_data_dir_allows_quick_once_the_full_result_exists(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(cf, "RESULTS", tmp_path / "results")
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "confirm.json").write_text("{}")

        class Args:
            quick, out = True, tmp_path / "o"

        assert cf._refusal(Args, cf.DATA) is None

    def test_a_fixture_data_dir_may_still_run_quick_with_out(self, guarded, tmp_path):
        assert cf.main(["--quick", "--out", str(tmp_path / "o")], **guarded) == 0


class TestPinFailureIsNotFollowedByOk:
    def test_a_placeholder_pin_prints_fail_and_no_matches_line(self, fx):
        # Catches "FAIL: ... is not the PREREG's" followed by "... matches the PREREG".
        bad = {
            **fx.prereg,
            "confirm_split_sha256": {t: "<placeholder>" for t in cf.TASKS},
        }

        rep = cf.check(**{**fx.kwargs, "prereg": bad})

        assert sum("is not the PREREG's" in p for p in rep.problems) == 2
        assert "matches the PREREG" not in text(rep)


class TestVFailureEndToEnd:
    def test_a_v1_failure_makes_that_task_inconclusive_in_the_verdict(
        self, tmp_path, monkeypatch
    ):
        # Catches v_failed lost between check and run_study (False passed instead):
        # numbers forced to pass must still end INCONCLUSIVE on the failed task.
        fx = Fx(tmp_path)
        tight_v1 = {
            **fx.prereg["v"]["v1"],
            "intent": {"metric": "accuracy", "center": 0.5, "tol": 0.01},
        }
        fx.prereg = {**fx.prereg, "v": {**fx.prereg["v"], "v1": tight_v1}}
        cf.lock(**fx.kwargs)
        monkeypatch.setattr(cf, "_git_state", lambda: CLEAN)
        monkeypatch.setattr(cf, "RESULTS", tmp_path / "results")
        real = rn.task_status
        seen: dict[str, bool] = {}

        def passing_numbers(entry, prereg):
            seen[str(len(seen))] = entry["v_failed"]
            return real(
                {**entry, "L": 5.0, "L_lb": 3.0, "G": 50.0, "G_lb": 20.0}, prereg
            )

        monkeypatch.setattr(rn, "task_status", passing_numbers)

        assert cf.main(["--out", str(tmp_path / "o")], **fx.kwargs) == 0

        results = json.loads((tmp_path / "o" / "confirm.json").read_text())
        assert results["verdict"]["statuses"] == {
            "intent": "INCONCLUSIVE",
            "faithfulness": "PASS",
        }
        assert results["verdict"]["v_failed"] == ["intent"]
        assert results["verdict"]["verdict"] == "CONFIRMED(faithfulness)"
        assert sorted(seen.values()) == [False, True]


class TestNullsNeverAgree:
    def write_null(self, fx, stems, row=4):
        for stem in stems:
            rows = answers("intent", stem)
            rows[row] = {"id": rows[row]["id"], "answer": None}
            write_jsonl(fx.sample("intent", stem), rows)

    def test_two_files_null_on_one_row_count_as_dissent_and_do_not_agree(self, fx):
        # Catches a shared "\x00null" marker: two nulls on one row would agree in D
        # (when the delivered answer is null too), in V3 and in identical-columns.
        self.write_null(fx, ["haiku-s1", "haiku-s2", "haiku-s3"])
        samples = cf.load_confirm_samples("intent", data_dir=fx.data)

        assert len({samples.codes[4, 0], samples.codes[4, 1], samples.codes[4, 2]}) == 3
        assert (
            rn.dissent(samples)[4] == 3
        )  # s2, s3 and the intact s4 all differ from a null s1
        assert not samples.valid[4, :3].any()

    def test_v3_and_the_identical_columns_check_do_not_see_null_pairs_as_agreeing(
        self, fx
    ):
        # s2 and s3 differ from each other only where they are both null on row 4,
        # which must not make them agree: they stay non-identical.
        self.write_null(fx, ["haiku-s2", "haiku-s3"])
        write_jsonl(fx.sample("intent", "haiku-s3"), answers("intent", "haiku-s2"))
        self.write_null(fx, ["haiku-s3"])
        samples = cf.load_confirm_samples("intent", data_dir=fx.data)

        assert not (samples.codes[:, 1] == samples.codes[:, 2]).all()
        assert (1, 2) not in addendum1.identical_columns(samples.codes)
        agree_12 = (samples.codes[:, 1] == samples.codes[:, 2]).mean()
        assert agree_12 < 1.0

    def test_a_copied_file_with_nulls_is_still_caught_at_the_file_level(self, fx):
        # The by-row marker stays in the per-file V4 copy check.
        self.write_null(fx, ["haiku-s2"])
        write_jsonl(
            fx.sample("intent", "haiku-s3"),
            [
                json.loads(x)
                for x in fx.sample("intent", "haiku-s2").read_text().splitlines()
            ],
        )

        rep = fx.check()

        assert any("identical on every row to ['haiku-s2']" in p for p in rep.problems)


class TestHoldFiles:
    def hold(self, fx, task="intent"):
        folder = fx.data / "labels" / task / "confirm"
        path = folder / "haiku-s1-hold-123.cost.json"
        path.write_text(json.dumps({"cost": 0.14, "reserved": True, "rows": 0}))
        return path

    def test_a_hold_is_charged_listed_and_never_selected_or_locked(self, fx):
        self.hold(fx)

        rep = fx.check()

        assert rep.spend == pytest.approx(0.08 + 0.14)
        assert rep.holds == ["labels/intent/confirm/haiku-s1-hold-123.cost.json"]
        assert "unreleased hold files" in text(rep)
        assert not rep.problems, rep.problems
        assert rep.selected["intent"] == STEMS
        assert not [
            k for k in cf.lock_content(fx.data, fx.prereg)["data_sha256"] if "hold" in k
        ]

    def test_lock_refuses_while_any_hold_file_exists(self, fx):
        # Catches a lock written over a crashed run's unaccounted spend.
        self.hold(fx)

        with pytest.raises(SystemExit, match="unreleased hold files"):
            cf.lock(**fx.kwargs)

        assert not fx.lock.exists()

    def test_a_hold_appearing_after_the_lock_is_a_hard_problem(self, full):
        self.hold(full)

        rep = full.check()

        assert any("a hold file appeared after the lock" in p for p in rep.problems)

    def test_an_aborted_file_does_not_break_the_check(self, fx):
        folder = fx.data / "labels" / "intent" / "confirm"
        (folder / "haiku-s2-aborted.cost.json").write_text(json.dumps({"cost": 0.14}))

        rep = fx.check()

        assert not rep.problems
        assert rep.spend == pytest.approx(0.08 + 0.14)
