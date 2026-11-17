"""benchmarks/planner/replay.py on synthetic tasks, where the answer is known.

The replay is a pre-registered study: nobody may look at a policy number on
the real rows before the code is reviewed. These tests therefore build tasks
whose right answer follows from how they are constructed (a task where routing
cannot help, a task with a planted weak value, points with a known hull) and
never touch the real golden data. The loader tests use a tiny fixture
written to a temporary directory in the golden layout.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPLAY = Path(__file__).resolve().parents[2] / "benchmarks" / "planner" / "replay.py"
_spec = importlib.util.spec_from_file_location("planner_replay", _REPLAY)
replay = importlib.util.module_from_spec(_spec)
sys.modules["planner_replay"] = replay  # dataclasses look the module up by name
_spec.loader.exec_module(replay)

MODELS = replay.MODELS
ROW_COST = {
    "haiku": 1e-4,
    "luna": 8e-5,
    "sonnet": 5e-3,
    "sol": 5e-3,
    "opus": 5e-3,
}


# ── synthetic tasks ──────────────────────────────────────────────────────


def classification_task(
    gold,
    values,
    row_cost=None,
    kind="accuracy",
    name="intent",
    lens=None,
    valid=None,
    gold_class=None,
):
    """A classification task from the gold codes and each model's answer codes."""
    gold = np.asarray(gold)
    n = len(gold)
    row_cost = row_cost or ROW_COST
    values = {m: np.asarray(values[m]) for m in MODELS}
    n_cells = int(max(v.max() for v in values.values())) + 1
    return replay.make_task(
        name=name,
        gold_kind="original",
        metric_kind=kind,
        ids=[f"r{i}" for i in range(n)],
        mask=np.ones((n, 1), bool),
        gold_class=gold_class if gold_class is not None else np.zeros(n, int),
        value={m: values[m][:, None] for m in MODELS},
        cell={m: values[m][:, None] for m in MODELS},
        correct={m: (values[m] == gold)[:, None] for m in MODELS},
        valid=valid or {m: np.ones(n, bool) for m in MODELS},
        cost={m: (row_cost[m] * n, n) for m in MODELS},
        lens=np.ones(n) if lens is None else lens,
        len_total=float(n if lens is None else np.sum(lens)),
        n_cells=n_cells,
    )


def planted_weak_value_task(n=231, seed=0):
    """Haiku and Luna fail, identically, on two of ten classes; the rest are right.

    Their wrong answers are two values (code 20, 21) that are never a right
    answer, so a router that learns "this value was wrong" loses nothing by
    routing it. Sonnet, Sol and Opus are perfect and cost the same.
    """
    gold = np.random.default_rng(seed).integers(0, 10, n)
    cheap = np.where(gold >= 8, gold + 12, gold)
    return classification_task(
        gold, {"haiku": cheap, "luna": cheap, **{m: gold for m in MODELS[2:]}}
    )


def useless_routing_task(n=231, seed=0):
    """All five models give the same answers, so no policy can change the accuracy."""
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, 10, n)
    wrong = rng.integers(10, 60, n)
    answer = np.where(rng.random(n) < 0.7, gold, wrong)
    return classification_task(gold, {m: answer for m in MODELS})


def attributes_task(n=40, nf=5, bad_field=4, seed=0):
    """Per-field task: the cheap models always miss field `bad_field`; s is perfect."""
    rng = np.random.default_rng(seed)
    mask = np.ones((n, nf), bool)
    cell = {m: np.tile(np.arange(nf), (n, 1)) for m in MODELS}
    correct = {m: np.ones((n, nf), bool) for m in MODELS}
    for m in ("haiku", "luna"):
        correct[m][:, bad_field] = False
    value = {m: rng.integers(0, 3, (n, nf)) for m in MODELS}
    for m in ("luna", "sonnet", "sol", "opus"):
        value[m] = value["haiku"]
    return replay.make_task(
        name="attributes",
        gold_kind="original",
        metric_kind="micro",
        ids=[f"a{i}" for i in range(n)],
        mask=mask,
        gold_class=np.zeros(n, int),
        value=value,
        cell=cell,
        correct=correct,
        valid={m: np.ones(n, bool) for m in MODELS},
        cost={m: (ROW_COST[m] * n, n) for m in MODELS},
        lens=np.ones(n),
        len_total=float(n),
        n_cells=nf,
    )


def small_prereg(**over):
    return {
        **replay.PREREG,
        "splits": 60,
        "bootstrap": 16,
        "bootstrap_quick": 16,
        "bootstrap_splits_per_draw": 5,
        **over,
    }


def one_split(task, train_rows):
    """A single hand-made split: weights (1, n) for the labelled and test rows."""
    train = np.zeros((1, task.n))
    train[0, list(train_rows)] = 1.0
    return train, 1.0 - train


def with_gold_poisoned(task, rows):
    """The same task with every model's right/wrong flipped on `rows`."""
    correct = {m: c.copy() for m, c in task.correct.items()}
    for c in correct.values():
        c[list(rows)] = ~c[list(rows)] & task.mask[list(rows)]
    return dataclasses.replace(task, correct=correct, strict=correct, _fit={})


# ── estimators ───────────────────────────────────────────────────────────


class TestShrunkPrecision:
    def test_reproduces_the_design_worked_example(self):
        # p_bar = 0.81 on 100 labelled rows; value 1 seen once right, value 2
        # once wrong, value 3 once right and once wrong (design section 2).
        codes = [0] * 96 + [1, 2, 3, 3]
        right = [True] * 79 + [False] * 17 + [True, False, True, False]
        table, p_bar = replay.shrunk_precision(codes, right, 4, n_codes=5)

        assert p_bar == pytest.approx(0.81)
        assert table[1] == pytest.approx((1 + 4 * 0.81) / 5)  # 0.848: stays
        assert table[2] == pytest.approx((0 + 4 * 0.81) / 5)  # 0.648: routed
        assert table[3] == pytest.approx((1 + 4 * 0.81) / 6)  # 0.707: routed
        assert table[4] == pytest.approx(0.81)  # unseen value: p_bar
        assert [bool(x < 0.8) for x in table[1:4]] == [False, True, True]

    def test_weights_equal_repeated_rows(self):
        codes = np.array([0, 1, 1, 2])
        right = np.array([True, False, True, True])
        weighted = replay.shrunk_precision(
            codes, right, 4, n_codes=3, weights=[2, 1, 3, 1]
        )
        repeated = replay.shrunk_precision(
            np.repeat(codes, [2, 1, 3, 1]), np.repeat(right, [2, 1, 3, 1]), 4, n_codes=3
        )

        np.testing.assert_allclose(weighted[0], repeated[0])
        assert weighted[1] == pytest.approx(repeated[1])

    def test_a_value_with_no_valid_form_counts_toward_p_bar_only(self):
        table, p_bar = replay.shrunk_precision([0, -1, -1], [True, False, False], 4)

        assert p_bar == pytest.approx(1 / 3)
        assert table[0] == pytest.approx((1 + 4 / 3) / 5)

    def test_batch_fit_matches_the_scalar_function(self):
        task = planted_weak_value_task(n=60)
        rows = [1, 5, 9, 12, 20, 33, 40, 41, 50, 59]
        train_w, _ = one_split(task, rows)
        batch, p_bar = replay._fit_precision(task, "haiku", train_w, 4)
        table, p_bar_1 = replay.shrunk_precision(
            task.cell["haiku"][rows, 0],
            task.correct["haiku"][rows, 0],
            4,
            n_codes=task.n_cells,
        )

        np.testing.assert_allclose(batch[0], table)
        assert p_bar[0] == pytest.approx(p_bar_1)


class TestMetric:
    def test_balanced_accuracy_averages_recall_over_gold_classes(self):
        task = classification_task(
            [0, 0, 0, 1], dict.fromkeys(MODELS, [0, 0, 0, 1]), kind="balanced"
        )
        right = np.array([[True], [True], [False], [True]])
        gold = np.array([0, 0, 0, 1])

        assert replay.metric(task, right, gold, np.ones(4)) == pytest.approx(
            (2 / 3 + 1) / 2
        )
        # Weight 2 on the wrong class-0 row: recall 2/4, class 1 still 1.
        weights = np.array([1.0, 1.0, 2.0, 1.0])
        assert replay.metric(task, right, gold, weights) == pytest.approx(0.75)

    def test_balanced_accuracy_skips_a_class_absent_from_the_test_rows(self):
        task = classification_task(
            [0, 0, 1], dict.fromkeys(MODELS, [0, 0, 1]), kind="balanced"
        )
        right = np.array([[True], [False], [True]])
        value = replay.metric(
            task, right, np.array([0, 0, 1]), np.array([1.0, 1.0, 0.0])
        )

        assert value == pytest.approx(0.5)

    def test_micro_accuracy_counts_fields_not_rows(self):
        task = attributes_task(n=2, nf=4, bad_field=3)
        task.mask[1, 2:] = False  # row 1 has two fields
        right = np.array([[1, 1, 1, 0], [1, 0, 0, 0]], bool)

        # 3 right of 4 fields in row 0, 1 right of 2 in row 1.
        assert replay.metric(task, right, task.gold_class, np.ones(2)) == pytest.approx(
            4 / 6
        )

    def test_per_split_weights_give_one_value_per_split(self):
        task = planted_weak_value_task(n=30)
        w = np.vstack([np.ones(30), np.r_[np.ones(10), np.zeros(20)]])
        out = replay.metric(task, task.correct["haiku"][None], task.gold_class, w)

        assert out.shape == (2,)


class TestHull:
    @staticmethod
    def reference_hull(points, c):
        """Independent upper concave envelope, made non-decreasing, by monotone chain."""
        pts = sorted(set(map(tuple, points)))
        upper = []
        for p in pts:
            while len(upper) >= 2:
                (x1, y1), (x2, y2) = upper[-2], upper[-1]
                if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                    upper.pop()
                else:
                    break
            upper.append(p)
        xs = [p[0] for p in upper]
        ys = list(np.maximum.accumulate([p[1] for p in upper]))
        return float(np.interp(c, xs, ys))

    def test_known_points(self):
        # Cost, metric. (2, .5) lies below the chord from (1, .4) to (4, .9).
        f = replay.hull([(1, 0.4), (2, 0.5), (4, 0.9), (8, 0.8)])

        assert f(1.0)[0] == pytest.approx(0.4)
        assert f(2.5)[0] == pytest.approx(0.4 + 0.5 * 1.5 / 3)  # on the chord
        assert f(4.0)[0] == pytest.approx(0.9)
        assert f(8.0)[0] == pytest.approx(0.9)  # a dearer, worse model adds nothing
        assert f(100.0)[0] == pytest.approx(0.9)
        assert f(0.5)[0] == pytest.approx(0.4)  # below the cheapest: held flat

    def test_matches_an_independent_envelope_on_random_points(self):
        rng = np.random.default_rng(3)
        for _ in range(200):
            pts = np.c_[rng.random(5) * 10 + 0.1, rng.random(5)]
            cs = rng.random(8) * 14
            got = replay.hull(pts)(cs)
            want = [self.reference_hull(pts, max(c, pts[:, 0].min())) for c in cs]
            np.testing.assert_allclose(got, want, atol=1e-9)

    def test_is_per_split(self):
        xs = np.array([[1.0, 2.0], [1.0, 2.0]])
        ys = np.array([[0.2, 0.4], [0.2, 0.8]])
        out = replay.hull_value(xs, ys, np.array([1.5, 1.5]))

        np.testing.assert_allclose(out, [0.3, 0.5])


# ── policies ─────────────────────────────────────────────────────────────


class TestPolicies:
    def params(self, **over):
        return {
            "cheap": "haiku",
            "partner": "luna",
            "strong": "opus",
            "m": 4,
            "tau": 0.8,
            "n_l": 100,
            **over,
        }

    def test_a_policy_that_is_one_hull_model_has_zero_delta(self):
        # Nested correctness: acc 0.6 / 0.5 / 0.7 / 0.65 / 0.95 at costs 1 / 1 / 2 / 3 / 10.
        n = 100
        acc = {"haiku": 0.6, "luna": 0.5, "sonnet": 0.7, "sol": 0.65, "opus": 0.95}
        cost = {"haiku": 1, "luna": 1, "sonnet": 2, "sol": 3, "opus": 10}
        idx = np.arange(n)
        values = {m: np.where(idx < acc[m] * n, 0, 1) for m in MODELS}
        task = classification_task(np.zeros(n, int), values, row_cost=cost)
        train_w, test_w = np.zeros((1, n)), np.ones((1, n))
        pts = replay.model_points(task, test_w)

        def delta(model):
            out = replay.single(task, train_w, test_w, {"model": model})
            ev = replay.evaluate(task, out, test_w, pts, {"model": model}, "opus")
            return ev["delta_flat"][0]

        for on_hull in ("haiku", "sonnet", "opus"):
            assert delta(on_hull) == pytest.approx(0.0, abs=1e-12)
        assert delta("sol") < -0.02  # dearer and worse than Sonnet: under the hull
        assert delta("luna") < -0.05

    def test_cascade_cost_is_the_sum_of_the_models_it_asks(self):
        rng = np.random.default_rng(1)
        n = 50
        gold = rng.integers(0, 4, n)
        noisy = lambda: np.where(rng.random(n) < 0.7, gold, rng.integers(0, 4, n))  # noqa: E731
        task = classification_task(
            gold,
            {
                "haiku": noisy(),
                "luna": noisy(),
                "sonnet": gold,
                "sol": gold,
                "opus": gold,
            },
            lens=rng.integers(10, 200, n).astype(float),
        )
        train_w, test_w = one_split(task, range(10))
        p = self.params()

        for fn in (replay.cascade_agree, replay.agree_or_value):
            right, cost, routed = fn(task, train_w, test_w, p)
            row_routed = routed.any(-1)
            for kind in replay.COST_KINDS:
                want = (
                    task.unit(kind, "haiku")
                    + task.unit(kind, "luna")
                    + row_routed * task.unit(kind, "opus")
                )
                np.testing.assert_allclose(cost[kind], want)
        right, cost, routed = replay.value_router(task, train_w, test_w, p)
        np.testing.assert_allclose(
            cost["flat"], task.u_flat["haiku"] + routed.any(-1) * task.u_flat["opus"]
        )

    def test_cascade_delivers_the_cheap_answer_on_agreement_and_s_otherwise(self):
        gold = np.array([0, 1, 2, 3])
        task = classification_task(
            gold,
            {
                "haiku": [0, 5, 2, 7],  # right, wrong, right, wrong
                "luna": [0, 5, 6, 8],  # agrees on rows 0, 1 only
                "sonnet": gold,
                "sol": gold,
                "opus": gold,
            },
        )
        train_w, test_w = one_split(task, [])
        right, _, routed = replay.cascade_agree(task, train_w, test_w, self.params())

        assert routed[0, :, 0].tolist() == [False, False, True, True]
        # Row 1: both cheap models agree on a wrong answer, so C1 delivers it.
        assert right[0, :, 0].tolist() == [True, False, True, True]

    def test_c3_routes_on_disagreement_or_a_low_value(self):
        task = planted_weak_value_task(n=100)
        good = int(np.flatnonzero(task.value["haiku"][:, 0] < 20)[0])
        # Luna disagrees with Haiku on one row whose Haiku value is a good one.
        luna = task.value["luna"].copy()
        luna[good, 0] += 1
        task = dataclasses.replace(task, value={**task.value, "luna": luna}, _fit={})
        train_w, test_w = one_split(task, range(60))
        _, _, c2 = replay.value_router(task, train_w, test_w, self.params())
        _, _, c3 = replay.agree_or_value(task, train_w, test_w, self.params())

        assert not c2[0, good, 0]
        assert c3[0, good, 0]
        assert c3.sum() == c2.sum() + 1  # C3 = C2 plus the one disagreement

    def test_planted_weak_value_is_learned_and_lifts_over_the_hull(self):
        task = planted_weak_value_task()
        rng = replay.split_rng(replay.PREREG, "intent", "original", 0, 100)
        train = replay.draw_splits(rng, task.n, 100, 40)
        train_w, test_w = train.astype(float), (~train).astype(float)
        pts = replay.model_points(task, test_w)
        p = self.params()
        out = replay.value_router(task, train_w, test_w, p)
        ev = replay.evaluate(task, out, test_w, pts, p, "opus")

        weak = task.value["haiku"][:, 0] >= 20
        assert weak.sum() > 30
        # Routed rows are the weak ones; almost all of them, and nothing else.
        routed = out[2][:, :, 0]
        assert (routed & ~weak).sum() == 0
        assert ev["routed_cheap_wrong"].min() == pytest.approx(1.0)
        assert ev["acc"].mean() > 0.97
        assert ev["delta_flat"].mean() > 0.10  # 10+ points over the hull
        assert ev["d2_flat"].mean() > 0.10

    def test_c2_never_reads_a_test_rows_label(self):
        task = planted_weak_value_task(n=120)
        train_w, test_w = one_split(task, range(0, 120, 3))
        test_rows = [i for i in range(120) if i % 3]
        poisoned = with_gold_poisoned(task, test_rows)
        p = self.params()

        for fn in (replay.value_router, replay.agree_or_value):
            clean = fn(task, train_w, test_w, p)
            dirty = fn(poisoned, train_w, test_w, p)
            np.testing.assert_array_equal(clean[2], dirty[2])
        # Sanity: poisoning the labelled rows does move the routing.
        moved = with_gold_poisoned(task, range(0, 120, 3))
        assert (
            replay.value_router(moved, train_w, test_w, p)[2]
            != replay.value_router(task, train_w, test_w, p)[2]
        ).any()

    def test_the_cascade_learns_nothing(self):
        task = planted_weak_value_task(n=120)
        poisoned = with_gold_poisoned(task, range(120))
        train_w, test_w = one_split(task, range(40))

        np.testing.assert_array_equal(
            replay.cascade_agree(task, train_w, test_w, self.params())[2],
            replay.cascade_agree(poisoned, train_w, test_w, self.params())[2],
        )

    def test_an_invalid_value_is_routed_by_every_policy(self):
        task = planted_weak_value_task(n=60)
        valid = {m: v.copy() for m, v in task.valid.items()}
        valid["haiku"][3] = False
        task = dataclasses.replace(task, valid=valid, _fit={})
        train_w, test_w = one_split(task, range(30, 60))

        for fn in (replay.cascade_agree, replay.value_router, replay.agree_or_value):
            assert fn(task, train_w, test_w, self.params())[2][0, 3, 0]

    def test_per_field_routing_still_pays_the_full_strong_price_per_row(self):
        task = attributes_task(n=40, nf=5, bad_field=4)
        train_w, test_w = one_split(task, range(20))
        p = self.params(cheap="haiku", strong="opus")
        right, cost, routed = replay.value_router(task, train_w, test_w, p)

        assert routed[0, :, 4].all()  # the bad field is routed...
        assert not routed[0, :, :4].any()  # ...and only it
        assert right.all()  # routed field takes s's value
        np.testing.assert_allclose(
            cost["flat"], task.u_flat["haiku"] + task.u_flat["opus"]
        )
        # Opus alone is perfect and cheaper than this: the hull sees no lift.
        pts = replay.model_points(task, test_w)
        ev = replay.evaluate(task, (right, cost, routed), test_w, pts, p, "opus")
        assert ev["delta_flat"][0] == pytest.approx(0.0, abs=1e-12)

    def test_whole_row_routing_sends_every_field_when_any_is_routed(self):
        task = attributes_task(n=30, nf=5, bad_field=4)
        train_w, test_w = one_split(task, range(15))
        p = self.params(gran="row")
        right, _, routed = replay.value_router(task, train_w, test_w, p)

        assert routed.all()
        assert right.all()

    def test_oracle_pays_only_the_cheapest_right_model(self):
        gold = np.array([0, 0, 0, 0])
        task = classification_task(
            gold,
            {
                "haiku": [0, 1, 1, 1],
                "luna": [1, 0, 1, 1],
                "sonnet": [1, 1, 0, 1],
                "sol": [1, 1, 1, 1],
                "opus": [1, 1, 1, 0],
            },
        )
        train_w, test_w = one_split(task, [])
        right, cost, _ = replay.oracle_cheapest_right(task, train_w, test_w, {})

        assert right[0, :, 0].tolist() == [True, True, True, True]
        u = task.u_flat
        # Cheapest right: haiku (row 0), luna (row 1), sonnet (row 2), opus (row 3);
        # luna is cheaper than haiku, so a row both get right goes to luna.
        assert cost["flat"][0].tolist() == pytest.approx(
            [u["haiku"], u["luna"], u["sonnet"], u["opus"]]
        )

    def test_in_sample_oracle_fits_on_the_test_rows_too(self):
        task = planted_weak_value_task(n=100)
        train_w, test_w = one_split(task, [])  # nothing labelled
        p = self.params()
        o3 = replay.oracle_in_sample(task, train_w, test_w, p)
        c2 = replay.value_router(task, train_w, test_w, p)

        weak = task.value["haiku"][:, 0] >= 20
        assert (o3[2][0, :, 0] == weak).all()  # it saw every row's label
        assert not c2[2].any()  # C2 with no labels has nothing to route on


# ── splits and the bootstrap ─────────────────────────────────────────────


class TestSplits:
    def test_each_split_has_exactly_n_l_labelled_rows_and_is_seeded(self):
        a = replay.draw_splits(
            replay.split_rng(replay.PREREG, "intent", "original", 0, 100), 231, 100, 50
        )
        b = replay.draw_splits(
            replay.split_rng(replay.PREREG, "intent", "original", 0, 100), 231, 100, 50
        )
        c = replay.draw_splits(
            replay.split_rng(replay.PREREG, "intent", "original", 0, 50), 231, 50, 50
        )

        assert (a.sum(1) == 100).all()
        np.testing.assert_array_equal(a, b)
        assert (c.sum(1) == 50).all()

    def test_draws_are_uniform_over_rows(self):
        rng = np.random.default_rng(0)
        freq = replay.draw_splits(rng, 40, 10, 4000).mean(0)

        assert np.abs(freq - 0.25).max() < 0.04

    def test_bootstrap_never_puts_a_row_on_both_sides(self):
        rng = np.random.default_rng(5)
        batches = list(replay.bootstrap_batches(rng, 60, 25, 8, 5, 20))

        assert sum(d for *_, d in batches) == 8
        for train_w, test_w, d in batches:
            assert train_w.shape == (d * 5, 60)
            assert (train_w * test_w == 0).all()
            np.testing.assert_allclose(
                (train_w + test_w).sum(1), 60
            )  # n resampled rows
            # The five splits of one draw share the draw's weights.
            both = (train_w + test_w).reshape(d, 5, 60)
            assert (both == both[:, :1]).all()

    def test_bootstrap_weights_are_multinomial_counts(self):
        rng = np.random.default_rng(6)
        train_w, test_w, _ = next(replay.bootstrap_batches(rng, 200, 100, 4, 1, 100))
        w = train_w + test_w

        assert (w == np.round(w)).all()
        assert w.max() >= 2
        assert (w == 0).any()


class TestBootstrapBounds:
    def test_lower_bound_is_the_bonferroni_quantile(self):
        draws = np.random.default_rng(0).normal(5.0, 2.0, 1000)
        prereg = replay.PREREG

        assert replay.lower_bound(draws, prereg) == pytest.approx(
            np.quantile(draws, 0.05 / 3)
        )
        assert replay.lower_bound(draws, prereg) < np.quantile(draws, 0.05)

    def test_bootstrap_draws_are_paired_across_policies(self):
        task = planted_weak_value_task(n=80)
        prereg = small_prereg(n_labelled=40)
        conf = replay.confirmatory_params("intent", prereg)
        configs = {p: (replay.POLICY_FNS[p], conf[p]) for p in replay.POLICIES}
        out = replay.cluster_bootstrap(
            task, configs, prereg, np.random.default_rng(0), 12, "opus"
        )

        assert {k: v["delta"].shape for k, v in out.items()} == dict.fromkeys(
            replay.POLICIES, (12,)
        )
        assert out["C2"]["delta"].mean() > out["C1"]["delta"].mean()


# ── the whole study, and the verdict ─────────────────────────────────────


def run(tasks, **over):
    prereg = small_prereg(**over)
    return replay.run_study(
        {name: {"original": t, "adjudicated": t} for name, t in tasks.items()},
        prereg,
        quick=True,
        log=lambda *_: None,
    )


class TestStudy:
    def test_routing_that_cannot_help_is_killed(self):
        res = run({"intent": useless_routing_task()})
        conf = res["tasks"]["intent"]["confirmatory"]

        # Same answers everywhere: every policy equals the hull exactly.
        for pol in replay.POLICIES:
            assert conf[pol]["delta_flat"] == pytest.approx(0.0, abs=1e-9)
            assert conf[pol]["d2_flat"] == pytest.approx(0.0, abs=1e-9)
        assert res["verdict"]["verdict"] == "KILL"

    def test_a_planted_weak_value_builds_the_ruler_fed_planner(self):
        res = run({"intent": planted_weak_value_task()})
        conf = res["tasks"]["intent"]["confirmatory"]
        v = res["verdict"]

        assert conf["C2"]["delta_flat"] > 10
        assert conf["C2"]["delta_length"] > 10
        assert conf["C2"]["delta_lb"] > 0
        # Luna repeats Haiku's mistakes, so the cascade never routes them.
        assert conf["C1"]["delta_flat"] < 1
        assert v["verdict"] == "BUILD_RULER_FED_PLANNER"
        assert v["scope"] == "classification_only_single_task"

    def test_results_embed_both_hashes_and_the_constants(self):
        res = run({"intent": useless_routing_task()}, splits=20)
        meta = res["meta"]
        prereg = small_prereg(splits=20)

        assert (
            meta["prereg_sha256"]
            == hashlib.sha256(json.dumps(prereg, sort_keys=True).encode()).hexdigest()
        )
        assert meta["prereg"] == prereg
        assert meta["design_sha256"] == replay.design_hash()

    def test_exploratory_grid_cannot_change_the_verdict(self):
        base = run({"intent": useless_routing_task()})
        with_grid = base["verdict"]["verdict"]
        for g in base["tasks"]["intent"]["golds"]["original"]["grid"].values():
            g["summary"]["delta_flat"]["mean"] = 99.0  # an amazing grid point
        assert replay.verdict(base, small_prereg())["verdict"] == with_grid == "KILL"

    def test_results_are_reproducible(self):
        a = run({"intent": planted_weak_value_task()}, splits=30)
        b = run({"intent": planted_weak_value_task()}, splits=30)

        assert (
            a["tasks"]["intent"]["confirmatory"] == b["tasks"]["intent"]["confirmatory"]
        )

    def test_outputs_are_written(self, tmp_path):
        res = run({"intent": planted_weak_value_task()}, splits=20)
        paths = replay.write_results(res, tmp_path)

        names = sorted(p.name for p in paths)
        assert names == ["frontier-intent.svg", "replay.json", "replay.md"]
        assert "BUILD_RULER_FED_PLANNER" in (tmp_path / "replay.md").read_text()
        assert json.loads((tmp_path / "replay.json").read_text())["verdict"]
        svg = (tmp_path / "frontier-intent.svg").read_text()
        assert svg.startswith("<svg")
        assert "<polyline" in svg
        assert "C2" in svg

    def test_all_three_tasks_run_end_to_end(self, tmp_path):
        rng = np.random.default_rng(2)
        n = 200
        gold = rng.integers(0, 2, n)
        flip = lambda p: np.where(rng.random(n) < p, 1 - gold, gold)  # noqa: E731
        faith = classification_task(
            gold,
            {
                "haiku": flip(0.2),
                "luna": flip(0.25),
                "sonnet": flip(0.1),
                "sol": flip(0.1),
                "opus": flip(0.1),
            },
            kind="balanced",
            name="faithfulness",
            gold_class=gold,
        )
        res = run(
            {
                "intent": planted_weak_value_task(),
                "faithfulness": faith,
                "attributes": attributes_task(n=199, nf=6),
            },
            splits=20,
        )

        assert list(res["tasks"]) == ["intent", "faithfulness", "attributes"]
        attr = res["tasks"]["attributes"]["golds"]["original"]["confirmatory"]["C2"]
        assert attr["acc_strict"]["mean"] is not None  # secondary metrics exist
        assert attr["success_rate"]["mean"] is not None
        assert res["tasks"]["faithfulness"]["metric_kind"] == "balanced"
        paths = replay.write_results(res, tmp_path)
        assert len(paths) == 5  # md, json, three frontiers

    def test_grid_sizes_match_the_design(self):
        assert len(replay.grid_configs("intent", replay.PREREG)) == 75
        assert len(replay.grid_configs("faithfulness", replay.PREREG)) == 75
        attr = replay.grid_configs("attributes", replay.PREREG)
        assert len(attr) == 78  # 75 + the three whole-row variants (design section 5)
        ids = [replay.config_id(p, q) for p, q in attr]
        assert len(set(ids)) == len(ids)


def entry(
    delta=0.0, delta_len=None, d2=0.0, d2_len=None, delta_lb=0.0, d2_lb=-9.0, cost=1.0
):
    return {
        "delta_flat": delta,
        "delta_length": delta if delta_len is None else delta_len,
        "d2_flat": d2,
        "d2_length": d2 if d2_len is None else d2_len,
        "delta_lb": delta_lb,
        "d2_lb": d2_lb,
        "cost_flat": cost,
        "cost_length": cost,
    }


def task_result(c1=None, c2=None, c3=None, paired=None):
    nothing = entry()
    return {
        "confirmatory": {"C1": c1 or nothing, "C2": c2 or nothing, "C3": c3 or nothing},
        "paired": paired
        or {
            p: {"diff_flat": 0.0, "diff_length": 0.0, "diff_lb": -1.0}
            for p in ("C2", "C3")
        },
    }


LIFT = {"delta": 4.0, "delta_lb": 1.0}
CUT = {"delta": 1.5, "d2": 0.5, "d2_lb": -1.0}
BEATS = {
    "C2": {"diff_flat": 2.5, "diff_length": 2.5, "diff_lb": 0.3},
    "C3": {"diff_flat": 0.0, "diff_length": 0.0, "diff_lb": -1.0},
}


class TestVerdictLogic:
    def decide(self, **tasks):
        results = {"tasks": {t: task_result(**kw) for t, kw in tasks.items()}}
        return replay.verdict(results, replay.PREREG)

    def test_kill_when_nothing_passes_on_intent(self):
        # Faithfulness passing does not rescue a task the design says decides.
        v = self.decide(intent={}, faithfulness={"c2": entry(**LIFT)})

        assert v["verdict"] == "KILL"

    def test_lift_thresholds_are_inclusive_for_the_mean_and_strict_for_the_bound(self):
        assert replay.policy_passes(entry(delta=3.0, delta_lb=0.01), replay.PREREG)[
            "lift"
        ]
        assert not replay.policy_passes(entry(delta=2.99, delta_lb=1.0), replay.PREREG)[
            "pass"
        ]
        assert not replay.policy_passes(entry(delta=5.0, delta_lb=0.0), replay.PREREG)[
            "lift"
        ]

    def test_cut_thresholds(self):
        base = {"delta": 1.0, "d2": 0.0, "d2_lb": -2.0}

        assert replay.policy_passes(entry(**base), replay.PREREG)["cut"]
        for bad in ({"delta": 0.99}, {"d2": -0.01}, {"d2_lb": -2.01}):
            assert not replay.policy_passes(entry(**{**base, **bad}), replay.PREREG)[
                "cut"
            ]

    def test_a_rule_must_hold_under_both_cost_models(self):
        e = entry(delta=4.0, delta_len=2.0, delta_lb=1.0)

        assert not replay.policy_passes(e, replay.PREREG)["pass"]

    def test_ruler_fed_when_c2_passes_and_c1_fails(self):
        v = self.decide(intent={"c2": entry(**LIFT)})

        assert v["verdict"] == "BUILD_RULER_FED_PLANNER"
        assert v["scope"] == "classification_only_single_task"

    def test_cascade_only_when_only_c1_passes(self):
        v = self.decide(intent={"c1": entry(**LIFT)})

        assert v["verdict"] == "BUILD_DISAGREEMENT_CASCADE_ONLY"

    def test_c1_and_c2_both_pass_without_a_clear_margin_gives_the_cascade(self):
        v = self.decide(intent={"c1": entry(**LIFT), "c2": entry(**LIFT)})

        assert v["verdict"] == "BUILD_DISAGREEMENT_CASCADE_ONLY"

    def test_c2_beating_c1_by_the_margin_at_no_more_cost_gives_ruler_fed(self):
        v = self.decide(
            intent={
                "c1": entry(**LIFT, cost=1.0),
                "c2": entry(**LIFT, cost=1.0),
                "paired": BEATS,
            }
        )

        assert v["verdict"] == "BUILD_RULER_FED_PLANNER"

    @pytest.mark.parametrize(
        "change",
        [
            {"diff_flat": 1.9},
            {"diff_length": 1.9},
            {"diff_lb": 0.0},
        ],
    )
    def test_the_margin_over_c1_needs_size_and_a_paired_bound(self, change):
        paired = {**BEATS, "C2": {**BEATS["C2"], **change}}
        v = self.decide(
            intent={"c1": entry(**LIFT), "c2": entry(**LIFT), "paired": paired}
        )

        assert v["verdict"] == "BUILD_DISAGREEMENT_CASCADE_ONLY"

    def test_the_margin_over_c1_needs_no_higher_cost(self):
        v = self.decide(
            intent={
                "c1": entry(**LIFT, cost=1.0),
                "c2": entry(**LIFT, cost=1.01),
                "paired": BEATS,
            }
        )

        assert v["verdict"] == "BUILD_DISAGREEMENT_CASCADE_ONLY"

    def test_scope_is_general_when_another_task_passes_with_the_same_family(self):
        general = self.decide(
            intent={"c2": entry(**LIFT)}, attributes={"c3": entry(**LIFT)}
        )
        narrow = self.decide(
            intent={"c2": entry(**LIFT)}, faithfulness={"c1": entry(**LIFT)}
        )

        assert general["scope"] == "general_floor"
        assert narrow["scope"] == "classification_only_single_task"

    def test_the_sign_flip_flag_is_reported_not_decisive(self):
        res = run({"intent": planted_weak_value_task()})
        res["tasks"]["intent"]["confirmatory"]["C2"]["sign_flip"] = True

        assert (
            replay.verdict(res, small_prereg())["verdict"] == "BUILD_RULER_FED_PLANNER"
        )


# ── loading and integrity, on a tiny golden-layout fixture ───────────────

INTENT_NAMES = ["card_arrival", "lost_card", "top_up"]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_layout(root: Path) -> Path:
    """Write dev rows, five models' labels, costs and the baseline report."""
    data = root / "data"
    intent = [
        {"id": f"i{k}", "text": f"msg {k}", "label": INTENT_NAMES[k % 3]}
        for k in range(6)
    ]
    faith = [
        {
            "id": f"f{k}",
            "doc": "doc " * 5,
            "summary": "sum",
            "label": ["consistent", "inconsistent"][k % 2],
        }
        for k in range(6)
    ]
    attrs = [
        {
            "id": f"a{k}",
            "category": "Shoes" if k % 2 else "Hats",
            "title": "title",
            "description": "desc",
            "target": {"Color": ["Red"], "Size": ["8 1/2", "8.5"], "Brand": ["n/a"]},
        }
        for k in range(4)
    ]
    write_jsonl(data / "intent" / "dev.jsonl", intent)
    write_jsonl(data / "intent" / "dev-adjudicated.jsonl", intent[:5])
    (data / "intent" / "labels.json").write_text(json.dumps(INTENT_NAMES))
    write_jsonl(data / "faithfulness" / "dev.jsonl", faith)
    write_jsonl(data / "faithfulness" / "dev-adjudicated.jsonl", faith[1:])
    write_jsonl(data / "attributes" / "dev.jsonl", attrs)
    # A held-out file with garbage in it: the study must never parse it.
    (data / "intent" / "holdout.jsonl").write_text("not json\n")

    def answers(task, model, rows):
        out = []
        wrong = {"haiku": 2, "luna": 3, "sonnet": 1, "sol": 1, "opus": 0}[model]
        for k, r in enumerate(rows):
            bad = k < wrong
            if task == "intent":
                a = "top_up" if bad else r["label"]
                if model == "sonnet" and k == 5:
                    a = f"Thinking about it.\n**{r['label']}**"  # reasoning first
            elif task == "faithfulness":
                a = "maybe" if bad else r["label"]
            else:
                size = "9" if bad else "8.5"
                a = json.dumps({"Color": "Red", "Size": size, "Brand": "n/a"})
                if model == "haiku" and k == 3:
                    a = "```json\n" + a + "\n```"
                if model == "luna" and k == 2:
                    a = "no json here"
            out.append({"id": r["id"], "answer": a})
        return out

    rows = {"intent": intent, "faithfulness": faith, "attributes": attrs}
    price = {"haiku": 0.03, "luna": 0.02, "sonnet": 0.4, "sol": 0.3, "opus": 1.0}
    for task, task_rows in rows.items():
        for model in MODELS:
            labels = data / "labels" / task / f"{model}.jsonl"
            write_jsonl(labels, answers(task, model, task_rows))
            labels.with_suffix(".cost.json").write_text(
                json.dumps({"cost": price[model], "tokens": 1, "rows": len(task_rows)})
            )
    # The baseline, produced the way score.py's main() produces it.
    report = {}
    for task, task_rows in rows.items():
        adj = (
            replay.read_jsonl(
                data / task / "dev-adjudicated.jsonl", data, missing_ok=True
            )
            or task_rows
        )
        report[task] = {}
        for model in MODELS:
            ans = {
                r["id"]: r["answer"]
                for r in replay.read_jsonl(
                    data / "labels" / task / f"{model}.jsonl", data
                )
            }
            report[task][model] = {
                "original": replay.score.score(task, task_rows, ans),
                "adjudicated": replay.score.score(task, adj, ans),
                "cost_usd": price[model],
            }
    (data / "baseline-report.json").write_text(json.dumps(report))
    return data


STRONG = {"intent": "opus", "faithfulness": "opus", "attributes": "opus"}


@pytest.fixture
def layout(tmp_path):
    return make_layout(tmp_path)


def check(data):
    return replay.check_integrity(data, {**replay.PREREG, "strong": STRONG})


class TestLoading:
    def test_a_consistent_layout_passes(self, layout):
        rep = check(layout)

        assert rep.ok, rep.problems

    def test_the_task_arrays_reproduce_score_dot_score(self, layout):
        for task in replay.TASKS:
            for gold in replay.GOLDS:
                t = replay.load_task(task, gold, layout)
                rows = replay.read_jsonl(layout / task / "dev.jsonl", layout)
                adj = replay.read_jsonl(
                    layout / task / "dev-adjudicated.jsonl", layout, missing_ok=True
                )
                rows = rows if gold == "original" else (adj or rows)
                for m in MODELS:
                    ans = {
                        r["id"]: r["answer"]
                        for r in replay.read_jsonl(
                            layout / "labels" / task / f"{m}.jsonl", layout
                        )
                    }
                    want = replay.score.score(task, rows, ans)
                    got = replay._derived_score(t, m)
                    assert got == pytest.approx(want), (task, gold, m)

    def test_attributes_load_as_fields_and_cells(self, layout):
        t = replay.load_task("attributes", "original", layout)

        assert t.mask.shape == (4, 3)
        assert t.mask.all()
        assert t.valid["luna"].tolist() == [
            True,
            True,
            False,
            True,
        ]  # row 2 unparseable
        # The same attribute on the same category is the same cell; n/a is its own.
        assert t.cell["opus"][0, 0] == t.cell["opus"][2, 0]
        assert t.cell["opus"][0, 0] != t.cell["opus"][1, 0]  # Hats vs Shoes
        assert t.cell["opus"][0, 2] not in (t.cell["opus"][0, 0], t.cell["opus"][0, 1])
        assert t.same_as_original is False
        assert replay.load_task("attributes", "adjudicated", layout).same_as_original

    def test_a_reasoning_first_answer_is_read_from_its_last_line(self, layout):
        t = replay.load_task("intent", "original", layout)

        assert t.correct["sonnet"][5, 0]
        assert t.valid["sonnet"][5]

    def test_invalid_values_are_flagged(self, layout):
        t = replay.load_task("faithfulness", "original", layout)

        assert (~t.valid["luna"]).sum() == 3  # "maybe" on its first three rows

    def test_a_missing_row_in_a_label_file_fails_the_check(self, layout):
        f = layout / "labels" / "intent" / "haiku.jsonl"
        f.write_text("".join(f.read_text().splitlines(True)[:-1]))

        rep = check(layout)

        assert not rep.ok
        assert any("intent/haiku" in p and "cover" in p for p in rep.problems)

    def test_a_duplicate_id_fails_the_check(self, layout):
        f = layout / "labels" / "intent" / "sol.jsonl"
        lines = f.read_text().splitlines(True)
        f.write_text("".join(lines + lines[:1]))

        assert any("duplicate" in p for p in check(layout).problems)

    def test_a_baseline_that_does_not_match_score_fails_the_check(self, layout):
        path = layout / "baseline-report.json"
        report = json.loads(path.read_text())
        report["intent"]["sol"]["original"]["accuracy"] += 1e-6
        path.write_text(json.dumps(report))

        rep = check(layout)

        assert not rep.ok
        assert any("baseline" in p for p in rep.problems)

    def test_a_wrong_strong_model_fails_the_check(self, layout):
        rep = replay.check_integrity(
            layout, {**replay.PREREG, "strong": {**STRONG, "intent": "sol"}}
        )

        assert any("best single model" in p for p in rep.problems)


class TestHoldoutGuard:
    def test_the_held_out_split_is_refused_in_every_place_it_could_be(self, layout):
        for rel in (
            "intent/holdout.jsonl",
            "labels/intent/holdout.jsonl",
            "attributes/holdout.jsonl",
            "labels/intent/haiku-holdout.jsonl",
            "labels/holdout/haiku.jsonl",
        ):
            with pytest.raises(PermissionError):
                replay.guard_path(layout / rel, layout)

    def test_only_the_listed_files_are_readable(self, layout):
        ok = (
            "intent/dev.jsonl",
            "faithfulness/dev-adjudicated.jsonl",
            "labels/attributes/opus.jsonl",
            "labels/attributes/opus.cost.json",
            "intent/labels.json",
            "baseline-report.json",
        )
        for rel in ok:
            assert replay.guard_path(layout / rel, layout)
        for rel in (
            "attributes/schemas.json",
            "intent/dev.jsonl.bak",
            "labels/intent/gpt.jsonl",
            "labels/intent/opus-candidates.jsonl",
            "adjudication/intent.jsonl",
        ):
            with pytest.raises(PermissionError):
                replay.guard_path(layout / rel, layout)

    def test_a_path_that_escapes_the_data_directory_is_refused(self, layout):
        with pytest.raises(PermissionError):
            replay.guard_path(
                layout / "intent" / ".." / ".." / "x" / "dev.jsonl", layout
            )

    def test_the_loader_never_opens_a_held_out_file(self, layout):
        # check() above passes with garbage in intent/holdout.jsonl; make it unreadable too.
        (layout / "intent" / "holdout.jsonl").chmod(0)
        try:
            assert check(layout).ok
        finally:
            (layout / "intent" / "holdout.jsonl").chmod(0o644)

    def test_the_word_holdout_appears_in_the_script_only_inside_the_guard(self):
        source = _REPLAY.read_text()
        guard = inspect.getsource(replay.guard_path) + inspect.getsource(
            replay.guard_selftest
        )

        assert source.lower().count("holdout") >= 1
        assert source.lower().count("holdout") == guard.lower().count("holdout")


class TestPreregistration:
    def test_hash_is_the_sha256_of_the_sorted_json(self):
        want = hashlib.sha256(
            json.dumps(replay.PREREG, sort_keys=True).encode()
        ).hexdigest()

        assert replay.prereg_hash() == want

    def test_constants_are_the_designs(self):
        p = replay.PREREG

        assert (p["seed"], p["splits"], p["bootstrap"]) == (20261117, 1000, 1000)
        assert (p["bootstrap_quick"], p["bootstrap_splits_per_draw"]) == (100, 20)
        assert (p["shrinkage_m"], p["tau"], p["n_labelled"]) == (4, 0.8, 100)
        assert p["strong"] == {
            "intent": "opus",
            "faithfulness": "sol",
            "attributes": "sonnet",
        }
        assert p["pass"] == {
            "lift_delta_pts": 3.0,
            "cut_d2_mean_pts": 0.0,
            "cut_d2_lb_pts": -2.0,
            "cut_delta_pts": 1.0,
        }
        assert p["alpha"] / p["n_confirmatory"] == pytest.approx(0.05 / 3)
        assert p["grid"]["tau"] == [0.6, 0.7, 0.8, 0.9]
        assert p["grid"]["n_labelled"] == [50, 100]
        assert p["verdict"]["beat_c1_pts"] == 2.0
        assert p["hull_cost_multiple"] == 2.0

    def test_the_design_file_hash_is_embedded(self):
        design = replay.DESIGN
        if not design.exists():
            pytest.skip("design file not present in this checkout")

        assert replay.design_hash() == hashlib.sha256(design.read_bytes()).hexdigest()
