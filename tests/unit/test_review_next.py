"""benchmarks/review_next/review_next.py on synthetic data.

Review-next is pre-registered: nobody may look at a ranker's precision, lift or
gain on the real dev rows before the code is reviewed. Every test therefore
builds a task, a set of samples or a golden-layout fixture whose right answer
follows from how it is constructed. Nothing reads the real data directory,
nothing calls a model. Each test names the regression it would catch.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from test_planner_replay import (  # noqa: E402
    STRONG,
    classification_task,
    make_layout,
    one_split,
    with_gold_poisoned,
    write_jsonl,
)

_ROOT = _HERE.parents[1]
_MODULE = _ROOT / "benchmarks" / "review_next" / "review_next.py"
_spec = importlib.util.spec_from_file_location("review_next", _MODULE)
rn = importlib.util.module_from_spec(_spec)
sys.modules["review_next"] = rn
_spec.loader.exec_module(rn)
replay = rn.replay
addendum1 = rn.addendum1

FRACS = rn.PREREG_REVIEW["k_fractions"]
PRIMARY = rn.PREREG_REVIEW["primary_k"]


# ── k and the expected hits ──────────────────────────────────────────────


class TestKAtAndExpectedHits:
    def test_k_rounds_half_up_not_to_even(self):
        # Python's round(6.5) is 6; the design says floor(f*W + 1/2) in fractions.
        assert rn.k_at("0.05", 130) == 7
        assert round(0.05 * 130) == 6  # the trap this test guards against
        assert [rn.k_at(f, 131) for f in FRACS] == [7, 13, 26]
        assert [rn.k_at(f, 100) for f in FRACS] == [5, 10, 20]

    def test_k_is_clipped_to_the_test_weight(self):
        assert rn.k_at("0.05", 3) == 1  # never zero
        assert rn.k_at("0.20", 1) == 1  # never above W

    def test_k_takes_a_vector_of_weights_and_agrees_with_the_scalar(self):
        w = np.array([100, 130, 131, 7])

        got = rn.k_at("0.10", w)

        assert got.tolist() == [rn.k_at("0.10", int(x)) for x in w]

    def test_all_keys_tied_gives_k_times_the_base_rate_whatever_the_row_order(self):
        # Wrong rows first, in input order: a stable sort that leaks row order
        # would credit the first k rows and report precision 1.
        hit = np.array([True] * 4 + [False] * 16)
        keys = np.zeros((20, 1))
        w = np.ones(20)

        got = rn.expected_hits(keys, hit, w, 5)

        assert got == pytest.approx(5 * 4 / 20)
        assert rn.expected_hits(keys, hit[::-1], w, 5) == pytest.approx(5 * 4 / 20)

    def test_the_boundary_group_adds_a_fraction_of_its_errors(self):
        # Keys 3,2,2,2,1, k = 2: the 3-row counts fully (1 hit); the three 2-rows
        # hold 1 hit and share the one remaining slot: 1/3. Total 4/3.
        keys = np.array([[3.0], [2.0], [2.0], [2.0], [1.0]])
        hit = np.array([True, True, False, False, True])

        assert rn.expected_hits(keys, hit, np.ones(5), 2) == pytest.approx(4 / 3)

    def test_a_row_with_weight_two_equals_the_row_duplicated(self):
        keys = np.array([[2.0], [1.0], [1.0], [0.0]])
        hit = np.array([True, True, False, False])
        weighted = rn.expected_hits(keys, hit, np.array([1.0, 2.0, 1.0, 1.0]), 3)
        copies = rn.expected_hits(
            np.repeat(keys, [1, 2, 1, 1], axis=0),
            np.repeat(hit, [1, 2, 1, 1]),
            np.ones(5),
            3,
        )

        assert weighted == pytest.approx(copies)
        # The top row counts (1); the tied group has 3 copies holding 2 hits and
        # shares the 2 remaining slots: 2 * 2/3.
        assert weighted == pytest.approx(1 + 2 * 2 / 3)

    def test_a_row_of_weight_zero_is_excluded(self):
        keys = np.array([[5.0], [1.0], [1.0]])
        hit = np.array([True, True, False])

        with_zero = rn.expected_hits(keys, hit, np.array([0.0, 1.0, 1.0]), 1)

        assert with_zero == pytest.approx(0.5)  # the top row does not exist

    def test_it_equals_the_average_over_every_random_tie_break(self):
        # Independent oracle: enumerate every order of the copies, break ties
        # by that order, and average the hits in the first k.
        keys = np.array([[2.0], [1.0], [1.0], [1.0], [0.0]])
        hit = np.array([False, True, False, True, True])
        w = np.array([1.0, 2.0, 1.0, 1.0, 1.0])
        copies = [(i, c) for i in range(5) for c in range(int(w[i]))]
        k = 3
        total, n_orders = 0.0, 0
        for perm in itertools.permutations(range(len(copies))):
            ranked = sorted(
                (copies[j] for j in perm),
                key=lambda ic: -keys[ic[0], 0],  # stable: ties keep perm order
            )
            total += sum(hit[i] for i, _ in ranked[:k])
            n_orders += 1

        assert rn.expected_hits(keys, hit, w, k) == pytest.approx(total / n_orders)

    def test_a_batch_of_splits_equals_the_splits_one_at_a_time(self):
        rng = np.random.default_rng(3)
        keys = rng.integers(0, 3, (6, 12, 2)).astype(float)
        hit = rng.random(12) < 0.4
        w = rng.integers(0, 3, (6, 12)).astype(float)
        k = np.array([2, 3, 1, 4, 2, 3])

        batch = rn.expected_hits_batch(keys, hit, w, k)

        for s in range(6):
            assert batch[s] == pytest.approx(rn.expected_hits(keys[s], hit, w[s], k[s]))


# ── synthetic tasks and samples ──────────────────────────────────────────

N = 231


def planted(
    signal: str, n: int = N, rate: float = 0.2, seed: int = 0, luna: str = "same"
):
    """Haiku is wrong on about `rate` of the rows; the extra samples carry `signal`.

    The wrong answer is another valid class, so no delivered value is wrong more
    often than another: the per-value rate R1 has nothing to learn. Luna repeats
    Haiku, so P = 0, unless `luna` is "right": Luna is perfect, so P marks
    exactly Haiku's errors. `signal` is "errors" (every wrong row has D = 3, every
    right row D = 0), "independent" (a random 20% of rows have D = 3) or "none".
    """
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, 10, n)
    wrong = rng.random(n) < rate
    cheap = np.where(wrong, (gold + 1) % 10, gold)
    task = classification_task(
        gold,
        {
            "haiku": cheap,
            "luna": {"same": cheap, "right": gold}[luna],
            **{m: gold for m in replay.MODELS[2:]},
        },
    )
    marked = {
        "errors": wrong,
        "independent": rng.random(n) < 0.2,
        "none": np.zeros(n, bool),
    }[signal]
    extra = np.column_stack([np.where(marked, 40 + j, cheap) for j in range(3)])
    codes = np.column_stack([cheap, extra])
    samples = addendum1.SampleSet(
        list(task.ids), codes, np.ones(codes.shape, bool), codes == gold[:, None]
    )
    return task, samples, wrong


def small(**over):
    """A reduced PREREG_REVIEW: 60 splits, 16 bootstrap draws of 5 splits."""
    return {
        **rn.PREREG_REVIEW,
        "splits": 60,
        "bootstrap": 16,
        "bootstrap_quick": 16,
        "bootstrap_splits_per_draw": 5,
        **over,
    }


def study(task, samples, prereg=None, checks=None, adjudicated=None):
    tasks = {
        task.name: {
            "original": (task, samples),
            "adjudicated": adjudicated
            or (dataclasses.replace(task, same_as_original=True), samples),
        }
    }
    return rn.run_study(
        tasks,
        checks or rn.Checks(),
        prereg or small(),
        quick=True,
        log=lambda *_: None,
    )


# ── the signals ──────────────────────────────────────────────────────────


class TestSignals:
    def test_dissent_counts_differences_from_the_delivered_answer_not_the_mode(self):
        # r0: original a, samples b, b, b. The mode is b, so a mode-based count
        # would give 0; against the delivered answer it is 3.
        # r1: the first sample has no answer (a null): a code of its own, dissent.
        # r2: one sample says z.
        ids = ["r0", "r1", "r2"]
        columns = [
            {"r0": "a", "r1": "a", "r2": "a"},  # the delivered run
            {"r0": "b", "r2": "a"},
            {"r0": "b", "r1": "a", "r2": "z"},
            {"r0": "b", "r1": "a", "r2": "a"},
        ]
        samples = addendum1.build_sample_set(ids, columns, {"a", "b"}, ["a", "a", "a"])

        assert rn.dissent(samples).tolist() == [3, 1, 1]
        assert rn.dissent(samples, [1]).tolist() == [1, 1, 0]
        assert rn.dissent(samples, [2, 3]).tolist() == [2, 0, 1]

    def test_a_row_with_every_sample_equal_to_the_original_has_no_dissent(self):
        _, samples, _ = planted("none")

        assert not rn.dissent(samples).any()

    def test_the_pair_signal_is_the_parents_c1_disagreement(self):
        gold = np.arange(8) % 4
        luna = gold.copy()
        luna[[1, 5]] = 9  # two rows where the pair differs
        task = classification_task(
            gold, {"haiku": gold, "luna": luna, **{m: gold for m in replay.MODELS[2:]}}
        )

        assert rn.pair_disagree(task).tolist() == [0, 1, 0, 0, 0, 1, 0, 0]

    def test_unseen_value_gets_the_overall_precision_so_its_key_is_one_minus_pbar(self):
        task, _, _ = planted("none")
        train_w, _ = one_split(task, range(10))
        cheap = task.value["haiku"][:, 0]
        seen = set(cheap[:10].tolist())
        unseen = next(i for i in range(10, task.n) if cheap[i] not in seen)

        score = rn.r1_score(task, train_w)
        p_bar = task.correct["haiku"][:10, 0].mean()

        assert score[0, unseen] == pytest.approx(1 - p_bar)


class TestRankerKeys:
    def test_tiers_are_separate_columns_so_a_summed_float_cannot_cross_them(self):
        # D = 1 with the worst r1 (1.0) must not outrank D = 2 with the best (0.0):
        # D + r1 would tie them at 2.0. With separate columns the D = 2 row is first.
        keys = np.array([[1.0, 1.0], [2.0, 0.0]])
        hit = np.array([False, True])

        assert rn.expected_hits(keys, hit, np.ones(2), 1) == 1.0

    def test_the_keys_of_a_small_fixture_equal_the_hand_computed_values(self):
        # Delivered [0,0,1,1,2,2] against gold [0,3,1,3,2,3]: right on rows 0, 2, 4.
        # Labelled rows 0, 1, 2, 4: p-bar = 3/4 and, with m = 4,
        #   p(0) = (1 + 3) / (2 + 4) = 2/3, p(1) = (1 + 3) / (1 + 4) = 4/5,
        #   p(2) = (1 + 3) / (1 + 4) = 4/5: the r1 column is 1/3, 1/3, 1/5 ...
        # Luna differs on rows 1 and 3 (P); sample 1 and 2 differ on row 3, sample 3
        # on row 5 (D = 2 and 1).
        gold = np.array([0, 3, 1, 3, 2, 3])
        cheap = np.array([0, 0, 1, 1, 2, 2])
        luna = cheap.copy()
        luna[[1, 3]] += 5
        task = classification_task(
            gold,
            {"haiku": cheap, "luna": luna, **{m: gold for m in replay.MODELS[2:]}},
        )
        codes = np.column_stack([cheap, cheap, cheap, cheap])
        codes[3, 1:3] = 90
        codes[5, 3] = 91
        samples = addendum1.SampleSet(
            list(task.ids), codes, np.ones(codes.shape, bool), codes == gold[:, None]
        )
        train_w, _ = one_split(task, [0, 1, 2, 4])
        r1 = np.array([1 / 3, 1 / 3, 1 / 5, 1 / 5, 1 / 5, 1 / 5])

        def keys(name):
            return rn.ranker_keys(name, task, samples, train_w)[0]

        np.testing.assert_allclose(keys("R1"), r1[:, None])
        np.testing.assert_allclose(
            keys("R2"), np.column_stack([[0, 0, 0, 2, 0, 1], r1])
        )
        np.testing.assert_allclose(
            keys("R4"), np.column_stack([[0, 1, 0, 3, 0, 1], r1])
        )
        np.testing.assert_allclose(
            keys("E1"), np.column_stack([[0, 1, 0, 1, 0, 0], r1])
        )
        np.testing.assert_allclose(
            keys("E3"), np.column_stack([[0, 0, 0, 2, 0, 1], [0, 1, 0, 1, 0, 0], r1])
        )
        np.testing.assert_allclose(keys("E2"), np.array([[0, 0, 0, 2, 0, 1]]).T)

    def test_r4_adds_the_pair_disagreement_as_one_more_dissenting_answer(self):
        task, samples, _ = planted("none")
        luna = task.value["haiku"][:, 0].copy()
        luna[7] += 1
        task = dataclasses.replace(
            task, value={**task.value, "luna": luna[:, None]}, _fit={}
        )
        samples.codes[7, 1] = 99  # D = 1 on row 7 as well
        train_w, _ = one_split(task, range(100))

        r2 = rn.ranker_keys("R2", task, samples, train_w)
        r4 = rn.ranker_keys("R4", task, samples, train_w)

        assert r2[0, 7, 0] == 1
        assert r4[0, 7, 0] == 2
        assert (r4[0, :, 0] - r2[0, :, 0]).sum() == 1

    def test_subset_rankers_read_only_their_samples(self):
        task, samples, _ = planted("none")
        samples.codes[3, 3] = 77  # only the third sample dissents, on row 3
        train_w, _ = one_split(task, range(100))

        def d_of(name):
            return rn.ranker_keys(name, task, samples, train_w)[0, 3, 0]

        assert [d_of(x) for x in ("R2", "E4", "E5", "E6", "E7")] == [1, 0, 1, 1, 0]

    def test_r0_is_one_constant_column(self):
        task, samples, _ = planted("errors")
        train_w, _ = one_split(task, range(100))

        keys = rn.ranker_keys("R0", task, samples, train_w)

        assert keys.shape == (1, task.n, 1)
        assert not keys.any()

    def test_r1_never_reads_a_test_rows_label_but_does_read_the_labelled_ones(self):
        # Flip the correctness of every test row: R1's keys do not change. Flip
        # one labelled row: they do. Catches a fit on the test labels.
        task, samples, _ = planted("none")
        train_rows = range(100)
        train_w, _ = one_split(task, train_rows)
        base = rn.ranker_keys("R1", task, samples, train_w)

        poisoned_test = with_gold_poisoned(task, range(100, task.n))
        poisoned_train = with_gold_poisoned(task, [3])

        np.testing.assert_array_equal(
            rn.ranker_keys("R1", poisoned_test, samples, train_w), base
        )
        assert (rn.ranker_keys("R1", poisoned_train, samples, train_w) != base).any()

    def test_the_all_rows_fit_is_the_unreachable_reference(self):
        task, samples, _ = planted("none")
        train_w, _ = one_split(task, range(100))

        r1 = rn.ranker_keys("R1", task, samples, train_w)
        e8 = rn.ranker_keys("E8", task, samples, train_w)

        assert (r1 != e8).any()  # it saw the test labels too

    def test_misaligned_samples_are_refused(self):
        task, samples, _ = planted("none")
        shifted = dataclasses.replace(samples, ids=samples.ids[::-1])

        with pytest.raises(ValueError, match="same rows"):
            rn.ranker_keys("R2", task, shifted, np.ones((1, task.n)))


# ── aggregation and the bootstrap ────────────────────────────────────────


def hand_ev(precision, base, other=None):
    """A hand-made `evaluate` result with two splits."""
    one = {
        "P": np.array(precision, float),
        "b": np.array(base, float),
        "recall": np.array([0.5, 0.5]),
        "k": np.array([10, 10]),
    }
    ev = {"R1": {PRIMARY: one}}
    if other is not None:
        ev["R2"] = {PRIMARY: {**one, "P": np.array(other, float)}}
    return ev


class TestAggregate:
    def test_lift_is_the_ratio_of_means_not_the_mean_of_ratios(self):
        # (P, b) = (0.5, 0.1) and (0.5, 0.4): the mean of ratios is 3.125.
        got = rn.aggregate(hand_ev([0.5, 0.5], [0.1, 0.4]))

        assert got["R1"][PRIMARY]["L"] == pytest.approx(0.5 / 0.25)
        assert got["R1"][PRIMARY]["L"] != pytest.approx(3.125)

    def test_gain_is_in_points_against_r1_and_rows_per_error_is_one_over_precision(
        self,
    ):
        got = rn.aggregate(hand_ev([0.5, 0.5], [0.1, 0.4], other=[0.7, 0.7]))

        r2 = got["R2"][PRIMARY]
        assert r2["G"] == pytest.approx(20.0)
        assert got["R1"][PRIMARY]["G"] == 0.0
        assert r2["per100"] == pytest.approx(70.0)
        assert r2["rows_per_error"] == pytest.approx(1 / 0.7)

    def test_blocks_of_splits_are_reduced_separately(self):
        ev = hand_ev([0.5, 0.5], [0.1, 0.4])
        for m in ev["R1"].values():
            for key in ("P", "b", "recall", "k"):
                m[key] = np.tile(m[key], 2)
        m = ev["R1"][PRIMARY]
        m["P"] = np.array([0.5, 0.5, 1.0, 1.0])

        red = rn.reduce_splits(ev, groups=2)

        np.testing.assert_allclose(red["R1"][PRIMARY]["P"], [0.5, 1.0])
        np.testing.assert_allclose(red["R1"][PRIMARY]["L"], [2.0, 4.0])


class TestBootstrap:
    def test_a_ranker_equal_to_r1_has_gain_exactly_zero_in_every_draw(self):
        # Paired weights: same splits, same keys, so the difference is exactly 0.
        task, samples, _ = planted("none")
        prereg = small(rankers={**rn.PREREG_REVIEW["rankers"], "R2": ["r1"]})

        draws = rn.bootstrap(task, samples, 8, prereg)

        for f in FRACS:
            assert len(draws["R2"][f]["G"]) == 8
            assert (draws["R2"][f]["G"] == 0.0).all()

    def test_the_lower_bound_is_the_0125_quantile_not_the_one_for_three_tests(self):
        draws = np.arange(1001.0)

        assert rn.effective_prereg()["n_confirmatory"] == 4
        assert rn.lower_bound(draws) == pytest.approx(np.quantile(draws, 0.0125))
        assert rn.lower_bound(draws) == pytest.approx(12.5)
        assert rn.lower_bound(draws) != pytest.approx(np.quantile(draws, 0.05 / 3))

    def test_it_returns_l_and_g_for_r2_and_r4_and_the_paired_difference_at_every_k(
        self,
    ):
        task, samples, _ = planted("errors")

        draws = rn.bootstrap(task, samples, 8, small())

        assert set(draws) == {"R2", "R4", "R4-R2"}
        for f in FRACS:
            assert set(draws["R2"][f]) == {"L", "G"}
            assert set(draws["R4-R2"][f]) == {"G"}
            assert draws["R4"][f]["L"].shape == (8,)

    def test_the_draws_are_reproducible_from_the_parents_stream(self):
        task, samples, _ = planted("errors")

        a = rn.bootstrap(task, samples, 8, small())
        b = rn.bootstrap(task, samples, 8, small())

        for f in FRACS:
            np.testing.assert_array_equal(a["R2"][f]["G"], b["R2"][f]["G"])

    def test_r4_equal_to_r2_when_luna_always_agrees_has_a_zero_paired_difference(self):
        task, samples, _ = planted("errors")  # luna repeats haiku: P = 0

        draws = rn.bootstrap(task, samples, 8, small())

        assert (draws["R4-R2"][PRIMARY]["G"] == 0.0).all()


# ── the verdict ──────────────────────────────────────────────────────────

PASSING = {"L": 3.0, "L_lb": 1.5, "G": 20.0, "G_lb": 5.0}


def statuses(r2, r4):
    return {
        "R2": dict(zip(("intent", "faithfulness"), r2, strict=True)),
        "R4": dict(zip(("intent", "faithfulness"), r4, strict=True)),
    }


class TestVerdict:
    def test_a_planted_signal_passes_and_a_signal_without_information_is_killed(self):
        # End to end through run_study. "errors": D marks every error and R1 has
        # nothing to learn, so R2 and R4 pass. "none": D is constant, R2 and R4
        # reduce to R1, so the gain is 0 and the verdict is KILL.
        task, samples, _ = planted("errors")
        good = study(task, samples)["tasks"]["intent"]
        task, samples, _ = planted("none")
        bad = study(task, samples)["tasks"]["intent"]

        assert good["status"] == {"R2": "PASS", "R4": "PASS"}
        assert good["entries"]["R2"]["G"] > 50
        assert bad["status"] == {"R2": "KILL", "R4": "KILL"}
        assert bad["entries"]["R2"]["G"] == pytest.approx(0.0, abs=1e-9)
        assert bad["entries"]["R2"]["L"] < 1.5

    def test_a_dissent_signal_independent_of_the_errors_has_a_mean_gain_near_zero(self):
        # One fixed dataset gives a lucky or unlucky gain (here +/- 10 points);
        # the mean over many datasets is near 0, and that is what must hold.
        prereg = small(splits=200)
        eff = rn.effective_prereg(prereg)
        gains = []
        for seed in range(12):
            task, samples, _ = planted("independent", seed=seed)
            train_w, test_w, _ = replay._SplitCache(task, eff).get(100)
            agg = rn.aggregate(
                rn.evaluate(task, samples, train_w, test_w, ["R1", "R2"], FRACS, prereg)
            )
            gains.append(agg["R2"][PRIMARY]["G"])

        assert abs(np.mean(gains)) < 5.0

    def test_the_study_reports_the_overall_verdict_for_what_it_ran(self):
        task, samples, _ = planted("errors")

        out = study(task, samples)

        assert out["verdict"]["verdict"] == "BUILD_GENERAL(R2)"  # simpler model wins

    def test_general_scoped_kill_and_inconclusive_each_come_out_on_the_right_inputs(
        self,
    ):
        flags = {"R2": {}, "R4": {}}

        general = rn.verdict(statuses(("PASS", "PASS"), ("KILL", "KILL")), flags, set())
        scoped = rn.verdict(statuses(("PASS", "KILL"), ("KILL", "KILL")), flags, set())
        killed = rn.verdict(statuses(("KILL", "KILL"), ("KILL", "KILL")), flags, set())
        mixed = rn.verdict(
            statuses(("KILL", "INCONCLUSIVE"), ("KILL", "KILL")), flags, set()
        )

        assert general["verdict"] == "BUILD_GENERAL(R2)"
        assert scoped["verdict"] == "BUILD_SCOPED(intent, R2)"
        assert killed["verdict"] == "KILL"
        assert mixed["verdict"] == "INCONCLUSIVE"

    def test_a_v123_failure_forces_inconclusive_even_when_the_numbers_pass(self):
        flags = {"R2": {}, "R4": {}}
        assert rn.task_status({**PASSING, "v_failed": True}) == "INCONCLUSIVE"

        both = rn.verdict(
            statuses(("PASS", "PASS"), ("PASS", "PASS")), flags, {"intent"}
        )
        all_fail = rn.verdict(
            statuses(("PASS", "PASS"), ("PASS", "PASS")),
            flags,
            {"intent", "faithfulness"},
        )
        killed_but_failed = rn.verdict(
            statuses(("KILL", "KILL"), ("KILL", "KILL")), flags, {"intent"}
        )

        assert both["statuses"]["R2"]["intent"] == "INCONCLUSIVE"
        assert both["verdict"] == "BUILD_SCOPED(faithfulness, R2)"
        assert all_fail["verdict"] == "INCONCLUSIVE"
        assert killed_but_failed["verdict"] == "INCONCLUSIVE"  # not all four are KILL

    def test_r4_is_chosen_only_with_a_gain_of_5_points_and_a_paired_lower_bound_above_0(
        self,
    ):
        prereg = rn.PREREG_REVIEW
        both = statuses(("PASS", "PASS"), ("PASS", "PASS"))
        flags = {"R2": {}, "R4": {}}

        assert rn.r4_beats_r2(5.0, 0.1, prereg)
        assert not rn.r4_beats_r2(4.99, 1.0, prereg)
        assert not rn.r4_beats_r2(9.0, 0.0, prereg)  # LB must be above 0
        assert not rn.r4_beats_r2(None, 1.0, prereg)
        yes = {"intent": True, "faithfulness": True}
        assert rn.verdict(both, flags, set(), yes)["verdict"] == "BUILD_GENERAL(R4)"
        assert (
            rn.verdict(both, flags, set(), {**yes, "faithfulness": False})["verdict"]
            == "BUILD_GENERAL(R2)"
        )
        assert rn.verdict(both, flags, set())["verdict"] == "BUILD_GENERAL(R2)"

    def test_flags_on_a_passing_pair_are_reported_and_do_not_change_the_verdict(self):
        both = statuses(("PASS", "PASS"), ("KILL", "KILL"))
        flags = {"R2": {"intent": ["SIGN_FLIP"], "faithfulness": []}, "R4": {}}

        out = rn.verdict(both, flags, set())

        assert out["verdict"] == "BUILD_GENERAL(R2)"
        assert out["flags"] == {"R2/intent": ["SIGN_FLIP"], "R2/faithfulness": []}

    def test_boundaries_g_10_passes_and_9_99_does_not_g_5_is_not_kill(self):
        def status(**over):
            return rn.task_status({**PASSING, **over})

        assert status(G=10.0) == "PASS"
        assert status(G=9.99) == "INCONCLUSIVE"
        assert status(G=5.0) == "INCONCLUSIVE"  # KILL needs G < 5
        assert status(G=4.99) == "KILL"
        assert status(L=2.0) == "PASS"
        assert status(L=1.99) == "INCONCLUSIVE"
        assert status(L=1.5) == "INCONCLUSIVE"  # KILL needs L < 1.5
        assert status(L=1.49) == "KILL"
        assert status(L_lb=1.0) == "INCONCLUSIVE"  # strictly above 1
        assert status(G_lb=0.0) == "INCONCLUSIVE"  # strictly above 0
        assert status(G=None) == "INCONCLUSIVE"

    def test_the_four_flags_fire_on_their_own_rules(self):
        base = {
            "G_adjudicated": 12.0,
            "G_by_frac": {"0.05": 20.0, "0.10": 15.0, "0.20": 8.0},
            "ambiguity": False,
            "G_subsets": {"E4": 12.0, "E5": 10.0, "E6": 30.0},
        }

        assert rn.flags(base) == []
        assert rn.flags({**base, "G_adjudicated": 0.0}) == ["SIGN_FLIP"]
        assert rn.flags({**base, "G_by_frac": {**base["G_by_frac"], "0.05": 0.0}}) == [
            "K_FRAGILE"
        ]
        assert rn.flags({**base, "ambiguity": True}) == ["AMBIGUITY"]
        assert rn.flags({**base, "G_subsets": {"E4": 9.99, "E5": 10.0}}) == ["SUBSET"]


# ── mechanism, costs and disclosure ──────────────────────────────────────


def adjudication_pair():
    """Original: rows r0-r4 wrong, r5-r7 right. Adjudicated drops r4 and relabels."""
    gold = np.zeros(8, int)
    cheap = np.where(np.arange(8) < 5, 1, 0)
    original = classification_task(
        gold, {"haiku": cheap, "luna": cheap, **{m: gold for m in replay.MODELS[2:]}}
    )
    keep = [0, 1, 2, 3, 5, 6, 7]
    new_gold = np.array([0, 0, 1, 1, 1, 0, 0])  # r2, r3 now right; r5 now wrong
    adj = classification_task(
        new_gold,
        {
            "haiku": cheap[keep],
            "luna": cheap[keep],
            **{m: new_gold for m in replay.MODELS[2:]},
        },
    )
    adj = dataclasses.replace(
        adj, ids=[original.ids[i] for i in keep], gold_kind="adjudicated"
    )
    return original, adj


class TestNoiseControl:
    def test_the_wrong_rows_split_into_still_wrong_now_right_dropped_and_newly_wrong(
        self,
    ):
        original, adj = adjudication_pair()
        flagged = {
            "r0": 1,
            "r1": 0,
            "r2": 2,
            "r3": 1,
            "r4": 3,
            "r5": 0,
            "r6": 0,
            "r7": 0,
        }

        out = rn.noise_control(original, adj, flagged)
        g = out["groups"]

        assert [g[x]["n"] for x in ("A", "F", "X", "N")] == [2, 2, 1, 1]
        assert [g[x]["flagged"] for x in ("A", "F", "X", "N")] == [1, 2, 1, 0]
        assert g["FX"]["flagged"] == 3
        assert g["FX"]["n"] == 3
        assert g["A"]["wilson"] == addendum1.wilson(1, 2, rn.PREREG_REVIEW["wilson_z"])
        assert out["gap_fx_minus_a"] == pytest.approx(1.0 - 0.5)
        assert out["ambiguity"]  # 0.5 >= 0.20: the flag finds disputed labels

    def test_the_ambiguity_bound_is_exact_at_0_20(self):
        original, adj = adjudication_pair()
        # A: 2 rows; F u X: 3 rows. flag(A) = 1/2; gap 0.20 is not reachable with
        # these sizes, so check both sides: 2/3 - 1/2 = 0.1667 (no), 3/3 - 1/2 (yes).
        near = rn.noise_control(
            original,
            adj,
            {
                "r0": 1,
                "r1": 0,
                "r2": 1,
                "r3": 1,
                "r4": 0,
                **dict.fromkeys(("r5", "r6", "r7"), 0),
            },
        )

        assert near["gap_fx_minus_a"] == pytest.approx(2 / 3 - 1 / 2)
        assert not near["ambiguity"]

    def test_flag_rates_on_wrong_and_right_rows_cover_all_original_rows(self):
        original, adj = adjudication_pair()

        out = rn.noise_control(original, adj, dict.fromkeys(original.ids, 1))

        assert out["groups"]["wrong"]["n"] == 5
        assert out["groups"]["right"]["n"] == 3
        assert out["groups"]["right"]["flagged"] == 3


class TestCostsAndDisclosure:
    def test_unit_costs_and_break_even_follow_the_calls_per_row(self):
        task, samples, _ = planted("errors")
        agg = rn.aggregate(
            rn.evaluate(
                task, samples, *one_split(task, range(100)), ["R1", "R2", "R4"], FRACS
            )
        )

        costs = rn.unit_costs(task)
        be = rn.break_even(task, agg)

        assert costs["R2"] == pytest.approx(3 * 1000 * task.u_flat["haiku"])
        assert costs["R4"] == pytest.approx(
            1000 * (3 * task.u_flat["haiku"] + task.u_flat["luna"])
        )
        assert costs["R1"] == 0.0
        # 10 c / h with c = 3e-4 and h = 0.25: R2 needs a ratio above 1.012.
        assert be["R2"]["needed_ratio"]["0.25"] == pytest.approx(1 + 10 * 3e-4 / 0.25)
        assert be["R2"]["needed_ratio"]["0.01"] == pytest.approx(1 + 10 * 3e-4 / 0.01)

    def test_diff_disclosure_names_each_count_that_does_not_reproduce(self):
        got = {"rows": 10, "wrong": 3, "adj": {"rows": 9, "wrong": 2}}

        assert rn.diff_disclosure(got, got) == {}
        assert rn.diff_disclosure(got, {"rows": 10, "wrong": 4}) == {"wrong": (3, 4)}
        assert rn.diff_disclosure(got, {"adj": {"wrong": 1, "rows": 9}}) == {
            "adj.wrong": (2, 1)
        }

    def test_disclosure_counts_are_recomputed_from_the_arrays(self):
        task, samples, wrong = planted("errors")
        both = (task, samples)

        got = rn.disclosure_counts(both, both, small())

        assert got["rows"] == N
        assert got["wrong"] == wrong.sum()
        assert got["dissent"] == {0: N - wrong.sum(), 1: 0, 2: 0, 3: wrong.sum()}
        assert got["flag_wrong"] == wrong.sum()
        assert got["flag_right"] == 0
        assert got["pair"] == 0
        assert got["dissent_and_pair"] == 0
        assert got["test_rows"] == N - 100
        assert sum(got["value_counts"]) == N


# ── the study end to end, and its outputs ────────────────────────────────


class TestOutputs:
    def test_the_results_embed_the_actual_hash_values(self, tmp_path):
        task, samples, _ = planted("errors")
        checks = rn.Checks(data_sha256={"intent/dev.jsonl": "ab" * 32})
        results = study(task, samples, checks=checks)

        md, js = rn.write_results(results, tmp_path)
        meta = json.loads(js.read_text())["meta"]

        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        planner = _ROOT / "benchmarks" / "planner"
        assert meta["prereg_review_sha256"] == replay.prereg_hash(small())
        assert meta["design_sha256"] == sha(rn.DESIGN)
        assert meta["code_sha256"] == sha(_MODULE)
        assert meta["parent_design_sha256"] == sha(planner / "DESIGN.md")
        assert meta["parent_prereg_sha256"] == replay.prereg_hash()
        assert meta["addendum1_design_sha256"] == sha(planner / "DESIGN-addendum-1.md")
        assert meta["addendum1_prereg_sha256"] == addendum1.prereg_addendum_hash()
        assert meta["data_sha256"] == {"intent/dev.jsonl": "ab" * 32}
        assert set(meta["git"]) == {"head", "dirty"}
        assert (
            meta["prereg_review"]["design_sha256"] == rn.PREREG_REVIEW["design_sha256"]
        )
        text = md.read_text()
        for key in ("design_sha256", "code_sha256", "parent_design_sha256"):
            assert meta[key] in text
        assert "QUICK" in text
        assert rn.SELECTION_RULE in text

    def test_the_markdown_reports_every_ranker_and_the_verdict(self, tmp_path):
        task, samples, _ = planted("errors")

        md, _ = rn.write_results(study(task, samples), tmp_path)
        text = md.read_text()

        assert "BUILD_GENERAL(R2)" in text
        for name in ("R0", "R1", "R2", "R4", "E1", "E8"):
            assert f"| {name} |" in text
        assert "Break-even" in text
        assert "Mechanism" not in text  # no adjudicated file

    def test_the_preregistered_design_hash_is_the_files_hash(self):
        digest = hashlib.sha256(rn.DESIGN.read_bytes()).hexdigest()

        assert digest == rn.PREREG_REVIEW["design_sha256"]
        assert digest.startswith("57e8962a0173")


# ── --check on a golden-layout fixture ───────────────────────────────────

STEMS = rn.PREREG_REVIEW["sample_files"]


def write_sample_files(data: Path, flips=(5, 4, 3)) -> None:
    """Haiku's intent answers three times, sample j flipping row flips[j]."""
    rows = replay.read_jsonl(data / "intent" / "dev.jsonl", data)
    original = {
        r["id"]: r["answer"]
        for r in replay.read_jsonl(data / "labels" / "intent" / "haiku.jsonl", data)
    }
    cost = replay.read_json(data / "labels" / "intent" / "haiku.cost.json", data)
    implied = addendum1.implied_original_output_tokens(cost, addendum1.PREREG_ADDENDUM)
    for stem, flip in zip(STEMS, flips, strict=True):
        answers = dict(original)
        rid = rows[flip]["id"]
        answers[rid] = "lost_card" if original[rid] != "lost_card" else "card_arrival"
        write_jsonl(
            data / "labels" / "intent" / f"{stem}.jsonl",
            [{"id": r["id"], "answer": answers[r["id"]]} for r in rows],
        )
        (data / "labels" / "intent" / f"{stem}.cost.json").write_text(
            json.dumps(
                {
                    "route": "openrouter",
                    "model": addendum1.PREREG_ADDENDUM["model_id"],
                    "cost": 0.01,
                    "tokens": 5000,
                    "rows": len(rows),
                    "input_tokens": 4000,
                    "output_tokens": round(implied * len(rows)),
                }
            )
        )


def fixture_prereg(data: Path) -> dict:
    """A PREREG_REVIEW for the 6-row layout: intent only, 3 labelled rows.

    The data hashes and the disclosure are frozen from the fixture itself, the
    way the real ones were frozen from the real data.
    """
    rels = [
        "intent/dev.jsonl",
        "intent/dev-adjudicated.jsonl",
        "intent/labels.json",
        *(f"labels/intent/{m}.jsonl" for m in ("haiku", "luna", *STEMS)),
        *(f"labels/intent/{m}.cost.json" for m in ("haiku", "luna")),
    ]
    prereg = {
        **rn.PREREG_REVIEW,
        "tasks": ["intent"],
        "n_labelled": 3,
        "splits": 20,
        "bootstrap": 8,
        "bootstrap_quick": 8,
        "bootstrap_splits_per_draw": 2,
        "data_sha256": {r: addendum1.sha256_file(data / r) for r in rels},
    }
    tasks = rn.load_tasks(data, prereg)
    counts = rn.disclosure_counts(
        tasks["intent"]["original"], tasks["intent"]["adjudicated"], prereg
    )
    return {**prereg, "disclosure": {"intent": counts}}


def fixture_kwargs(data: Path, prereg: dict) -> dict:
    lenient = {
        **addendum1.PREREG_ADDENDUM,
        "run_tasks": ["intent"],
        "disclosure": {},
        "v": {
            **addendum1.PREREG_ADDENDUM["v"],
            "v1": {"intent": {"metric": "accuracy", "center": 0.67, "tol": 0.5}},
            "v3_tol_pts": 100.0,
        },
    }
    tiny = {**replay.PREREG, "splits": 20, "n_labelled": 3, "n_confirmatory": 2}
    return {
        "data_dir": data,
        "prereg": prereg,
        "integrity_prereg": {**replay.PREREG, "strong": STRONG},
        "sample_args": {"addendum": lenient, "prereg": tiny, "reference": None},
    }


@pytest.fixture
def layout(tmp_path):
    data = make_layout(tmp_path)
    write_sample_files(data)
    return data


class TestCheck:
    def test_a_consistent_fixture_passes_and_prints_test_rows_k_and_r0(self, layout):
        prereg = fixture_prereg(layout)

        rep = rn.check(**fixture_kwargs(layout, prereg))

        assert rep.hard_ok, rep.problems
        text = "\n".join(rep.lines)
        assert "test rows per split [3.0]" in text
        assert "R0 lift 1 to within" in text
        assert rep.splits["intent"]["original"]["test_rows"] == 3
        assert rep.splits["intent"]["original"]["r0_max_dev"] <= 1e-12
        assert len(rep.data_sha256) == len(prereg["data_sha256"])

    def test_check_mode_never_runs_a_ranker_or_the_bootstrap(self, layout, monkeypatch):
        # evaluate, ranker_keys, aggregate and bootstrap all raise; --check passes.
        def boom(*_a, **_k):
            raise AssertionError("check mode computed a result")

        prereg = fixture_prereg(layout)
        for name in ("evaluate", "ranker_keys", "aggregate", "bootstrap", "run_study"):
            monkeypatch.setattr(rn, name, boom)

        rep = rn.check(**fixture_kwargs(layout, prereg))

        assert rep.hard_ok, rep.problems

    def test_one_changed_answer_makes_check_exit_2_on_the_disclosure(
        self, layout, capsys
    ):
        frozen = fixture_prereg(layout)
        path = layout / "labels" / "intent" / "haiku-s2.jsonl"
        rows = replay.read_jsonl(path, layout)
        rows[0]["answer"] = "top_up" if rows[0]["answer"] != "top_up" else "lost_card"
        write_jsonl(path, rows)
        # Hashes re-frozen after the edit, so only the disclosure can fail.
        refrozen = {
            **frozen,
            "data_sha256": {
                r: addendum1.sha256_file(layout / r) for r in frozen["data_sha256"]
            },
        }

        code = rn.main(["--check"], **fixture_kwargs(layout, refrozen))

        out = capsys.readouterr().out
        assert code == 2
        assert "differ from the disclosure" in out
        assert "sha256 " not in "".join(
            line for line in out.splitlines() if line.startswith("FAIL")
        )

    def test_a_data_file_that_changed_fails_its_hash(self, layout, capsys):
        prereg = fixture_prereg(layout)
        path = layout / "labels" / "intent" / "haiku-s3.jsonl"
        rows = replay.read_jsonl(path, layout)
        rows[1]["answer"] = "top_up" if rows[1]["answer"] != "top_up" else "lost_card"
        write_jsonl(path, rows)

        code = rn.main(["--check"], **fixture_kwargs(layout, prereg))

        assert code == 2
        assert "labels/intent/haiku-s3.jsonl: sha256" in capsys.readouterr().out

    def test_a_parent_hash_mismatch_fails(self, layout):
        prereg = fixture_prereg(layout)
        bad = {
            **prereg,
            "parents": {
                **prereg["parents"],
                "replay": {**prereg["parents"]["replay"], "design_sha256": "0" * 64},
            },
        }

        rep = rn.check(**fixture_kwargs(layout, bad))

        assert any("replay design hash" in p for p in rep.problems)

    def test_the_intact_fixture_exits_0(self, layout, capsys):
        prereg = fixture_prereg(layout)

        code = rn.main(["--check"], **fixture_kwargs(layout, prereg))

        assert code == 0
        assert "CHECK OK" in capsys.readouterr().out

    def test_a_missing_sample_file_is_a_hard_problem(self, layout):
        prereg = fixture_prereg(layout)
        (layout / "labels" / "intent" / "haiku-s3.jsonl").unlink()

        rep = rn.check(**fixture_kwargs(layout, prereg))

        assert not rep.hard_ok

    def test_the_held_out_file_is_never_read_even_when_it_is_unreadable(self, layout):
        # The fixture's held-out file is garbage; make it unreadable too.
        held = layout / "intent" / "holdout.jsonl"
        held.chmod(0)
        try:
            rep = rn.check(**fixture_kwargs(layout, fixture_prereg(layout)))
        finally:
            held.chmod(0o644)

        assert rep.hard_ok, rep.problems


class TestSourceHygiene:
    def test_the_held_out_word_and_raw_file_reads_never_appear_in_the_module(self):
        for path in (_MODULE.parent).glob("*.py"):
            text = path.read_text()

            assert "holdout" not in text.lower()
            for raw in ("open(", "read_text(", "json.load("):
                assert raw not in text, f"{path.name} reads files itself: {raw}"


# ── the bootstrap stream, the wiring and the remaining boundaries ────────


class TestBootstrapStream:
    def test_the_weights_are_the_parents_stream_purpose_1_original_gold_chunk_1000(
        self, monkeypatch
    ):
        # Pins the stream: purpose 1, gold "original", n_l = 100, 20 splits per
        # draw and chunk 1000. Any other stream gives different weights.
        task, samples, _ = planted("errors")
        real = replay.bootstrap_batches
        seen: list = []

        def spy(rng, n, n_l, n_draws, per_draw, chunk):
            seen.append((n, n_l, n_draws, per_draw, chunk))
            for batch in real(rng, n, n_l, n_draws, per_draw, chunk):
                seen.append(batch)
                yield batch

        monkeypatch.setattr(replay, "bootstrap_batches", spy)
        rn.bootstrap(task, samples, 4)
        monkeypatch.setattr(replay, "bootstrap_batches", real)
        want = list(
            real(
                replay.split_rng(replay.PREREG, task.name, "original", 1, 100),
                task.n,
                100,
                4,
                20,
                1000,
            )
        )

        assert seen[0] == (task.n, 100, 4, 20, 1000)
        assert len(seen) == 1 + len(want)
        for got, ref in zip(seen[1:], want, strict=True):
            np.testing.assert_array_equal(got[0], ref[0])
            np.testing.assert_array_equal(got[1], ref[1])
            assert got[2] == ref[2]


def reversed_gold(task, samples):
    """The same rows with every right/wrong flipped: an adjudication that disagrees."""
    flipped = with_gold_poisoned(task, range(task.n))
    return dataclasses.replace(flipped, gold_kind="adjudicated"), samples


class TestRunTaskWiring:
    def test_the_entry_reads_the_right_bounds_and_the_adjudicated_gold(self):
        task, samples, _ = planted("errors")

        res = study(task, samples, adjudicated=reversed_gold(task, samples))["tasks"][
            "intent"
        ]
        entry = res["entries"]["R2"]
        draws = rn.bootstrap(task, samples, 16, small())
        adj = res["golds"]["adjudicated"]["rankers"]["R2"][PRIMARY]["G"]

        assert entry["L_lb"] == pytest.approx(
            rn.lower_bound(draws["R2"][PRIMARY]["L"], small())
        )
        assert entry["G_lb"] == pytest.approx(
            rn.lower_bound(draws["R2"][PRIMARY]["G"], small())
        )
        assert entry["L_lb"] != pytest.approx(entry["G_lb"])
        assert entry["G_adjudicated"] == adj
        assert entry["G"] > 50
        assert adj < 0  # D marks rows the adjudicated gold calls right
        assert "SIGN_FLIP" in res["flags"]["R2"]
        assert entry["G_by_frac"] == {
            f: res["golds"]["original"]["rankers"]["R2"][f]["G"] for f in FRACS
        }

    def test_p_marking_the_errors_gives_r4_a_positive_gain_over_r2_with_the_sign_right(
        self,
    ):
        # Luna is perfect, D carries nothing: R2 reduces to R1, R4 finds every
        # error. R4 - R2 must be large and positive, in the point estimate, the
        # paired lower bound and the draws; a swapped subtraction flips all three.
        task, samples, _ = planted("none", luna="right")

        out = study(task, samples)
        res = out["tasks"]["intent"]
        draws = rn.bootstrap(task, samples, 16, small())

        assert res["r4_over_r2"]["gain"] > 50
        assert res["r4_over_r2"]["lb"] > 0
        assert draws["R4-R2"][PRIMARY]["G"].mean() > 50
        assert res["r4_beats_r2"]
        assert res["status"] == {"R2": "KILL", "R4": "PASS"}
        assert out["verdict"]["verdict"] == "BUILD_GENERAL(R4)"

    def test_the_r4_over_r2_gain_is_the_difference_of_the_two_not_r4_over_r1(self):
        # D and P both mark every error: R2 and R4 both find them all, so R4 - R2
        # is about 0 while each beats R1 by a lot. A gain read as R4 - R1 is large.
        task, samples, _ = planted("errors", luna="right")

        res = study(task, samples)["tasks"]["intent"]
        g2, g4 = (res["entries"][r]["G"] for r in ("R2", "R4"))

        assert g2 > 50
        assert g4 > 50
        assert res["r4_over_r2"]["gain"] == pytest.approx(g4 - g2)
        assert abs(res["r4_over_r2"]["gain"]) < 5
        assert not res["r4_beats_r2"]

    def test_the_subset_flag_reads_only_e4_to_e6(self):
        ranker = {f: {"L": 3.0, "G": 20.0} for f in FRACS}
        low = {f: {"L": 1.0, "G": 0.0} for f in FRACS}
        res = {
            "golds": {
                "original": {
                    "rankers": {
                        "R2": ranker,
                        **dict.fromkeys(("E4", "E5", "E6"), ranker),
                        "E7": low,  # one sample only: not a SUBSET candidate
                    }
                }
            }
        }
        boot = {"R2": {f: {"L": {"lb": 2.0}, "G": {"lb": 3.0}} for f in FRACS}}

        entry = rn._entry(res, boot, "R2", False, rn.PREREG_REVIEW)

        assert set(entry["G_subsets"]) == {"E4", "E5", "E6"}
        assert "SUBSET" not in rn.flags(entry)


class TestBoundaries:
    def test_k_and_the_base_rate_use_the_weight_sum_not_the_row_count(self):
        task, samples, wrong = planted("errors")
        train_w, test_w = one_split(task, range(100))
        test_w[0, 100:110] = 2.0  # ten rows with weight 2: W = 141, 131 rows

        ev = rn.evaluate(task, samples, train_w, test_w, ["R0"], FRACS)
        got = ev["R0"][PRIMARY]

        assert test_w.sum() == 141
        assert got["k"].tolist() == [14] == [rn.k_at(PRIMARY, 141)]
        assert ev["R0"]["0.05"]["k"].tolist() == [rn.k_at("0.05", 141)]
        base = (test_w[0] * wrong).sum() / 141
        assert got["b"][0] == pytest.approx(base)
        assert got["P"][0] == pytest.approx(base)  # R0: every row ties

    def test_a_zero_gain_at_the_primary_k_alone_is_not_k_fragile(self):
        base = {"G_by_frac": {"0.05": 5.0, "0.10": -3.0, "0.20": 5.0}}

        assert "K_FRAGILE" not in rn.flags(base)
        assert rn.flags({"G_by_frac": {"0.05": 5.0, "0.10": 30.0, "0.20": 0.0}}) == [
            "K_FRAGILE"
        ]

    def test_ambiguity_fires_at_a_gap_of_exactly_one_fifth(self):
        # A: 5 rows, none flagged (0/5). F u X: 5 rows, one flagged (1/5).
        # The gap is exactly 0.20 and the rule is >=, so it fires.
        gold = np.zeros(10, int)
        wrong = {"haiku": np.ones(10, int), "luna": np.ones(10, int)}
        original = classification_task(
            gold, {**wrong, **{m: gold for m in replay.MODELS[2:]}}
        )
        keep = list(range(8))  # r8, r9 dropped (X); r5-r7 now right (F)
        new_gold = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        adjudicated = classification_task(
            new_gold,
            {
                "haiku": np.ones(8, int),
                "luna": np.ones(8, int),
                **{m: new_gold for m in replay.MODELS[2:]},
            },
        )
        adjudicated = dataclasses.replace(
            adjudicated, ids=[original.ids[i] for i in keep], gold_kind="adjudicated"
        )
        d = {rid: 0 for rid in original.ids}
        d["r8"] = 2  # the one flagged row of F u X

        out = rn.noise_control(original, adjudicated, d)

        assert out["groups"]["A"]["n"] == 5
        assert out["groups"]["A"]["flagged"] == 0
        assert out["groups"]["FX"]["n"] == 5
        assert out["groups"]["FX"]["flagged"] == 1
        assert out["gap_fx_minus_a"] == pytest.approx(0.2)
        assert out["ambiguity"]

    def test_a_nan_recall_is_skipped_in_the_mean_and_only_an_all_nan_block_is_nan(self):
        ev = {
            "R1": {
                PRIMARY: {
                    "P": np.ones(4),
                    "b": np.ones(4),
                    "recall": np.array([np.nan, np.nan, 0.2, 0.4]),
                    "k": np.ones(4),
                }
            }
        }

        one = rn.reduce_splits(ev, 1)["R1"][PRIMARY]["recall"]
        two = rn.reduce_splits(ev, 2)["R1"][PRIMARY]["recall"]

        assert one[0] == pytest.approx(0.3)
        assert np.isnan(two[0])
        assert two[1] == pytest.approx(0.3)

    def test_the_verdict_carries_the_selection_rule_in_one_line(self):
        out = rn.verdict(statuses(("PASS", "PASS"), ("KILL", "KILL")), {}, set())

        assert out["selection_rule"] == (
            "R2 unless R4 beats R2 by >= 5 points with paired LB > 0 on every task in scope"
        )

    def test_thresholds_are_compared_exactly_with_no_tolerance(self):
        # DESIGN section 5 states exact thresholds; a value one ulp below 10 fails.
        assert rn.task_status({**PASSING, "G": 10.0 - 1e-12}) == "INCONCLUSIVE"
        assert rn.task_status({**PASSING, "G": 10.0}) == "PASS"


# ── the design hash, the data hashes and the run guards ──────────────────


class TestLocks:
    def test_a_one_byte_edit_to_the_design_makes_check_fail(self, layout, tmp_path):
        prereg = fixture_prereg(layout)
        edited = tmp_path / "DESIGN.md"
        raw = bytearray(rn.DESIGN.read_bytes())
        raw[10] ^= 1
        edited.write_bytes(bytes(raw))

        rep = rn.check(**fixture_kwargs(layout, prereg), design_path=edited)
        intact = rn.check(**fixture_kwargs(layout, prereg), design_path=rn.DESIGN)

        assert any("design file hash" in p for p in rep.problems)
        assert intact.hard_ok, intact.problems

    def test_a_missing_design_file_fails_check(self, layout, tmp_path):
        rep = rn.check(
            **fixture_kwargs(layout, fixture_prereg(layout)),
            design_path=tmp_path / "none.md",
        )

        assert any("not found" in p for p in rep.problems)

    def test_the_labels_and_cost_files_are_hash_locked_too(self, layout):
        prereg = fixture_prereg(layout)
        names = [
            "intent/labels.json",
            "labels/intent/haiku.cost.json",
            "labels/intent/luna.cost.json",
        ]

        assert set(names) <= set(prereg["data_sha256"])
        assert set(names) <= set(rn.PREREG_REVIEW["data_sha256"])
        assert {n.replace("intent", "faithfulness") for n in names[1:]} <= set(
            rn.PREREG_REVIEW["data_sha256"]
        )
        assert len(rn.PREREG_REVIEW["data_sha256"]) == 19
        path = layout / "labels" / "intent" / "luna.cost.json"
        cost = json.loads(path.read_text())
        cost["note"] = "edited"  # integrity does not notice; the hash does
        path.write_text(json.dumps(cost))

        rep = rn.check(**fixture_kwargs(layout, prereg))

        assert any("luna.cost.json: sha256" in p for p in rep.problems)

    def test_the_dirty_flag_sees_the_studys_own_files_but_not_its_results(
        self, tmp_path, monkeypatch
    ):
        import subprocess

        def git(*args):
            subprocess.run(
                ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
                cwd=tmp_path,
                check=True,
                capture_output=True,
            )

        for rel in (
            "benchmarks/review_next/review_next.py",
            "benchmarks/review_next/results/x.json",
            "tests/unit/test_review_next.py",
        ):
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / rel).write_text("a\n")
        git("init", "-q")
        git("add", "-A")
        git("commit", "-q", "-m", "init")
        monkeypatch.setattr(rn, "REPO", tmp_path)
        monkeypatch.setattr(replay, "_git_state", lambda: {"head": "h", "dirty": False})

        assert rn._git_state() == {"head": "h", "dirty": False}
        (tmp_path / "benchmarks/review_next/results/x.json").write_text("changed\n")
        assert not rn._git_state()["dirty"]  # the results directory is excluded
        (tmp_path / "tests/unit/test_review_next.py").write_text("changed\n")
        assert rn._git_state()["dirty"]
        git("checkout", "--", "tests/unit/test_review_next.py")
        (tmp_path / "benchmarks/review_next/review_next.py").write_text("changed\n")
        assert rn._git_state()["dirty"]
        monkeypatch.setattr(replay, "_git_state", lambda: {"head": "h", "dirty": True})
        git("checkout", "--", "benchmarks/review_next/review_next.py")
        assert rn._git_state()["dirty"]  # the parent's flag is kept


CLEAN = {"head": "abc", "dirty": False}


@pytest.fixture
def guarded(layout, tmp_path, monkeypatch):
    """The fixture layout, a scratch results root and a clean tree."""
    monkeypatch.setattr(rn, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(rn, "_git_state", lambda: CLEAN)
    return fixture_kwargs(layout, fixture_prereg(layout))


class TestRunGuards:
    def test_quick_is_refused_before_anything_runs_unless_out_or_a_result_exists(
        self, guarded, tmp_path, monkeypatch, capsys
    ):
        def boom(*_a, **_k):
            raise AssertionError("a refused run computed something")

        with monkeypatch.context() as m:
            m.setattr(rn, "check", boom)
            m.setattr(rn, "run_study", boom)
            code = rn.main(["--quick"], **guarded)
        out = capsys.readouterr().out

        assert code == 2
        assert "REFUSED" in out
        assert "reveals the outcome" in out
        assert not (tmp_path / "results").exists()

        scratch = tmp_path / "scratch"
        assert rn.main(["--quick", "--out", str(scratch)], **guarded) == 0
        assert (scratch / "review-next.json").exists()

        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "review-next.json").write_text("{}")
        assert rn.main(["--quick"], **guarded) == 0
        assert (tmp_path / "results" / "quick" / "review-next.json").exists()

    def test_check_is_never_refused(self, guarded, monkeypatch):
        monkeypatch.setattr(rn, "_git_state", lambda: {"head": "x", "dirty": True})

        assert rn.main(["--check"], **guarded) == 0

    def test_a_full_run_on_a_dirty_tree_exits_2_and_a_clean_one_writes_the_result(
        self, guarded, tmp_path, monkeypatch, capsys
    ):
        out_dir = tmp_path / "full"
        monkeypatch.setattr(rn, "_git_state", lambda: {"head": "x", "dirty": True})

        dirty = rn.main(["--out", str(out_dir)], **guarded)

        assert dirty == 2
        assert "uncommitted changes" in capsys.readouterr().out
        assert not out_dir.exists()

        monkeypatch.setattr(rn, "_git_state", lambda: CLEAN)
        assert rn.main(["--out", str(out_dir)], **guarded) == 0
        meta = json.loads((out_dir / "review-next.json").read_text())["meta"]
        assert meta["git"] == CLEAN
        assert meta["quick"] is False

    def test_a_second_full_run_may_not_overwrite_the_result(
        self, guarded, tmp_path, capsys
    ):
        out_dir = tmp_path / "full"
        assert rn.main(["--out", str(out_dir)], **guarded) == 0
        first = (out_dir / "review-next.json").read_text()

        code = rn.main(["--out", str(out_dir)], **guarded)

        assert code == 2
        assert "would overwrite" in capsys.readouterr().out
        assert (out_dir / "review-next.json").read_text() == first
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "review-next.json").write_text("{}")
        assert rn.main([], **guarded) == 2
