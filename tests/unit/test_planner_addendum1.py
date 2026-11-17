"""benchmarks/planner/addendum1.py and label.py's --sample options, on synthetic data.

Addendum 1 is pre-registered: nobody may look at a policy number on the real
rows before the code is reviewed, and the extra Haiku samples do not exist yet.
Every test therefore builds a task or a golden-layout fixture whose right answer
follows from how it is constructed. Nothing reads the real data directory,
nothing calls a model: the label.py tests replace the pipeline with a recorder.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from test_planner_replay import (  # noqa: E402
    INTENT_NAMES,
    ROW_COST,
    classification_task,
    make_layout,
    small_prereg,
    write_jsonl,
)

_ROOT = _HERE.parents[1]
_ADDENDUM = _ROOT / "benchmarks" / "planner" / "addendum1.py"
_spec = importlib.util.spec_from_file_location("planner_addendum1", _ADDENDUM)
add = importlib.util.module_from_spec(_spec)
sys.modules["planner_addendum1"] = add
_spec.loader.exec_module(add)
replay = add.replay

_LABEL = _ROOT / "benchmarks" / "golden" / "label.py"
_lspec = importlib.util.spec_from_file_location("golden_label", _LABEL)
label = importlib.util.module_from_spec(_lspec)
sys.modules["golden_label"] = label
_lspec.loader.exec_module(label)

# ── synthetic tasks and samples ──────────────────────────────────────────

N = 231


def gold_codes(n=N, seed=0):
    return np.random.default_rng(seed).integers(0, 10, n)


def sample_set(task, gold, extra, valid=None):
    """A SampleSet: column 0 is Haiku's answer on `task`, then the `extra` columns.

    `extra` is (n, k) answer codes; `valid` defaults to all valid.
    """
    codes = np.column_stack([task.value["haiku"][:, 0], np.asarray(extra)])
    return add.SampleSet(
        list(task.ids),
        codes,
        np.ones(codes.shape, bool) if valid is None else valid,
        codes == np.asarray(gold)[:, None],
    )


def planted_task(luna="same", disagree="all", n=N, wrong_every=4):
    """Haiku is wrong on every `wrong_every`-th row; the strong models are perfect.

    The samples disagree with Haiku on the wrong rows `disagree` selects ("all",
    "most" = all but one in four, or "none"); on the other rows all four answers
    are equal. Luna repeats Haiku ("same"), is perfect ("right"), or is right on
    Haiku's wrong rows but wrong on every sixth row ("noisy": a realistic cheap
    model, so that it does not dominate the hull).
    """
    idx = np.arange(n)
    gold = gold_codes(n)
    wrong = idx % wrong_every == 0
    cheap = np.where(wrong, gold + 12, gold)
    luna_values = {
        "same": cheap,
        "right": gold,
        "noisy": np.where(idx % 6 == 2, gold + 20, gold),
    }[luna]
    task = classification_task(
        gold,
        {"haiku": cheap, "luna": luna_values, **{m: gold for m in replay.MODELS[2:]}},
    )
    picked = {
        "all": wrong,
        "most": wrong & (idx % 16 != 12),
        "none": np.zeros(n, bool),
    }[disagree]
    extra = np.column_stack([np.where(picked, 40 + j, cheap) for j in range(3)])
    return task, sample_set(task, gold, extra), wrong, picked


def study(task, samples, checks=None, adjudicated=None, **over):
    tasks = {
        task.name: {
            "original": (task, samples),
            "adjudicated": adjudicated
            or (dataclasses.replace(task, same_as_original=True), samples),
        }
    }
    return add.run_study(
        tasks,
        checks or add.Checks(),
        add.PREREG_ADDENDUM,
        small_prereg(n_confirmatory=2, **over),
        quick=True,
        log=lambda *_: None,
    )


# ── the preregistration ──────────────────────────────────────────────────


class TestPreregistration:
    def test_constants_are_the_designs(self):
        p = add.PREREG_ADDENDUM

        assert p["k"] == 3
        assert p["sample_files"] == ["haiku-s1", "haiku-s2", "haiku-s3"]
        assert p["replacement_file"] == "haiku-s4"
        assert p["route"] == "openrouter"
        assert p["model_id"] == "openrouter/anthropic/claude-haiku-5-5"
        assert p["prices_per_1k_tokens"] == {"input": "0.0001", "output": "0.0005"}
        assert p["n_confirmatory"] == 2
        assert p["verdict"]["beat_c4_pts"] == 2.0
        assert p["hull_cost_multiple"] == 2.0 == replay.PREREG["hull_cost_multiple"]
        assert p["strong"] == {"intent": "opus", "faithfulness": "sol"}
        assert p["v"]["v1"]["intent"] == {
            "metric": "accuracy",
            "center": 0.81,
            "tol": 0.05,
        }
        assert p["v"]["v1"]["faithfulness"] == {
            "metric": "balanced",
            "center": 0.82,
            "tol": 0.06,
        }
        assert p["v"]["v2"] == {"low": 0.5, "high": 2.0}
        assert p["v"]["v3_tol_pts"] == 5.0
        assert p["v"]["v4_max_bad_pct"] == 3.0
        assert p["diagnostic"]["useful_r_blind_min"] == 0.40
        assert p["diagnostic"]["useful_ff_max"] == 0.10
        assert p["diagnostic"]["not_useful_r_blind_max"] == 0.20
        assert p["noise"] == {"flag_a_min": 0.40, "ambiguity_gap_min": 0.20}

    def test_it_keeps_the_parents_task_order_so_the_split_streams_are_identical(self):
        prereg = add.effective_prereg()

        assert add.PREREG_ADDENDUM["tasks"] == replay.PREREG["tasks"]
        for task in ("intent", "faithfulness"):
            for purpose in (0, 1):
                mine = replay.split_rng(prereg, task, "original", purpose, 100)
                theirs = replay.split_rng(replay.PREREG, task, "original", purpose, 100)
                assert mine.random(5).tolist() == theirs.random(5).tolist()

    def test_the_parent_is_unchanged(self):
        parent = add.PREREG_ADDENDUM["parent"]

        assert replay.prereg_hash() == parent["prereg_sha256"]
        if replay.DESIGN.exists():
            assert replay.design_hash() == parent["design_sha256"]
        assert add.effective_prereg()["n_confirmatory"] == 2
        assert replay.PREREG["n_confirmatory"] == 3

    def test_the_hash_is_the_sha256_of_the_sorted_json(self):
        want = hashlib.sha256(
            json.dumps(add.PREREG_ADDENDUM, sort_keys=True).encode()
        ).hexdigest()

        assert add.prereg_addendum_hash() == want

    def test_label_py_uses_the_route_and_prices_the_design_names(self):
        p = add.PREREG_ADDENDUM

        assert p["model_id"] == label.OPENROUTER_MODEL
        assert {
            "input_cost_per_1k_tokens": Decimal(p["prices_per_1k_tokens"]["input"]),
            "output_cost_per_1k_tokens": Decimal(p["prices_per_1k_tokens"]["output"]),
        } == label.OPENROUTER_PRICES
        # Per token: the prices V2 divides by.
        assert float(
            label.OPENROUTER_PRICES["input_cost_per_1k_tokens"]
        ) / 1000 == pytest.approx(p["prices_per_token"]["input"])
        assert float(
            label.OPENROUTER_PRICES["output_cost_per_1k_tokens"]
        ) / 1000 == pytest.approx(p["prices_per_token"]["output"])
        assert p["budget_usd"]["per_run"] == label.SAMPLE_RUN_BUDGET
        assert p["budget_usd"]["total"] == label.SAMPLE_TOTAL_BUDGET

    def test_the_word_for_the_held_out_split_never_appears_in_the_module(self):
        assert "holdout" not in _ADDENDUM.read_text().lower()


# ── the policies ─────────────────────────────────────────────────────────


def apply(policy, task, samples, **params):
    base = {"cheap": "haiku", "partner": "luna", "strong": "opus", "samples": samples}
    train = np.zeros((1, task.n))
    return policy(task, train, 1.0 - train, {**base, **params})


class TestSelfConsistency:
    def test_identical_samples_route_nothing_and_cost_four_haiku_calls(self):
        task, samples, _, _ = planted_task(disagree="none")

        right, cost, routed = apply(add.self_consistency, task, samples)

        assert not routed.any()
        assert cost["flat"] == pytest.approx(
            np.zeros((1, task.n)) + 4 * ROW_COST["haiku"]
        )
        assert cost["flat"] == pytest.approx(4 * task.u_flat["haiku"])
        np.testing.assert_array_equal(right, task.correct["haiku"][None])

    def test_a_split_row_is_routed_and_pays_for_the_strong_model(self):
        task, samples, wrong, picked = planted_task(disagree="all")

        right, cost, routed = apply(add.self_consistency, task, samples)

        np.testing.assert_array_equal(routed[0, :, 0], picked)
        want = 4 * ROW_COST["haiku"] + picked * ROW_COST["opus"]
        assert cost["flat"][0] == pytest.approx(want)
        assert right[0, wrong, 0].all()  # the strong model is perfect

    def test_a_single_dissenting_sample_is_enough_to_route(self):
        # Unanimity, not majority: 3 of 4 equal is still routed.
        task, samples, _, _ = planted_task(disagree="none")
        samples.codes[7, 3] = 99
        samples.correct[7, 3] = False

        _, _, routed = apply(add.self_consistency, task, samples)

        assert routed[0, :, 0].nonzero()[0].tolist() == [7]

    def test_an_invalid_sample_routes_the_row_even_when_all_four_are_equal(self):
        task, samples, _, _ = planted_task(disagree="none")
        samples.valid[3, 2] = False

        _, _, routed = apply(add.self_consistency, task, samples)

        assert routed[0, :, 0].nonzero()[0].tolist() == [3]

    def test_an_invalid_original_answer_routes_the_row_too(self):
        task, samples, _, _ = planted_task(disagree="none")
        samples.valid[5, 0] = False

        _, _, routed = apply(add.self_consistency, task, samples)

        assert routed[0, :, 0].nonzero()[0].tolist() == [5]

    def test_length_weighted_cost_is_exact(self):
        lens = np.linspace(1, 5, N)
        gold = gold_codes()
        wrong = np.arange(N) % 4 == 0
        cheap = np.where(wrong, gold + 12, gold)
        task = classification_task(
            gold,
            {"haiku": cheap, "luna": cheap, **{m: gold for m in replay.MODELS[2:]}},
            lens=lens,
        )
        extra = np.column_stack([np.where(wrong, 40 + j, cheap) for j in range(3)])
        samples = sample_set(task, gold, extra)

        _, cost, _ = apply(add.self_consistency, task, samples)

        want = 4 * task.u_len["haiku"] + wrong * task.u_len["opus"]
        assert cost["length"][0] == pytest.approx(want)

    def test_it_learns_nothing_so_the_labelled_side_changes_no_output(self):
        task, samples, _, _ = planted_task()
        params = {
            "cheap": "haiku",
            "partner": "luna",
            "strong": "opus",
            "samples": samples,
        }
        none = np.zeros((1, task.n))
        some = (np.random.default_rng(1).random((1, task.n)) < 0.4).astype(float)

        for policy in (add.self_consistency, add.sc_or_pair):
            a = policy(task, none, 1.0 - none, params)
            b = policy(task, some, 1.0 - some, params)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[2], b[2])

    def test_it_refuses_samples_for_other_rows(self):
        task, samples, _, _ = planted_task()
        samples.ids[0] = "other"

        with pytest.raises(ValueError, match="same rows"):
            apply(add.self_consistency, task, samples)

    def test_two_answers_that_differ_only_in_formatting_are_unanimous(self, layout):
        # norm(final_answer(.)): "**X**" and "x", and "Top Up" and "top_up", are one answer.
        rows = replay.read_jsonl(layout / "intent" / "dev.jsonl", layout)
        write_samples(
            layout,
            "intent",
            {
                "haiku-s1": {r["id"]: f"**{r['label']}**" for r in rows},
                "haiku-s2": {r["id"]: r["label"].upper() for r in rows},
                "haiku-s3": {r["id"]: r["label"].replace("_", " ") for r in rows},
            },
        )
        # Row 5: the original run was right too (top_up), so all four are one answer.
        samples = add.load_samples(
            "intent", "original", ["haiku-s1", "haiku-s2", "haiku-s3"], layout
        )
        unanimous = add._unanimous(samples, 4)

        assert unanimous[5]
        assert samples.valid[5].all()
        # Rows 0 and 1: Haiku said "top_up", the samples say the label: not unanimous.
        assert not unanimous[0]
        assert not unanimous[1]


class TestScOrPair:
    def test_it_routes_a_superset_of_c4(self):
        rng = np.random.default_rng(3)
        task, samples, _, _ = planted_task(luna="right", disagree="none")
        samples.codes[:, 1:] = np.where(
            rng.random((N, 3)) < 0.1, 77, samples.codes[:, 1:]
        )

        _, cost4, routed4 = apply(add.self_consistency, task, samples)
        _, cost5, routed5 = apply(add.sc_or_pair, task, samples)

        assert (routed5 | ~routed4).all()
        assert routed5.sum() > routed4.sum()
        assert (cost5["flat"] >= cost4["flat"]).all()
        assert (cost5["length"] >= cost4["length"]).all()

    def test_it_pays_for_luna_on_every_row_and_for_the_strong_model_when_routed(self):
        task, samples, wrong, picked = planted_task(luna="right", disagree="most")

        _, cost, routed = apply(add.sc_or_pair, task, samples)

        # Luna is right where Haiku is wrong, so the pair disagrees on every wrong row.
        np.testing.assert_array_equal(routed[0, :, 0], wrong)
        want = 4 * ROW_COST["haiku"] + ROW_COST["luna"] + wrong * ROW_COST["opus"]
        assert cost["flat"][0] == pytest.approx(want)

    def test_length_weighted_cost_is_exact(self):
        lens = np.linspace(1, 5, N)
        gold = gold_codes()
        wrong = np.arange(N) % 4 == 0
        cheap = np.where(wrong, gold + 12, gold)
        task = classification_task(
            gold,
            {"haiku": cheap, "luna": gold, **{m: gold for m in replay.MODELS[2:]}},
            lens=lens,
        )
        samples = sample_set(task, gold, np.column_stack([cheap] * 3))

        _, cost, routed = apply(add.sc_or_pair, task, samples)

        want = 4 * task.u_len["haiku"] + task.u_len["luna"] + wrong * task.u_len["opus"]
        assert cost["length"][0] == pytest.approx(want)
        np.testing.assert_array_equal(routed[0, :, 0], wrong)

    def test_an_invalid_cheap_pair_value_routes_the_row(self):
        task, samples, _, _ = planted_task(luna="same", disagree="none")
        task.valid["luna"][9] = False

        _, _, routed = apply(add.sc_or_pair, task, samples)

        assert routed[0, :, 0].nonzero()[0].tolist() == [9]


class TestExploratoryPolicies:
    def make(self, rows):
        """A 1-row-per-case task: `rows` lists the four answers (0 is right)."""
        n = len(rows)
        codes = np.array(rows)
        gold = np.zeros(n, int)
        task = classification_task(
            gold,
            {
                "haiku": codes[:, 0],
                "luna": codes[:, 0],
                **{m: gold for m in replay.MODELS[2:]},
            },
        )
        return task, sample_set(task, gold, codes[:, 1:])

    def test_modal_route_delivers_three_of_four_and_routes_the_rest(self):
        task, samples = self.make(
            [
                [1, 0, 0, 0],  # modal 0 with 3 votes: delivered, right
                [0, 0, 0, 1],  # modal 0: delivered
                [1, 1, 1, 0],  # modal 1 with 3 votes: delivered, wrong
                [0, 0, 1, 1],  # 2-2 tie: routed
                [0, 1, 2, 3],  # no mode: routed
            ]
        )

        right, cost, routed = apply(add.modal_route, task, samples, min_votes=3)

        assert routed[0, :, 0].tolist() == [False, False, False, True, True]
        assert right[0, :, 0].tolist() == [True, True, False, True, True]
        want = 4 * ROW_COST["haiku"] + routed[0, :, 0] * ROW_COST["opus"]
        assert cost["flat"][0] == pytest.approx(want)

    def test_modal_route_routes_a_valid_looking_but_invalid_mode(self):
        task, samples = self.make([[1, 1, 1, 1]])
        samples.valid[0, :] = False

        _, _, routed = apply(add.modal_route, task, samples, min_votes=3)

        assert routed.all()

    def test_the_vote_point_breaks_ties_toward_the_original_and_costs_four_calls(self):
        task, samples = self.make(
            [[0, 0, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0], [1, 2, 3, 0]]
        )

        right, cost, routed = apply(add.vote_point, task, samples)

        # Row 0: tie, original (0) is right. Row 1: tie, original (1) is wrong.
        # Row 2: modal 0. Row 3: no mode, original (1) is wrong.
        assert right[0, :, 0].tolist() == [True, False, True, False]
        assert not routed.any()
        assert cost["flat"] == pytest.approx(4 * ROW_COST["haiku"])

    def test_at_most_eight_exploratory_configs_per_task(self):
        for name, strong in add.PREREG_ADDENDUM["strong"].items():
            task, samples, _, _ = planted_task()
            configs = add.exploratory_configs(name, samples, add.PREREG_ADDENDUM)

            ids = [c[0] for c in configs]
            assert len(ids) == len(set(ids)) == 8
            assert f"C4:{strong}:ask2" in ids
            assert f"C4:{strong}:ask3" in ids
            others = {"sonnet", "sol", "opus"} - {strong}
            for o in others:
                assert f"C4:{o}" in ids
                assert f"C5:{o}" in ids
            assert "vote4" in ids
            assert f"modal3of4:{strong}" in ids

    def test_c4_on_fewer_samples_pays_for_fewer_calls(self):
        task, samples, _, _ = planted_task(disagree="none")

        _, cost, _ = apply(add.self_consistency, task, samples, n_ask=2)

        assert cost["flat"] == pytest.approx(2 * task.u_flat["haiku"])


# ── the study and its verdict ────────────────────────────────────────────


class TestStudy:
    def test_planted_self_disagreement_on_wrong_rows_builds_the_sc_trigger(self):
        task, samples, _, _ = planted_task(luna="same", disagree="all")

        res = study(task, samples)
        conf = res["tasks"]["intent"]["confirmatory"]
        v = res["verdict"]

        assert conf["C4"]["delta_flat"] > 10
        assert conf["C4"]["delta_length"] > 10
        assert conf["C4"]["delta_lb"] > 0
        # Luna repeats Haiku, so C5 routes the same rows at a higher cost: no margin.
        assert res["tasks"]["intent"]["paired"]["diff_flat"] < 2
        assert v["verdict"] == "BUILD_SC_TRIGGER"
        assert v["scope"] == "classification_only_single_task"
        assert res["tasks"]["intent"]["diagnostic"]["class"] == "useful"

    def test_identical_samples_route_nothing_and_the_kill_stands(self):
        task, samples, _, _ = planted_task(luna="same", disagree="none")

        res = study(task, samples)
        t = res["tasks"]["intent"]

        for pol in ("C4", "C5"):
            s = res["tasks"]["intent"]["golds"]["original"]["confirmatory"][pol]
            assert s["routed_share"]["mean"] == 0.0
        # Exactly four Haiku calls per row, in $ per 1k rows.
        c4 = t["golds"]["original"]["confirmatory"]["C4"]["cost_flat"]["mean"]
        assert c4 == pytest.approx(1000 * 4 * ROW_COST["haiku"])
        assert res["verdict"]["verdict"] == "KILL_STANDS"
        assert t["flags"]["C4"] == 0.0
        assert t["diagnostic"]["class"] == "not_useful"

    def test_only_c5_passing_builds_the_pair_trigger(self):
        # Haiku is wrong and self-consistent where Luna is right: C4 is blind, C5 sees it.
        task, samples, _, _ = planted_task(luna="noisy", disagree="none")

        res = study(task, samples)
        ok = res["verdict"]["passes"]["intent"]

        assert ok == {"C4": False, "C5": True}
        assert res["verdict"]["verdict"] == "BUILD_SC_PLUS_PAIR_TRIGGER"
        assert "only C5" in res["verdict"]["reasons"][0]

    def test_c5_that_clears_the_margin_over_a_passing_c4_builds_the_pair_trigger(self):
        # C4 catches most of the wrong rows; Luna's disagreement catches the rest.
        task, samples, _, _ = planted_task(luna="noisy", disagree="most")

        res = study(task, samples)
        t = res["tasks"]["intent"]

        assert res["verdict"]["passes"]["intent"] == {"C4": True, "C5": True}
        assert t["paired"]["diff_flat"] >= 2
        assert t["paired"]["diff_length"] >= 2
        assert t["paired"]["diff_lb"] > 0
        assert res["verdict"]["verdict"] == "BUILD_SC_PLUS_PAIR_TRIGGER"
        assert "beats C4" in res["verdict"]["reasons"][0]

    def test_a_failed_sample_check_on_intent_is_inconclusive_even_when_a_policy_passes(
        self,
    ):
        task, samples, _, _ = planted_task(luna="same", disagree="all")
        checks = add.Checks(
            v123={"intent": ["haiku-s2 V1: 0.700 outside [0.760, 0.860]"]}
        )

        res = study(task, samples, checks)

        assert res["verdict"]["passes"]["intent"]["C4"]  # computed, but not decisive
        assert res["verdict"]["verdict"] == "INCONCLUSIVE"
        assert res["verdict"]["scope"] is None
        assert any("V1" in f for f in res["verdict"]["flags"])

    def test_the_bound_is_the_alpha_over_two_quantile(self):
        prereg = add.effective_prereg()
        draws = np.arange(1001) / 10.0

        assert replay.lower_bound(draws, prereg) == pytest.approx(
            np.quantile(draws, 0.025)
        )
        assert replay.lower_bound(draws, prereg) == pytest.approx(2.5)

        task, samples, _, _ = planted_task()
        res = study(task, samples)
        assert res["tasks"]["intent"]["bootstrap"]["alpha_each"] == pytest.approx(0.025)

    def test_a_sign_flip_on_the_adjudicated_gold_is_flagged_not_a_kill(self, tmp_path):
        task, samples, wrong, _ = planted_task(luna="same", disagree="all")
        # Adjudication: Haiku's "mistakes" were right and the strong models wrong
        # on them, so routing those rows now only costs.
        flipped = {m: c.copy() for m, c in task.correct.items()}
        for m in ("haiku", "luna"):
            flipped[m][:] = True
        for m in ("sonnet", "sol", "opus"):
            flipped[m][:, 0] = ~wrong
        adj = dataclasses.replace(task, correct=flipped, strict=flipped, _fit={})

        res = study(task, samples, adjudicated=(adj, samples))

        assert res["tasks"]["intent"]["confirmatory"]["C4"]["delta_flat"] > 0
        assert res["tasks"]["intent"]["confirmatory"]["C4"]["sign_flip"]
        assert res["verdict"]["verdict"] == "BUILD_SC_TRIGGER"
        assert any("changes sign" in f for f in res["verdict"]["flags"])
        add.write_markdown(res, tmp_path / "r.md")
        assert (
            "FLAG: Delta A changes sign on the adjudicated gold for C4"
            in (tmp_path / "r.md").read_text()
        )

    def test_a_pass_with_a_not_useful_diagnostic_is_flagged(self):
        # Rows 0 mod 4: Haiku and Luna share a mistake and the samples agree on it
        # (the blind spot, never flagged). Rows 1 mod 4: both are wrong, differently,
        # and Haiku's samples scatter. C4 passes on the second kind, misses the first.
        gold = gold_codes()
        blind = np.arange(N) % 4 == 0
        alone = np.arange(N) % 4 == 1
        cheap = np.where(blind | alone, gold + 12, gold)
        luna = np.where(blind, cheap, np.where(alone, gold + 20, gold))
        task = classification_task(
            gold, {"haiku": cheap, "luna": luna, **{m: gold for m in replay.MODELS[2:]}}
        )
        extra = np.column_stack([np.where(alone, 40 + j, cheap) for j in range(3)])
        samples = sample_set(task, gold, extra)

        res = study(task, samples)

        assert res["tasks"]["intent"]["diagnostic"]["class"] == "not_useful"
        assert res["tasks"]["intent"]["diagnostic"]["blind"]["flagged"] == 0
        assert res["verdict"]["passes"]["intent"]["C4"]
        assert res["verdict"]["verdict"] == "BUILD_SC_TRIGGER"
        assert any("not useful" in f for f in res["verdict"]["flags"])

    def test_results_are_json_serialisable_and_embed_the_provenance(self, tmp_path):
        task, samples, _, _ = planted_task()
        checks = add.Checks(
            files={
                "intent": {
                    "haiku-s1": {
                        "sha256": "ab" * 32,
                        "cost_sha256": "cd" * 32,
                        "cost": 0.01,
                        "used": True,
                    },
                    "haiku-s4": {
                        "sha256": "ef" * 32,
                        "cost_sha256": "01" * 32,
                        "cost": 0.02,
                        "used": False,
                    },
                }
            },
            selected={"intent": ["haiku-s1"]},
        )

        res = study(task, samples, checks)
        paths = add.write_results(res, tmp_path)
        loaded = json.loads((tmp_path / "addendum-1.json").read_text())
        md = (tmp_path / "addendum-1.md").read_text()
        meta = loaded["meta"]

        assert [p.name for p in paths] == ["addendum-1.md", "addendum-1.json"]
        assert meta["prereg_addendum_sha256"] == add.prereg_addendum_hash()
        assert meta["prereg_addendum"] == add.PREREG_ADDENDUM
        assert meta["code_sha256"] == hashlib.sha256(_ADDENDUM.read_bytes()).hexdigest()
        assert (
            meta["parent_design_sha256"]
            == add.PREREG_ADDENDUM["parent"]["design_sha256"]
        )
        assert (
            meta["parent_prereg_sha256"]
            == add.PREREG_ADDENDUM["parent"]["prereg_sha256"]
        )
        assert set(meta["git"]) == {"head", "dirty"}
        if add.ADDENDUM_DESIGN.exists():
            assert (
                meta["design_sha256"]
                == hashlib.sha256(add.ADDENDUM_DESIGN.read_bytes()).hexdigest()
            )
        assert meta["sample_files"]["intent"]["haiku-s1"] == {
            "sha256": "ab" * 32,
            "cost_sha256": "cd" * 32,
            "used": True,
        }
        assert loaded["measured_cost_usd"] == {"intent": 0.01, "total": 0.01}
        assert meta["prereg_addendum_sha256"] in md
        assert "ab" * 32 in md
        assert "BUILD_SC_TRIGGER" in md or "KILL_STANDS" in md


# ── mechanism diagnostic and label-noise control ─────────────────────────


def intent_shaped_task():
    """231 rows mirroring the disclosed counts: 33 blind, 44 Haiku-wrong, 187 right."""
    gold = np.zeros(N, int)
    cheap = np.where(np.arange(N) < 44, 1, 0)
    luna = np.where(np.arange(N) < 33, 1, 0)
    return classification_task(
        gold, {"haiku": cheap, "luna": luna, **{m: gold for m in replay.MODELS[2:]}}
    )


def flags(blind=0, other_wrong=0, right=0):
    flag = np.zeros(N, bool)
    flag[:blind] = True
    flag[33 : 33 + other_wrong] = True
    flag[44 : 44 + right] = True
    return flag


class TestDiagnostic:
    def test_counts_and_wilson_intervals_on_a_hand_built_example(self):
        # 8 rows: haiku wrong on 0-3 (blind: luna agrees) and 4-5 (luna right); 6-7 right.
        gold = np.zeros(8, int)
        haiku = np.array([1, 1, 1, 1, 1, 1, 0, 0])
        luna = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        task = classification_task(
            gold, {"haiku": haiku, "luna": luna, **{m: gold for m in replay.MODELS[2:]}}
        )
        flag = np.array([1, 1, 0, 0, 1, 0, 1, 0], bool)

        d = add.mechanism_diagnostic(task, flag)

        assert (d["blind"]["n"], d["blind"]["flagged"]) == (4, 2)
        assert (d["wrong"]["n"], d["wrong"]["flagged"]) == (6, 3)
        assert (d["right"]["n"], d["right"]["flagged"]) == (2, 1)
        lo, hi = d["blind"]["wilson"]
        assert lo == pytest.approx(0.150, abs=5e-4)
        assert hi == pytest.approx(0.850, abs=5e-4)
        assert d["blind"]["rate"] == 0.5
        assert d["class"] == "weak"

    def test_the_useful_cuts_are_exact_at_the_disclosed_counts(self):
        task = intent_shaped_task()
        assert task.correct["haiku"][:, 0].sum() == 187

        assert (
            add.mechanism_diagnostic(task, flags(blind=14, right=18))["class"]
            == "useful"
        )
        assert (
            add.mechanism_diagnostic(task, flags(blind=13, right=18))["class"] == "weak"
        )
        assert (
            add.mechanism_diagnostic(task, flags(blind=14, right=19))["class"] == "weak"
        )
        assert (
            add.mechanism_diagnostic(task, flags(blind=33, right=0))["class"]
            == "useful"
        )

    def test_the_not_useful_cut_is_exact(self):
        task = intent_shaped_task()

        assert add.mechanism_diagnostic(task, flags(blind=6))["class"] == "not_useful"
        assert add.mechanism_diagnostic(task, flags(blind=7))["class"] == "weak"
        # Not useful even when it flags every right row.
        assert (
            add.mechanism_diagnostic(task, flags(blind=0, right=187))["class"]
            == "not_useful"
        )

    def test_r_blind_r_wrong_and_ff_are_counted_on_the_right_rows(self):
        task = intent_shaped_task()

        d = add.mechanism_diagnostic(task, flags(blind=20, other_wrong=5, right=10))

        assert (d["blind"]["n"], d["blind"]["flagged"]) == (33, 20)
        assert (d["wrong"]["n"], d["wrong"]["flagged"]) == (44, 25)
        assert (d["right"]["n"], d["right"]["flagged"]) == (187, 10)

    def test_threshold_comparisons_do_not_round(self):
        assert add.at_least(2, 5, 0.4)
        assert not add.at_least(2, 6, 0.4)
        assert add.at_least(14, 33, 0.40)
        assert not add.at_least(13, 33, 0.40)
        assert add.at_most(18, 187, 0.10)
        assert not add.at_most(19, 187, 0.10)
        assert add.at_most(6, 33, 0.20)
        assert not add.at_most(7, 33, 0.20)
        assert not add.at_least(0, 0, 0.4)
        assert not add.at_most(0, 0, 0.4)

    def test_wilson_matches_a_known_value_and_handles_the_edges(self):
        lo, hi = add.wilson(0, 10, 1.959964)
        assert lo == pytest.approx(0.0, abs=1e-12)
        assert hi == pytest.approx(0.2775, abs=5e-4)
        assert add.wilson(0, 0, 1.96) is None
        lo, hi = add.wilson(10, 10, 1.959964)
        assert hi == pytest.approx(1.0, abs=1e-12)
        assert lo == pytest.approx(0.7225, abs=5e-4)


def adjudication_layout(layout):
    """Intent layout with blind rows i0, i1, i3, i4 and an adjudication that
    keeps i0 and i3 (still wrong), relabels i1 (now right) and drops i4."""
    data = layout
    rows = replay.read_jsonl(data / "intent" / "dev.jsonl", data)
    for model in ("haiku", "luna"):
        out = [
            {"id": r["id"], "answer": "top_up" if k in (0, 1, 3, 4) else r["label"]}
            for k, r in enumerate(rows)
        ]
        write_jsonl(data / "labels" / "intent" / f"{model}.jsonl", out)
    adjudicated = []
    for r in rows:
        if r["id"] == "i4":
            continue  # dropped: no defensible label
        adjudicated.append({**r, "label": "top_up" if r["id"] == "i1" else r["label"]})
    write_jsonl(data / "intent" / "dev-adjudicated.jsonl", adjudicated)
    return (
        replay.load_task("intent", "original", data),
        replay.load_task("intent", "adjudicated", data),
    )


class TestNoiseControl:
    def test_blind_spot_rows_are_grouped_by_what_adjudication_did(self, layout):
        original, adjudicated = adjudication_layout(layout)

        groups = add.blind_spot_groups(original, adjudicated)

        assert groups == {"A": ["i0", "i3"], "B": ["i1"], "C": ["i4"]}

    def test_flag_rates_per_group_and_the_two_conclusions(self, layout):
        original, adjudicated = adjudication_layout(layout)
        by_id = {
            "i0": True,
            "i1": True,
            "i2": False,
            "i3": False,
            "i4": True,
            "i5": False,
        }

        nc = add.noise_control(original, adjudicated, by_id)

        g = nc["groups"]
        assert (g["A"]["n"], g["A"]["flagged"]) == (2, 1)
        assert (g["B"]["n"], g["B"]["flagged"]) == (1, 1)
        assert (g["C"]["n"], g["C"]["flagged"]) == (1, 1)
        assert (g["B_or_C"]["n"], g["B_or_C"]["flagged"]) == (2, 2)
        assert nc["finds_real_errors"] is True  # 1/2 >= 0.40
        assert nc["gap_b_or_c_minus_a"] == pytest.approx(0.5)
        assert nc["tracks_ambiguity"] is True
        # The adjudicated gold has 5 rows: Haiku wrong on i0, i3 only (i1 was relabelled).
        assert (nc["wrong"]["n"], nc["wrong"]["flagged"]) == (2, 1)
        assert (nc["right"]["n"], nc["right"]["flagged"]) == (3, 1)

    def test_it_finds_nothing_when_the_flag_ignores_still_wrong_rows(self, layout):
        original, adjudicated = adjudication_layout(layout)
        by_id = {
            "i0": False,
            "i1": False,
            "i2": False,
            "i3": False,
            "i4": False,
            "i5": False,
        }

        nc = add.noise_control(original, adjudicated, by_id)

        assert nc["finds_real_errors"] is False
        assert nc["tracks_ambiguity"] is False
        assert nc["gap_b_or_c_minus_a"] == 0.0

    def test_a_dropped_row_is_neither_wrong_nor_right_on_the_adjudicated_gold(
        self, layout
    ):
        original, adjudicated = adjudication_layout(layout)

        assert "i4" in original.ids
        assert "i4" not in adjudicated.ids
        assert adjudicated.n == original.n - 1

    def test_the_counts_the_design_discloses_are_recomputed_from_the_labels(
        self, layout
    ):
        original, adjudicated = adjudication_layout(layout)

        got = add.disclosure_counts(original, adjudicated, "opus", add.PREREG_ADDENDUM)

        assert got["blind"] == 4
        assert got["groups"] == {"A": 2, "B": 1, "C": 1}
        assert got["adjudicated_rows"] == 5
        assert got["rows"] == 6


# ── the sample checks, on a golden-layout fixture ────────────────────────

STEMS = ["haiku-s1", "haiku-s2", "haiku-s3"]


def write_samples(data, task, answers, **cost_over):
    """Write sample files: `answers` maps a stem to {id: raw answer}."""
    rows = replay.read_jsonl(data / task / "dev.jsonl", data)
    original = replay.read_json(data / "labels" / task / "haiku.cost.json", data)
    implied = add.implied_original_output_tokens(original, add.PREREG_ADDENDUM)
    for stem, by_id in answers.items():
        write_jsonl(
            data / "labels" / task / f"{stem}.jsonl",
            [
                {"id": r["id"], "answer": by_id.get(r["id"])}
                for r in rows
                if r["id"] in by_id
            ],
        )
        cost = {
            "route": "openrouter",
            "model": add.PREREG_ADDENDUM["model_id"],
            "cost": 0.01,
            "tokens": 5000,
            "rows": len(rows),
            "input_tokens": 4000,
            "output_tokens": round(implied * len(rows)),
            "source": "test",
            **cost_over,
        }
        (data / "labels" / task / f"{stem}.cost.json").write_text(json.dumps(cost))


@pytest.fixture
def layout(tmp_path):
    return make_layout(tmp_path)


def lenient(**v):
    """An addendum on the 6-row layout: wide V1/V3 bands, no disclosure, intent only."""
    base = add.PREREG_ADDENDUM["v"]
    spec = {
        **base,
        "v1": {"intent": {"metric": "accuracy", "center": 0.67, "tol": 0.5}},
        "v3_tol_pts": 100.0,
        **v,
    }
    return {**add.PREREG_ADDENDUM, "run_tasks": ["intent"], "disclosure": {}, "v": spec}


def tiny_prereg():
    return {**replay.PREREG, "splits": 20, "n_labelled": 3, "n_confirmatory": 2}


def run_checks(layout, addendum=None, reference=None):
    return add.sample_checks(layout, addendum or lenient(), tiny_prereg(), reference)


def good_answers(layout, flips=(5, 4, 3)):
    """Haiku's own answers, with sample j flipping row flips[j] to another valid name."""
    rows = replay.read_jsonl(layout / "intent" / "dev.jsonl", layout)
    original = {
        r["id"]: r["answer"]
        for r in replay.read_jsonl(layout / "labels" / "intent" / "haiku.jsonl", layout)
    }
    out = {}
    for stem, flip in zip(STEMS, flips, strict=True):
        answers = dict(original)
        answers[rows[flip]["id"]] = (
            "lost_card" if original[rows[flip]["id"]] != "lost_card" else "card_arrival"
        )
        out[stem] = answers
    return out


class TestSampleChecks:
    def test_a_good_set_passes_and_records_every_files_hash(self, layout):
        write_samples(layout, "intent", good_answers(layout))

        rep = run_checks(layout)

        assert rep.hard_ok, rep.problems
        assert rep.ok, rep.v123
        assert rep.selected["intent"] == STEMS
        json.dumps({"files": rep.files, "v3": rep.v3, "disclosure": rep.disclosure})
        for stem in STEMS:
            f = rep.files["intent"][stem]
            path = layout / "labels" / "intent" / f"{stem}.jsonl"
            assert f["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            assert (
                f["cost_sha256"]
                == hashlib.sha256(
                    path.with_suffix(".cost.json").read_bytes()
                ).hexdigest()
            )
            assert f["used"]
        assert 0 <= rep.v3["intent"]["among_new"] <= 1

    def test_a_missing_file_is_a_hard_problem(self, layout):
        write_samples(
            layout,
            "intent",
            {k: v for k, v in good_answers(layout).items() if k != "haiku-s3"},
        )

        rep = run_checks(layout)

        assert not rep.hard_ok
        assert any("haiku-s3" in p and "missing" in p for p in rep.problems)
        assert rep.selected["intent"] == []

    def test_no_sample_files_at_all_fails_with_a_message_per_file(self, layout):
        rep = run_checks(layout)

        assert not rep.ok
        assert len([p for p in rep.problems if "missing" in p]) == 3

    def test_v1_failure_is_not_hard_but_makes_it_inconclusive_on_intent(self, layout):
        write_samples(layout, "intent", good_answers(layout))
        spec = {"v1": {"intent": {"metric": "accuracy", "center": 0.95, "tol": 0.05}}}

        rep = run_checks(layout, lenient(**spec))

        assert rep.hard_ok
        assert not rep.ok
        assert sum("V1" in m for m in rep.v123["intent"]) == 3

    def test_v2_failure_when_the_output_tokens_leave_the_band(self, layout):
        write_samples(layout, "intent", good_answers(layout))
        original = replay.read_json(
            layout / "labels" / "intent" / "haiku.cost.json", layout
        )
        implied = add.implied_original_output_tokens(original, add.PREREG_ADDENDUM)
        for stem, factor in zip(STEMS, (2.1, 1.0, 0.49), strict=True):
            path = layout / "labels" / "intent" / f"{stem}.cost.json"
            cost = json.loads(path.read_text())
            cost["output_tokens"] = round(implied * cost["rows"] * factor)
            path.write_text(json.dumps(cost))

        rep = run_checks(layout)

        failed = [m for m in rep.v123["intent"] if "V2" in m]
        assert len(failed) == 2
        assert any("haiku-s1" in m for m in failed)
        assert any("haiku-s3" in m for m in failed)

    def test_v2_inside_the_band_at_the_edges(self):
        spec = add.PREREG_ADDENDUM["v"]["v2"]

        assert add.check_v2(50.0, 100.0, spec)["ok"]
        assert add.check_v2(200.0, 100.0, spec)["ok"]
        assert not add.check_v2(49.9, 100.0, spec)["ok"]
        assert not add.check_v2(200.1, 100.0, spec)["ok"]
        assert not add.check_v2(None, 100.0, spec)["ok"]

    def test_v2_uses_the_original_runs_implied_output_tokens(self):
        # cost = 1e-7 in + 5e-7 out and T = in + out  =>  out = (cost - 1e-7 T) / 4e-7.
        cost = {"cost": 0.0297, "tokens": 209259, "rows": 231}

        got = add.implied_original_output_tokens(cost, add.PREREG_ADDENDUM)

        assert got == pytest.approx((0.0297 - 1e-7 * 209259) / 4e-7 / 231)
        assert got == pytest.approx(95, abs=0.5)

    def test_v1_band_edges(self):
        spec = {"center": 0.81, "tol": 0.05}

        assert add.check_v1(0.86, spec)["ok"]
        assert add.check_v1(0.76, spec)["ok"]
        assert not add.check_v1(0.8601, spec)["ok"]
        assert not add.check_v1(0.7599, spec)["ok"]

    def test_v3_signs_and_band(self):
        agree = np.zeros((10, 4), int)
        assert add.check_v3(agree, 5.0)["ok"]
        assert add.check_v3(agree, 5.0)["diff_pts"] == 0.0
        # New samples agree with each other but not with the original.
        codes = np.column_stack([np.arange(10), np.zeros((10, 3), int)])
        v3 = add.check_v3(codes, 5.0)
        assert v3["among_new"] == 1.0
        assert v3["with_original"] == pytest.approx(0.1)
        assert v3["diff_pts"] == pytest.approx(90.0)
        assert not v3["ok"]
        # The opposite: each new sample agrees with the original on 6 of 9 rows,
        # but any two new samples agree on only 3.
        codes = np.zeros((9, 4), int)
        codes[0:3, 1] = codes[3:6, 2] = codes[6:9, 3] = 1
        v3 = add.check_v3(codes, 5.0)
        assert v3["among_new"] == pytest.approx(3 / 9)
        assert v3["with_original"] == pytest.approx(6 / 9)
        assert v3["diff_pts"] < 0
        assert not v3["ok"]

    def test_a_null_answer_is_a_v4_failure_with_no_replacement_on_hand(self, layout):
        answers = good_answers(layout)
        answers["haiku-s2"] = {**answers["haiku-s2"], "i1": None}
        write_samples(layout, "intent", answers)

        rep = run_checks(layout)

        assert not rep.hard_ok
        assert any(
            "haiku-s2" in p and "null or invalid" in p and "haiku-s4" in p
            for p in rep.problems
        )

    def test_a_missing_id_is_a_v4_failure(self, layout):
        answers = good_answers(layout)
        del answers["haiku-s1"]["i2"]
        write_samples(layout, "intent", answers)

        rep = run_checks(layout)

        assert not rep.hard_ok
        assert any("haiku-s1" in p and "ids (1 missing" in p for p in rep.problems)

    def test_a_file_failing_only_v4_is_replaced_once_by_s4(self, layout):
        answers = good_answers(layout)
        bad = {**answers["haiku-s2"], "i1": "not an intent"}
        write_samples(
            layout,
            "intent",
            {
                **answers,
                "haiku-s2": bad,
                "haiku-s4": good_answers(layout, flips=(2, 1, 0))["haiku-s1"],
            },
        )

        rep = run_checks(layout)

        assert rep.hard_ok, rep.problems
        assert rep.selected["intent"] == ["haiku-s1", "haiku-s4", "haiku-s3"]
        assert rep.files["intent"]["haiku-s2"]["used"] is False
        assert rep.files["intent"]["haiku-s4"]["used"] is True
        assert any("replaced by haiku-s4" in line for line in rep.lines)

    def test_a_replacement_that_also_fails_v4_is_refused(self, layout):
        answers = good_answers(layout)
        bad = {**answers["haiku-s2"], "i1": "not an intent"}
        write_samples(layout, "intent", {**answers, "haiku-s2": bad, "haiku-s4": bad})

        rep = run_checks(layout)

        assert not rep.hard_ok
        assert any("replacement also fails" in p for p in rep.problems)

    def test_two_files_failing_v4_cannot_both_be_replaced(self, layout):
        answers = good_answers(layout)
        for stem in ("haiku-s1", "haiku-s2"):
            answers[stem] = {**answers[stem], "i1": "not an intent"}
        write_samples(layout, "intent", {**answers, "haiku-s4": answers["haiku-s3"]})

        rep = run_checks(layout)

        assert not rep.hard_ok

    def test_a_file_failing_v1_as_well_as_v4_is_not_replaced(self, layout):
        answers = good_answers(layout)
        spec = {"v1": {"intent": {"metric": "accuracy", "center": 0.95, "tol": 0.01}}}
        answers["haiku-s2"] = {**answers["haiku-s2"], "i1": "not an intent"}
        write_samples(layout, "intent", {**answers, "haiku-s4": answers["haiku-s3"]})

        rep = run_checks(layout, lenient(**spec))

        assert not rep.hard_ok
        assert any("cannot be replaced" in p for p in rep.problems)

    def test_a_replacement_file_when_nothing_failed_is_refused(self, layout):
        answers = good_answers(layout)
        write_samples(layout, "intent", {**answers, "haiku-s4": answers["haiku-s1"]})

        rep = run_checks(layout)

        assert any("nothing may be replaced" in p for p in rep.problems)

    def test_a_sample_identical_to_the_original_is_a_cache_signature(self, layout):
        answers = good_answers(layout)
        original = {
            r["id"]: r["answer"]
            for r in replay.read_jsonl(
                layout / "labels" / "intent" / "haiku.jsonl", layout
            )
        }
        write_samples(layout, "intent", {**answers, "haiku-s2": original})

        rep = run_checks(layout)

        assert not rep.hard_ok
        assert any(
            "haiku-s2" in p and "identical on every row to ['haiku']" in p
            for p in rep.problems
        )

    def test_two_identical_samples_fail_the_later_one(self, layout):
        answers = good_answers(layout)
        write_samples(layout, "intent", {**answers, "haiku-s3": answers["haiku-s2"]})

        rep = run_checks(layout)

        assert any(
            "haiku-s3" in p and "identical on every row to ['haiku-s2']" in p
            for p in rep.problems
        )

    def test_identical_after_normalisation_still_counts_as_identical(self, layout):
        answers = good_answers(layout)
        shouted = {k: f"**{v.upper()}**" for k, v in answers["haiku-s2"].items()}
        write_samples(layout, "intent", {**answers, "haiku-s3": shouted})

        rep = run_checks(layout)

        assert any("haiku-s3" in p and "identical" in p for p in rep.problems)

    def test_the_route_and_model_must_be_the_designed_ones(self, layout):
        write_samples(layout, "intent", good_answers(layout), route="direct")

        rep = run_checks(layout)

        assert not rep.hard_ok
        assert sum("route/model/rows direct" in p for p in rep.problems) == 3

    def test_an_unexpected_sample_file_is_reported(self, layout):
        write_samples(
            layout,
            "intent",
            {**good_answers(layout), "haiku-s5": good_answers(layout)["haiku-s1"]},
        )

        rep = run_checks(layout)

        assert any("unexpected sample files ['haiku-s5']" in p for p in rep.problems)

    def test_smoke_files_are_ignored(self, layout):
        write_samples(layout, "intent", good_answers(layout))
        (layout / "labels" / "intent" / "haiku-s1-smoke3.jsonl").write_text(
            "not json\n"
        )

        rep = run_checks(layout)

        assert rep.ok

    def test_a_cost_file_without_output_tokens_fails_v2(self, layout):
        write_samples(layout, "intent", good_answers(layout))
        path = layout / "labels" / "intent" / "haiku-s1.cost.json"
        cost = json.loads(path.read_text())
        del cost["output_tokens"]
        path.write_text(json.dumps(cost))

        rep = run_checks(layout)

        assert any("haiku-s1" in m and "V2" in m for m in rep.v123["intent"])

    def test_the_parent_hash_must_match(self, layout):
        write_samples(layout, "intent", good_answers(layout))
        add_bad = {
            **lenient(),
            "parent": {**add.PREREG_ADDENDUM["parent"], "prereg_sha256": "0" * 64},
        }

        rep = run_checks(layout, add_bad)

        assert any("parent PREREG hash" in p for p in rep.problems)

    def test_the_disclosed_counts_must_reproduce(self, layout):
        write_samples(layout, "intent", good_answers(layout))
        wrong = {**lenient(), "disclosure": {"intent": {"blind": 33, "rows": 6}}}

        rep = run_checks(layout, wrong)

        assert any(
            "differ from the disclosure" in p and "'blind'" in p for p in rep.problems
        )

    def test_the_disclosed_counts_pass_when_they_match(self, layout):
        write_samples(layout, "intent", good_answers(layout))
        right = {
            **lenient(),
            "disclosure": {"intent": {"rows": 6, "adjudicated_rows": 5}},
        }

        rep = run_checks(layout, right)

        assert rep.hard_ok
        assert rep.disclosure["intent"]["rows"] == 6


class TestParentReproduction:
    def test_the_parents_c1_is_reproduced(self):
        task, _, _, _ = planted_task()
        prereg = small_prereg(n_confirmatory=2)
        ref = add.reproduce_parent_c1(task, prereg, None)["got"]

        again = add.reproduce_parent_c1(task, prereg, ref)

        assert again["ok"]
        assert max(again["diff"].values()) == 0.0

    def test_a_perturbed_reference_fails_loudly(self):
        task, _, _, _ = planted_task()
        prereg = small_prereg(n_confirmatory=2)
        ref = add.reproduce_parent_c1(task, prereg, None)["got"]

        for key in ("acc", "delta_flat"):
            off = {**ref, key: ref[key] + 1e-6}
            assert not add.reproduce_parent_c1(task, prereg, off)["ok"]
        assert add.reproduce_parent_c1(
            task, prereg, {**ref, "acc": ref["acc"] + 5e-10}
        )["ok"]

    def test_sample_checks_stop_the_run_on_a_perturbed_reference(self, layout):
        write_samples(layout, "intent", good_answers(layout))
        ref = add.reproduce_parent_c1(
            replay.load_task("intent", "original", layout), tiny_prereg(), None
        )["got"]

        assert run_checks(layout, reference=ref).hard_ok
        bad = run_checks(
            layout, reference={**ref, "delta_flat": ref["delta_flat"] + 0.01}
        )

        assert not bad.hard_ok
        assert any("not reproduced" in p for p in bad.problems)

    def test_the_committed_reference_is_read_from_the_parents_results(self, tmp_path):
        results = {
            "tasks": {
                "intent": {
                    "golds": {
                        "original": {
                            "confirmatory": {
                                "C1": {
                                    "acc": {"mean": 0.83},
                                    "delta_flat": {"mean": 0.77},
                                }
                            }
                        }
                    }
                }
            }
        }
        path = tmp_path / "replay.json"
        path.write_text(json.dumps(results))

        assert add.committed_reference(path) == {"acc": 0.83, "delta_flat": 0.77}
        assert add.committed_reference(tmp_path / "missing.json") is None


class TestLoadSamples:
    def test_two_null_answers_never_count_as_agreeing(self, layout):
        path = layout / "labels" / "intent" / "haiku-s1.jsonl"
        write_jsonl(path, [{"id": "a", "answer": None}, {"id": "b", "answer": None}])

        finals, _, nulls = add._final_by_id(path, layout)

        assert nulls == 2
        assert finals["a"] != finals["b"]

    def test_columns_are_the_original_then_the_samples_on_the_golds_rows(self, layout):
        write_samples(layout, "intent", good_answers(layout))

        orig = add.load_samples("intent", "original", STEMS, layout)
        adj = add.load_samples("intent", "adjudicated", STEMS, layout)

        assert orig.codes.shape == (6, 4)
        assert adj.codes.shape == (5, 4)
        assert orig.ids == replay.load_task("intent", "original", layout).ids
        assert adj.ids == replay.load_task("intent", "adjudicated", layout).ids
        t = replay.load_task("intent", "original", layout)
        np.testing.assert_array_equal(orig.correct[:, 0], t.correct["haiku"][:, 0])
        np.testing.assert_array_equal(orig.valid[:, 0], t.valid["haiku"])

    def test_a_missing_row_is_invalid_wrong_and_equal_to_nothing(self):
        samples = add.build_sample_set(
            ["a", "b"],
            [{"a": "x", "b": "y"}, {"a": "x"}, {"a": "x"}],
            {"x", "y"},
            ["x", "y"],
        )

        assert samples.valid.tolist() == [[True, True, True], [True, False, False]]
        assert samples.correct.tolist() == [[True, True, True], [True, False, False]]
        assert samples.codes[1, 1] != samples.codes[1, 2]
        assert not add._unanimous(samples, 3)[1]


# ── label.py: the new options, with no network ───────────────────────────


class FakePipeline:
    def __init__(self, builder):
        self.builder = builder
        self.specifications = SimpleNamespace(
            llm=SimpleNamespace(
                cache_config=builder.cache_config, extra_params=builder.extra_params
            )
        )

    def execute(self):
        self.builder.executed += 1
        frame = self.builder.frame
        costs = SimpleNamespace(
            total_cost=Decimal("0.0123"),
            total_tokens=1000,
            input_tokens=700,
            output_tokens=300,
        )
        out = pd.DataFrame({"answer": [f"top_up {i}" for i in range(len(frame))]})
        return SimpleNamespace(to_pandas=lambda: out, costs=costs, lost_row_indices=[])


class FakeBuilder:
    """Records what label.py asks of PipelineBuilder; calls nothing."""

    created = 0
    executed = 0
    cache_config = None
    extra_params = None
    last: dict = {}

    @classmethod
    def create(cls):
        cls.created += 1
        cls.last = {}
        return cls()

    def from_dataframe(self, frame, input_columns, output_columns):
        type(self).frame = frame
        type(self).last["inputs"] = list(input_columns)
        return self

    def with_prompt(self, prompt):
        return self

    def with_llm(self, **kw):
        type(self).last["llm"] = kw
        return self

    def with_concurrency(self, n):
        type(self).last["concurrency"] = n
        return self

    def with_max_budget(self, usd):
        type(self).last["budget"] = usd
        return self

    def with_error_policy(self, policy):
        type(self).last["policy"] = policy
        return self

    def build(self):
        return FakePipeline(type(self))


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """label.py pointed at a tiny data directory with a recording pipeline."""
    data = tmp_path / "data"
    write_jsonl(
        data / "intent" / "dev.jsonl",
        [
            {"id": f"i{k}", "text": f"msg {k}", "label": INTENT_NAMES[k % 3]}
            for k in range(4)
        ],
    )
    (data / "intent" / "labels.json").write_text(json.dumps(INTENT_NAMES))
    (data / "labels" / "intent").mkdir(parents=True)
    (data / "labels" / "faithfulness").mkdir(parents=True)
    monkeypatch.setattr(label, "DATA", data)
    # Other tests in the suite may leave a global LiteLLM cache set; label.py
    # rightly refuses one, so start each test without it.
    import litellm

    monkeypatch.setattr(litellm, "cache", None, raising=False)
    builder = type("B", (FakeBuilder,), {"created": 0, "executed": 0, "last": {}})
    monkeypatch.setattr(label, "PipelineBuilder", builder)
    return SimpleNamespace(data=data, builder=builder)


def run_label(*args):
    return label.main(["--task", "intent", "--model", "haiku", *args])


class TestLabelSample:
    def test_a_sample_run_writes_haiku_s_n_never_haiku(self, lab):
        run_label("--sample", "1", "--route", "openrouter")

        labels = lab.data / "labels" / "intent"
        assert sorted(p.name for p in labels.iterdir()) == [
            "haiku-s1.cost.json",
            "haiku-s1.jsonl",
        ]
        rows = [
            json.loads(line)
            for line in (labels / "haiku-s1.jsonl").read_text().splitlines()
        ]
        assert [r["id"] for r in rows] == ["i0", "i1", "i2", "i3"]

    def test_the_openrouter_route_sets_the_model_prices_and_haiku_params(self, lab):
        run_label("--sample", "2", "--route", "openrouter")

        assert lab.builder.last["llm"] == {
            "model": "openrouter/anthropic/claude-haiku-5-5",
            "temperature": 1.0,
            "reasoning_effort": "low",
            "input_cost_per_1k_tokens": Decimal("0.0001"),
            "output_cost_per_1k_tokens": Decimal("0.0005"),
        }
        assert lab.builder.last["concurrency"] == 6
        assert lab.builder.last["policy"] == "skip"

    def test_a_sample_run_goes_through_openrouter_unless_told_otherwise(self, lab):
        # The pre-registered route; the direct account has no credit.
        run_label("--sample", "1")

        assert lab.builder.last["llm"]["model"] == label.OPENROUTER_MODEL
        assert "output_cost_per_1k_tokens" in lab.builder.last["llm"]

    def test_the_direct_route_with_sample_keeps_the_original_model_and_no_prices(
        self, lab
    ):
        run_label("--sample", "1", "--route", "direct")

        assert lab.builder.last["llm"] == {
            "model": "anthropic/claude-haiku-5-5",
            "temperature": 1.0,
            "reasoning_effort": "low",
        }

    def test_the_cost_file_records_route_model_tokens_cost_rows_and_source(self, lab):
        run_label("--sample", "3", "--route", "openrouter")

        cost = json.loads(
            (lab.data / "labels" / "intent" / "haiku-s3.cost.json").read_text()
        )
        assert cost == {
            "route": "openrouter",
            "model": "openrouter/anthropic/claude-haiku-5-5",
            "served_models": [],
            "cost": 0.0123,
            "tokens": 1000,
            "rows": 4,
            "input_tokens": 700,
            "output_tokens": 300,
            "source": "label.py --sample 3 (result.costs)",
        }

    def test_the_per_run_budget_defaults_to_five_cents_with_sample(self, lab):
        run_label("--sample", "1", "--route", "openrouter")
        assert lab.builder.last["budget"] == 0.05

    def test_an_explicit_budget_wins(self, lab):
        run_label("--sample", "1", "--budget", "0.02")
        assert lab.builder.last["budget"] == 0.02

    def test_it_refuses_to_overwrite_a_sample_and_calls_nothing(self, lab):
        run_label("--sample", "1", "--route", "openrouter")
        before = (lab.data / "labels" / "intent" / "haiku-s1.jsonl").read_bytes()
        runs = lab.builder.executed

        with pytest.raises(SystemExit, match="never overwritten"):
            run_label("--sample", "1", "--route", "openrouter")

        assert lab.builder.executed == runs
        assert lab.builder.created == 1
        assert (
            lab.data / "labels" / "intent" / "haiku-s1.jsonl"
        ).read_bytes() == before

    def test_it_refuses_when_only_the_cost_file_exists(self, lab):
        (lab.data / "labels" / "intent" / "haiku-s1.cost.json").write_text(
            '{"cost": 0.0}'
        )

        with pytest.raises(SystemExit, match="never overwritten"):
            run_label("--sample", "1")

        assert lab.builder.created == 0

    def test_a_smoke_run_writes_its_own_file_and_leaves_the_full_sample_free(self, lab):
        run_label("--sample", "1", "--route", "openrouter", "--limit", "2")

        names = sorted(p.name for p in (lab.data / "labels" / "intent").iterdir())
        assert names == ["haiku-s1-smoke2.cost.json", "haiku-s1-smoke2.jsonl"]
        run_label(
            "--sample", "1", "--route", "openrouter"
        )  # the real file is still free
        assert (lab.data / "labels" / "intent" / "haiku-s1.jsonl").exists()

    def test_the_total_budget_counts_every_haiku_s_cost_file_across_tasks(self, lab):
        for task, name, cost in (
            ("intent", "haiku-s1", 0.12),
            ("faithfulness", "haiku-s1", 0.10),
            ("intent", "haiku-s1-smoke3", 0.02),
        ):
            (lab.data / "labels" / task / f"{name}.cost.json").write_text(
                json.dumps({"cost": cost})
            )
        (lab.data / "labels" / "intent" / "haiku.cost.json").write_text(
            json.dumps({"cost": 9.0})
        )

        assert label.sample_spend(lab.data) == pytest.approx(0.24)
        # 0.24 + 0.05 = 0.29 <= 0.30: allowed. A 0.07 run would reach 0.31: refused.
        run_label("--sample", "2", "--route", "openrouter")
        assert lab.builder.executed == 1
        with pytest.raises(SystemExit, match="total budget"):
            run_label("--sample", "3", "--budget", "0.07")
        assert lab.builder.created == 1  # refused before any pipeline was built

    def test_an_explicit_total_budget_overrides_the_default_cap(self, lab):
        (lab.data / "labels" / "intent" / "haiku-s1.cost.json").write_text(
            json.dumps({"cost": 0.29})
        )

        with pytest.raises(SystemExit, match="total budget"):
            run_label("--sample", "2")
        run_label("--sample", "2", "--total-budget", "0.40")

        assert lab.builder.executed == 1

    def test_budget_arithmetic_is_exact_at_the_boundary(self):
        label.check_total_budget(0.05, 0.30, 0.25)  # equal to the cap: allowed
        label.check_total_budget(0.05, 0.30, 0.15 + 0.10)
        with pytest.raises(SystemExit):
            label.check_total_budget(0.05, 0.30, 0.2501)
        with pytest.raises(SystemExit):
            label.check_total_budget(0.31, 0.30, 0.0)

    @pytest.mark.parametrize(
        "config",
        [{"cache_config": {"type": "redis"}}, {"extra_params": {"caching": True}}],
    )
    def test_a_response_cache_is_refused_before_any_call(
        self, lab, monkeypatch, config
    ):
        for key, value in config.items():
            monkeypatch.setattr(lab.builder, key, value)

        with pytest.raises(SystemExit, match="response cache"):
            run_label("--sample", "1", "--route", "openrouter")

        assert lab.builder.executed == 0
        assert list((lab.data / "labels" / "intent").iterdir()) == []

    def test_a_global_litellm_cache_is_refused_too(self, lab, monkeypatch):
        import litellm

        monkeypatch.setattr(litellm, "cache", object(), raising=False)

        with pytest.raises(SystemExit, match="response cache"):
            run_label("--sample", "1")

        assert lab.builder.executed == 0

    @pytest.mark.parametrize(
        "args",
        [
            ["--task", "intent", "--model", "opus", "--sample", "1"],
            ["--task", "intent", "--model", "haiku", "--sample", "0"],
            ["--task", "intent", "--model", "haiku", "--sample", "10"],
            ["--task", "intent", "--model", "haiku", "--sample", "1", "--candidates"],
            ["--task", "intent", "--model", "haiku", "--route", "openrouter"],
            ["--task", "intent", "--model", "haiku", "--total-budget", "0.3"],
        ],
    )
    def test_bad_combinations_exit_before_anything_runs(self, lab, args):
        with pytest.raises(SystemExit):
            label.main(args)

        assert lab.builder.created == 0
        assert list((lab.data / "labels" / "intent").iterdir()) == []

    def test_route_openrouter_without_sample_cannot_overwrite_haiku_jsonl(self, lab):
        original = lab.data / "labels" / "intent" / "haiku.jsonl"
        original.write_text("original\n")

        with pytest.raises(SystemExit, match="only for --sample"):
            run_label("--route", "openrouter")

        assert original.read_text() == "original\n"


class TestAbortedSampleRuns:
    def test_an_aborted_run_is_charged_its_whole_budget_toward_the_cap(
        self, lab, monkeypatch
    ):
        def boom(self):
            raise RuntimeError("provider down")

        monkeypatch.setattr(FakePipeline, "execute", boom)
        with pytest.raises(RuntimeError):
            run_label("--sample", "1", "--budget", "0.05")

        labels = lab.data / "labels" / "intent"
        assert not (labels / "haiku-s1.jsonl").exists()  # the slot can be retried
        assert label.sample_spend(lab.data) == pytest.approx(0.05)


class TestLabelWithoutTheNewOptions:
    """The default behaviour is what it was before: same call, same files."""

    def test_a_plain_run_uses_the_original_settings_and_the_old_cost_file(self, lab):
        run_label("--limit", "2")

        assert lab.builder.last["llm"] == {
            "model": "anthropic/claude-haiku-5-5",
            "temperature": 1.0,
            "reasoning_effort": "low",
        }
        assert lab.builder.last["budget"] == 2.0
        labels = lab.data / "labels" / "intent"
        assert sorted(p.name for p in labels.iterdir()) == [
            "haiku-smoke2.cost.json",
            "haiku-smoke2.jsonl",
        ]
        text = (labels / "haiku-smoke2.cost.json").read_text()
        assert text == '{"cost": 0.0123, "tokens": 1000, "rows": 2}'

    def test_a_plain_run_overwrites_as_before(self, lab):
        run_label()
        run_label()

        assert lab.builder.executed == 2

    def test_another_model_keeps_its_deterministic_settings(self, lab):
        label.main(
            ["--task", "intent", "--model", "opus", "--limit", "1", "--budget", "0.5"]
        )

        assert lab.builder.last["llm"] == {
            "model": "anthropic/claude-opus-5-5",
            "temperature": 0.0,
        }
        assert lab.builder.last["budget"] == 0.5

    def test_the_candidates_run_still_writes_its_suffix(self, lab):
        write_jsonl(
            lab.data / "adjudication" / "intent.jsonl", [{"id": "i1"}, {"id": "i2"}]
        )

        run_label("--candidates")

        assert (lab.data / "labels" / "intent" / "haiku-candidates.jsonl").exists()

    def test_llm_settings_for_each_model_and_route(self):
        assert label.llm_settings("haiku", "direct") == (
            "anthropic/claude-haiku-5-5",
            {"temperature": 1.0, "reasoning_effort": "low"},
        )
        assert label.llm_settings("sol", "direct") == (
            "openai/gpt-6.1-sol",
            {"temperature": 0.0},
        )
        # The shared MODEL_PARAMS dict must not be mutated by the openrouter route.
        label.llm_settings("haiku", "openrouter")
        assert label.MODEL_PARAMS["haiku"] == {
            "temperature": 1.0,
            "reasoning_effort": "low",
        }
