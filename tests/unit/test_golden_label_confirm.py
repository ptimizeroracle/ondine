"""label.py --split confirm (review-next DESIGN-confirm.md section 2), with no network.

The confirmation labels fresh rows with Haiku four times, on one route, under a
hard $1.00 cap. The pipeline is replaced by a recorder, so nothing here can
call a model or spend a cent. Each test names the regression it catches.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from test_planner_addendum1 import (  # noqa: E402
    INTENT_NAMES,
    FakeBuilder,
    FakePipeline,
    label,
    write_jsonl,
)


class RecordingBuilder(FakeBuilder):
    """FakeBuilder that also keeps the prompt it was given."""

    def with_prompt(self, prompt):
        type(self).last["prompt"] = prompt
        return self


@pytest.fixture
def conf(tmp_path, monkeypatch):
    """A tiny golden data dir with dev and confirm rows for both tasks."""
    data = tmp_path / "data"
    write_jsonl(
        data / "intent" / "dev.jsonl",
        [
            {"id": f"i{k}", "text": f"dev {k}", "label": INTENT_NAMES[0]}
            for k in range(4)
        ],
    )
    write_jsonl(
        data / "intent" / "confirm.jsonl",
        [
            {"id": f"c{k}", "text": f"confirm {k}", "label": INTENT_NAMES[k % 3]}
            for k in range(5)
        ],
    )
    write_jsonl(
        data / "faithfulness" / "dev.jsonl",
        [
            {"id": f"f{k}", "doc": "d", "summary": "s", "label": "consistent"}
            for k in range(4)
        ],
    )
    write_jsonl(
        data / "faithfulness" / "confirm.jsonl",
        [
            {"id": f"fc{k}", "doc": f"doc {k}", "summary": "s", "label": "consistent"}
            for k in range(6)
        ],
    )
    (data / "intent" / "labels.json").write_text(json.dumps(INTENT_NAMES))
    for task in ("intent", "faithfulness"):
        (data / "labels" / task).mkdir(parents=True)
    monkeypatch.setattr(label, "DATA", data)
    import litellm

    monkeypatch.setattr(litellm, "cache", None, raising=False)
    builder = type("B", (RecordingBuilder,), {"created": 0, "executed": 0, "last": {}})
    monkeypatch.setattr(label, "PipelineBuilder", builder)
    return SimpleNamespace(data=data, builder=builder)


def run(*args: str, task: str = "intent") -> int:
    return label.main(["--task", task, "--model", "haiku", *args])


def confirm(*args: str, task: str = "intent", sample: str = "1") -> int:
    return run("--split", "confirm", "--sample", sample, *args, task=task)


def write_cost(data: Path, task: str, name: str, cost: float, folder: str = "confirm"):
    path = data / "labels" / task / folder / f"{name}.cost.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cost": cost}))


class TestWhatItWritesAndReads:
    def test_it_labels_the_confirm_rows_and_writes_under_labels_task_confirm(
        self, conf
    ):
        # Catches a confirm run that labels dev or writes next to the dev samples.
        confirm()

        folder = conf.data / "labels" / "intent" / "confirm"
        assert sorted(p.name for p in folder.iterdir()) == [
            "haiku-s1.cost.json",
            "haiku-s1.jsonl",
        ]
        ids = [
            json.loads(x)["id"]
            for x in (folder / "haiku-s1.jsonl").read_text().splitlines()
        ]
        assert ids == [f"c{k}" for k in range(5)]
        assert list((conf.data / "labels" / "intent").glob("*.json*")) == []

    def test_the_faithfulness_destination_and_rows(self, conf):
        confirm(task="faithfulness", sample="3")

        out = conf.data / "labels" / "faithfulness" / "confirm" / "haiku-s3.jsonl"
        assert [json.loads(x)["id"] for x in out.read_text().splitlines()] == [
            f"fc{k}" for k in range(6)
        ]

    def test_a_smoke_run_has_its_own_name_inside_confirm(self, conf):
        confirm("--limit", "3")

        names = sorted(
            p.name for p in (conf.data / "labels" / "intent" / "confirm").iterdir()
        )
        assert names == ["haiku-s1-smoke3.cost.json", "haiku-s1-smoke3.jsonl"]
        assert len(conf.builder.frame) == 3
        confirm()  # the real slot stays free
        assert (conf.data / "labels" / "intent" / "confirm" / "haiku-s1.jsonl").exists()

    def test_the_prompt_and_input_columns_are_the_ones_prepare_gives(self, conf):
        # Catches a changed prompt: the confirm answers must be drawn exactly as dev's.
        for task, inputs in (
            ("intent", ["text"]),
            ("faithfulness", ["doc", "summary"]),
        ):
            confirm(task=task, sample="2")

            _, cols, prompt = label.prepare(task, label.load(task, "confirm"))
            assert conf.builder.last["inputs"] == inputs == cols
            assert conf.builder.last["prompt"] == prompt

    def test_the_dev_run_is_unchanged_and_ignores_confirm_rows(self, conf):
        # Catches --split leaking into the default: dev still reads dev.jsonl and
        # writes labels/<task>/haiku-s<N>.jsonl.
        run("--sample", "1")

        labels = conf.data / "labels" / "intent"
        assert sorted(p.name for p in labels.iterdir()) == [
            "haiku-s1.cost.json",
            "haiku-s1.jsonl",
        ]
        assert [r for r in conf.builder.frame["id"]] == ["i0", "i1", "i2", "i3"]

    def test_load_defaults_to_dev_and_names_the_missing_split(self, conf):
        assert label.load("intent")["id"].tolist() == ["i0", "i1", "i2", "i3"]
        (conf.data / "intent" / "confirm.jsonl").unlink()
        with pytest.raises(SystemExit, match="confirm.jsonl missing"):
            label.load("intent", "confirm")


class TestWhatItCalls:
    def test_the_route_model_prices_and_haiku_params_are_the_samples(self, conf):
        # Catches a drift from the pre-registered route: OpenRouter Haiku at
        # temperature 1.0, low reasoning, explicit prices, concurrency 6, skip.
        confirm()

        assert conf.builder.last["llm"] == {
            "model": "openrouter/anthropic/claude-haiku-5-5",
            "temperature": 1.0,
            "reasoning_effort": "low",
            "input_cost_per_1k_tokens": label.OPENROUTER_PRICES[
                "input_cost_per_1k_tokens"
            ],
            "output_cost_per_1k_tokens": label.OPENROUTER_PRICES[
                "output_cost_per_1k_tokens"
            ],
        }
        assert conf.builder.last["concurrency"] == 6
        assert conf.builder.last["policy"] == "skip"

    def test_the_cost_file_records_the_confirm_provenance(self, conf):
        confirm(sample="4")

        cost = json.loads(
            (
                conf.data / "labels" / "intent" / "confirm" / "haiku-s4.cost.json"
            ).read_text()
        )
        assert cost["route"] == "openrouter"
        assert cost["model"] == "openrouter/anthropic/claude-haiku-5-5"
        assert cost["split"] == "confirm"
        assert cost["rows"] == 5
        assert cost["source"] == "label.py --sample 4 (result.costs)"

    def test_a_response_cache_is_refused_before_any_call(self, conf, monkeypatch):
        # Catches a cache faking unanimous samples on the confirm rows too.
        monkeypatch.setattr(conf.builder, "cache_config", {"type": "redis"})

        with pytest.raises(SystemExit, match="response cache"):
            confirm()

        assert conf.builder.executed == 0
        assert not (conf.data / "labels" / "intent" / "confirm").exists()


class TestRefusals:
    @pytest.mark.parametrize(
        "args",
        [
            ["--split", "confirm"],  # no --sample
            ["--split", "confirm", "--sample", "0"],
            ["--split", "confirm", "--sample", "6"],
            ["--split", "confirm", "--sample", "1", "--route", "direct"],
            ["--split", "confirm", "--sample", "1", "--candidates"],
            ["--split", "confirm", "--sample", "1", "--total-budget", "5"],
            ["--split", "confirm", "--sample", "1", "--budget", "0.5"],
        ],
    )
    def test_anything_but_the_preregistered_run_exits_before_any_pipeline(
        self, conf, args
    ):
        # Catches a confirm run on a wrong route, a raised cap or an unlisted index.
        with pytest.raises(SystemExit):
            run(*args)

        assert conf.builder.created == 0
        assert list((conf.data / "labels" / "intent").iterdir()) == []

    def test_confirm_needs_haiku(self, conf):
        # Catches a Sol, Luna or Opus call on confirm rows (design section 8).
        with pytest.raises(SystemExit, match="--model haiku"):
            label.main(
                [
                    "--task",
                    "intent",
                    "--model",
                    "sol",
                    "--split",
                    "confirm",
                    "--sample",
                    "1",
                ]
            )

        assert conf.builder.created == 0

    def test_attributes_has_no_confirm_rows(self, conf):
        with pytest.raises(SystemExit, match="covers intent, faithfulness"):
            run("--split", "confirm", "--sample", "1", task="attributes")

    def test_a_sample_is_never_overwritten(self, conf):
        confirm()
        before = (
            conf.data / "labels" / "intent" / "confirm" / "haiku-s1.jsonl"
        ).read_bytes()

        with pytest.raises(SystemExit, match="never overwritten"):
            confirm()

        assert conf.builder.created == 1
        assert (
            conf.data / "labels" / "intent" / "confirm" / "haiku-s1.jsonl"
        ).read_bytes() == before

    def test_the_dev_file_of_the_same_index_does_not_block_the_confirm_one(self, conf):
        run("--sample", "1")

        confirm()

        assert conf.builder.executed == 2


class TestBudgets:
    @pytest.mark.parametrize(
        ("task", "args", "budget"),
        [
            ("intent", [], 0.14),
            ("faithfulness", [], 0.09),
            ("intent", ["--limit", "2"], 0.01),
            ("faithfulness", ["--limit", "2"], 0.01),
            ("intent", ["--budget", "0.05"], 0.05),
            ("faithfulness", ["--budget", "0.09"], 0.09),
        ],
    )
    def test_the_default_per_run_budget_and_a_lower_explicit_one(
        self, conf, task, args, budget
    ):
        # Catches a changed per-run cap: intent 0.14, faithfulness 0.09, smoke 0.01.
        confirm(*args, task=task)

        assert conf.builder.last["budget"] == budget

    @pytest.mark.parametrize(
        ("task", "args"),
        [
            ("intent", ["--budget", "0.15"]),
            ("faithfulness", ["--budget", "0.10"]),
            ("intent", ["--limit", "2", "--budget", "0.02"]),
        ],
    )
    def test_a_budget_above_the_default_is_refused_before_any_call(
        self, conf, task, args
    ):
        # Catches --budget raising a cap the design fixed.
        with pytest.raises(SystemExit, match="above the confirm default"):
            confirm(*args, task=task)

        assert conf.builder.created == 0

    def test_the_spend_counts_smoke_aborted_and_both_tasks_but_not_dev_files(
        self, conf
    ):
        # Catches an undercount: smoke and aborted runs are real spend.
        write_cost(conf.data, "intent", "haiku-s1", 0.12)
        write_cost(conf.data, "faithfulness", "haiku-s1", 0.07)
        write_cost(conf.data, "intent", "haiku-s2-smoke3", 0.004)
        write_cost(conf.data, "intent", "haiku-s2-aborted", 0.14)
        write_cost(conf.data, "intent", "haiku-s1", 9.0, folder="")  # a dev sample
        write_cost(conf.data, "intent", "haiku", 9.0, folder="")  # the original run

        assert label.confirm_spend(conf.data) == pytest.approx(0.334)

    def test_the_dev_sample_cap_ignores_confirm_files(self, conf):
        # Catches sample_spend's glob descending into confirm/: the two caps are separate.
        write_cost(conf.data, "intent", "haiku-s1", 0.90)
        write_cost(conf.data, "intent", "haiku-s2", 0.04, folder="")

        assert label.sample_spend(conf.data) == pytest.approx(0.04)

    def test_the_total_cap_refuses_before_any_call(self, conf):
        # Catches a cap checked after the call: 0.90 spent + 0.14 > 1.00.
        write_cost(conf.data, "intent", "haiku-s1", 0.50)
        write_cost(conf.data, "faithfulness", "haiku-s1", 0.40)

        with pytest.raises(SystemExit, match="total budget"):
            confirm(sample="2")

        assert conf.builder.created == 0
        assert conf.builder.executed == 0
        assert not (
            conf.data / "labels" / "intent" / "confirm" / "haiku-s2.jsonl"
        ).exists()

    def test_the_cap_is_exact_at_the_boundary(self, conf):
        # 0.86 + 0.14 = 1.00 is allowed; one cent more is not.
        write_cost(conf.data, "intent", "haiku-s1", 0.86)
        confirm(sample="2")
        assert conf.builder.executed == 1

        write_cost(conf.data, "intent", "haiku-s3", 0.0001)
        with pytest.raises(SystemExit, match="total budget"):
            confirm(sample="4")

    def test_a_smoke_run_still_counts_toward_the_cap(self, conf):
        # Catches a smoke run that is free: 0.995 + 0.01 > 1.00.
        write_cost(conf.data, "intent", "haiku-s1", 0.995)

        with pytest.raises(SystemExit, match="total budget"):
            confirm("--limit", "2")

        assert conf.builder.created == 0

    def test_an_aborted_run_is_charged_its_whole_budget_inside_confirm(
        self, conf, monkeypatch
    ):
        # Catches spend that escapes the cap when a run dies before its cost file.
        def boom(self):
            raise RuntimeError("provider down")

        monkeypatch.setattr(FakePipeline, "execute", boom)
        with pytest.raises(RuntimeError):
            confirm(task="faithfulness")

        folder = conf.data / "labels" / "faithfulness" / "confirm"
        assert not (folder / "haiku-s1.jsonl").exists()  # the slot can be retried
        assert label.confirm_spend(conf.data) == pytest.approx(0.09)
        assert label.sample_spend(conf.data) == 0.0

    def test_a_real_run_adds_its_measured_cost_to_the_spend(self, conf):
        confirm()

        assert label.confirm_spend(conf.data) == pytest.approx(0.0123)


class TestHardCap:
    """The $1.00 cap holds for aborted, repeated and late-crashing runs."""

    def crash_in_call(self, monkeypatch):
        def boom(self):
            raise RuntimeError("provider down mid-run")

        monkeypatch.setattr(FakePipeline, "execute", boom)

    def confirm_files(self, conf, task="intent"):
        return sorted(
            p.name for p in (conf.data / "labels" / task / "confirm").iterdir()
        )

    def test_two_aborts_of_one_slot_are_charged_twice_and_leave_two_records(
        self, conf, monkeypatch
    ):
        # Catches the reviewer's M1: a fixed "-aborted" name let a second abort of
        # the same slot overwrite the first, so the cap was not hard.
        self.crash_in_call(monkeypatch)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                confirm()

        assert label.confirm_spend(conf.data) == pytest.approx(2 * 0.14)
        assert len(self.confirm_files(conf)) == 2
        assert len(label.confirm_holds(conf.data)) == 2

    def test_a_third_abort_is_refused_by_the_cap_before_any_call(
        self, conf, monkeypatch
    ):
        # 7 x 0.14 = 0.98; the eighth run would reach 1.12: refused, nothing called.
        self.crash_in_call(monkeypatch)
        for _ in range(7):
            with pytest.raises(RuntimeError):
                confirm()
        made = conf.builder.created

        with pytest.raises(SystemExit, match="total budget"):
            confirm()

        assert conf.builder.created == made
        assert label.confirm_spend(conf.data) == pytest.approx(0.98)

    def test_a_crash_after_execute_returns_is_charged_one_budget_and_keeps_the_hold(
        self, conf, monkeypatch
    ):
        # Catches the reviewer's M2: a crash between execute() and the cost file
        # recorded $0 and left the slot free for an unrecorded retry.
        real = FakePipeline.execute

        def late_crash(self):
            result = real(self)
            result.to_pandas = lambda: (_ for _ in ()).throw(
                RuntimeError("frame build failed")
            )
            return result

        monkeypatch.setattr(FakePipeline, "execute", late_crash)
        with pytest.raises(RuntimeError, match="frame build failed"):
            confirm()

        assert label.confirm_spend(conf.data) == pytest.approx(0.14)
        holds = label.confirm_holds(conf.data)
        assert len(holds) == 1
        assert json.loads(holds[0].read_text()) == {
            "cost": 0.14,
            "reserved": True,
            "rows": 0,
            "split": "confirm",
        }
        assert not any(
            n.endswith(".cost.json") and "hold" not in n
            for n in self.confirm_files(conf)
        )

    def test_a_retry_after_a_late_crash_counts_both_runs(self, conf, monkeypatch):
        real = FakePipeline.execute

        def late_crash(self):
            result = real(self)
            result.to_pandas = lambda: (_ for _ in ()).throw(RuntimeError("x"))
            return result

        with monkeypatch.context() as m:
            m.setattr(FakePipeline, "execute", late_crash)
            with pytest.raises(RuntimeError):
                confirm()
        confirm()

        assert label.confirm_spend(conf.data) == pytest.approx(0.14 + 0.0123)

    def test_a_normal_run_leaves_no_hold_and_spends_its_real_cost(self, conf):
        confirm()

        assert label.confirm_holds(conf.data) == []
        assert self.confirm_files(conf) == ["haiku-s1.cost.json", "haiku-s1.jsonl"]
        assert label.confirm_spend(conf.data) == pytest.approx(0.0123)

    def test_the_hold_exists_with_the_whole_budget_while_the_call_runs(
        self, conf, monkeypatch
    ):
        # Catches a hold written after the call: it must already count during it.
        seen = {}
        real = FakePipeline.execute

        def peek(self):
            seen["spend"] = label.confirm_spend(conf.data)
            seen["holds"] = len(label.confirm_holds(conf.data))
            return real(self)

        monkeypatch.setattr(FakePipeline, "execute", peek)
        confirm(task="faithfulness")

        assert seen == {"spend": pytest.approx(0.09), "holds": 1}

    def test_the_precheck_counts_holds_and_refuses_before_any_call(self, conf):
        label.reserve_confirm_budget(
            conf.data / "labels" / "intent" / "confirm" / "haiku-s1.jsonl", 0.90
        )

        with pytest.raises(SystemExit, match="hold included"):
            confirm(sample="2")

        assert conf.builder.created == 0

    def test_holds_never_overwrite_each_other_even_with_a_frozen_clock(
        self, conf, monkeypatch
    ):
        monkeypatch.setattr(label.time, "time_ns", lambda: 42)
        dest = conf.data / "labels" / "intent" / "confirm" / "haiku-s1.jsonl"

        a = label.reserve_confirm_budget(dest, 0.1)
        b = label.reserve_confirm_budget(dest, 0.1)

        assert a != b
        assert a.exists()
        assert b.exists()

    def test_a_cache_refusal_leaves_no_hold(self, conf, monkeypatch):
        monkeypatch.setattr(conf.builder, "cache_config", {"type": "redis"})

        with pytest.raises(SystemExit, match="response cache"):
            confirm()

        assert label.confirm_holds(conf.data) == []

    def test_the_dev_sample_path_writes_no_hold_and_keeps_its_aborted_name(
        self, conf, monkeypatch
    ):
        # The dev behaviour is unchanged: the fixed -aborted file, no hold.
        self.crash_in_call(monkeypatch)
        with pytest.raises(RuntimeError):
            run("--sample", "1")

        names = sorted(p.name for p in (conf.data / "labels" / "intent").iterdir())
        assert names == ["haiku-s1-aborted.cost.json"]
        assert label.confirm_holds(conf.data) == []

    def test_the_refusal_names_the_confirm_files(self, conf):
        write_cost(conf.data, "intent", "haiku-s1", 0.95)

        with pytest.raises(SystemExit, match=r"labels/\*/confirm/ cost files"):
            confirm(sample="2")


class TestReplacementIndex:
    def test_sample_5_is_refused_while_no_primary_file_exists(self, conf):
        # Catches a replacement drawn out of nothing: s5 only follows an s1-s4.
        with pytest.raises(SystemExit, match="single replacement"):
            confirm(sample="5")

        assert conf.builder.created == 0

    def test_a_smoke_file_does_not_count_as_a_primary_file(self, conf):
        confirm("--limit", "2")

        with pytest.raises(SystemExit, match="single replacement"):
            confirm(sample="5")

    def test_sample_5_runs_once_a_primary_file_exists_for_that_task(self, conf):
        confirm(sample="2")

        confirm(sample="5")
        with pytest.raises(SystemExit, match="single replacement"):
            confirm(sample="5", task="faithfulness")  # another task: still none

        assert (conf.data / "labels" / "intent" / "confirm" / "haiku-s5.jsonl").exists()
