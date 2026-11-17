# ruff: noqa: S311 - seeded random.Random is what the builder takes; nothing here is cryptographic
"""The confirmation split of benchmarks/golden/build.py (review-next DESIGN-confirm.md).

The pre-registered study needs rows nobody has seen, drawn without moving a
single dev or second-split row. These tests pin that: the earlier splits
rebuild byte for byte, the confirm draw is deterministic and obeys its quotas,
and ``--confirm`` writes one file and two manifest entries and nothing else.
Synthetic sources stand in for the downloads; the one test that needs the real
sources is skipped when they are absent and only compares hashes in memory.
Each test names the regression it catches.
"""

from __future__ import annotations

import importlib.util
import json
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

_BUILD = Path(__file__).resolve().parents[2] / "benchmarks" / "golden" / "build.py"
_spec = importlib.util.spec_from_file_location("golden_build_confirm", _BUILD)
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)

REAL_DATA = _BUILD.parent / "data"
REAL_MANIFEST = _BUILD.parent / "manifest.json"
TASKS = ("intent", "faithfulness")


def _real_sources_present() -> bool:
    return all(
        (REAL_DATA / "sources" / (n + Path(build.SOURCES[n]["url"]).suffix)).exists()
        for t in TASKS
        for n in build.CONFIRM_SOURCES[t]
    )


# ── synthetic sources ────────────────────────────────────────────────────

DOMAINS = sorted(build.SUMMEDITS_DOMAINS)


def write_summedits(sources: Path, docs: int = 20, per_label: int = 10) -> None:
    """One SummEdits-shaped file per domain: `docs` documents, 2 * per_label rows each."""
    sources.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        rows = [
            {
                "id": f"{d}-{k}",
                "doc": f"{domain}-doc{d}",
                "summary": f"summary {d} {k}",
                "label": k % 2,
            }
            for d in range(docs)
            for k in range(2 * per_label)
        ]
        (sources / f"summedits_{domain}.json").write_text(json.dumps(rows))


def write_banking(sources: Path, classes: int = 5, per_class: int = 24) -> None:
    sources.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {"text": f"msg {c} {k}", "label_text": f"class_{c:02d}"}
            for c in range(classes)
            for k in range(per_class)
        ]
    )
    frame.to_parquet(sources / "banking77_test.parquet")


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """Synthetic sources under tmp_path/data, pinned and pointed at by build.py."""
    data = tmp_path / "data"
    write_summedits(data / "sources")
    write_banking(data / "sources")
    monkeypatch.setattr(build, "DATA", data)
    monkeypatch.setattr(build, "SOURCES_DIR", data / "sources")
    # A scratch copy of the registry, so the pinned real hashes stay untouched.
    sources = {k: dict(v) for k, v in build.SOURCES.items()}
    monkeypatch.setattr(build, "SOURCES", sources)
    for name in (n for t in TASKS for n in build.CONFIRM_SOURCES[t]):
        suffix = Path(sources[name]["url"]).suffix
        sources[name]["sha256"] = build.sha256_prefix(
            data / "sources" / (name + suffix)
        )
    # The pool must hold the 20/20 quota under the cap: 8 dev docs here.
    return data


def make_manifest(path: Path) -> dict:
    """What a normal build would have recorded for the synthetic sources."""
    manifest: dict = {
        "seed": build.SEED,
        "canary": build.CANARY,
        "sources": {},
        "tasks": {},
        "built_at": "2026-10-24T20:54:29+00:00",
    }
    for task in TASKS:
        built = build.BUILDERS[task](random.Random(f"{build.SEED}:{task}"))
        dev = [{**r, "canary": build.CANARY} for r in built["dev"]]
        strata = built["strata"]
        manifest["tasks"][task] = {
            "hashes": {
                "dev": build.rows_sha256(dev),
                "holdout": build.rows_sha256(built["holdout"]),
            },
            "counts": {
                "dev": {
                    "rows": len(dev),
                    "by_" + strata: dict(sorted(build._count(dev, strata).items())),
                },
                "holdout": {"rows": len(built["holdout"])},
            },
        }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def built(task: str, synthetic) -> dict:
    return build.BUILDERS[task](random.Random(f"{build.SEED}:{task}"))


def draw(task: str, seed: str | None = None, pool: dict | None = None) -> list[dict]:
    pool = pool if pool is not None else built(task, None)["confirm_pool"]
    return build.build_confirm(
        task, pool, random.Random(seed or build.CONFIRM_RNG[task])
    )


# ── byte identity: the earlier splits must not move ──────────────────────


@pytest.mark.skipif(
    not (_real_sources_present() and REAL_MANIFEST.exists()),
    reason="real sources absent",
)
@pytest.mark.parametrize("task", TASKS)
def test_dev_and_the_second_split_rebuild_to_the_manifest_hashes(task):
    # Catches a changed rng order or pool carve in build_intent / build_faithfulness:
    # dev.jsonl and the release split would no longer be the files the manifest
    # and every earlier study point at. Only hashes are compared, in memory.
    manifest = json.loads(REAL_MANIFEST.read_text())["tasks"][task]["hashes"]

    got = build.BUILDERS[task](random.Random(f"{build.SEED}:{task}"))

    dev = [{**r, "canary": build.CANARY} for r in got["dev"]]
    assert build.rows_sha256(dev) == manifest["dev"]
    assert build.rows_sha256(got["holdout"]) == manifest["holdout"]


@pytest.mark.skipif(not _real_sources_present(), reason="real sources absent")
def test_the_real_confirm_draw_meets_its_quotas_and_the_disclosed_pools():
    # Catches a pool carve that differs from DESIGN-confirm.md section 0 (66
    # documents, 1,756 unused rows: 579 consistent and 1,177 inconsistent) and a
    # draw that misses 20/20 per domain, 10 per class or the cap of 5.
    faith = build.build_faithfulness(random.Random(f"{build.SEED}:faithfulness"))
    pool = faith["confirm_pool"]
    labels = Counter(r["label"] for rows in pool.values() for r in rows)
    assert sum(labels.values()) == 1756
    assert (labels["consistent"], labels["inconsistent"]) == (579, 1177)
    assert len({r["doc"] for rows in pool.values() for r in rows}) == 66
    rows = draw("faithfulness", pool=pool)
    assert len(rows) == 320
    assert set(Counter((r["domain"], r["label"]) for r in rows).values()) == {20}
    assert max(Counter((r["doc"], r["label"]) for r in rows).values()) <= 5
    held_docs = {r["doc"] for r in faith["holdout"]}
    assert not {r["doc"] for r in rows} & held_docs
    intent = build.build_intent(random.Random(f"{build.SEED}:intent"))
    assert {len(v) for v in intent["confirm_pool"].values()} <= {32, 33}
    chosen = draw("intent", pool=intent["confirm_pool"])
    assert len(chosen) == 770
    assert set(Counter(r["label"] for r in chosen).values()) == {10}


# ── the draw, on synthetic sources ───────────────────────────────────────


def legacy_faithfulness(rng: random.Random) -> tuple[list[dict], list[dict]]:
    """build_faithfulness as it was before --confirm: the oracle for 'no new rng call'."""
    dev: list[dict] = []
    holdout: list[dict] = []
    for domain in sorted(build.SUMMEDITS_DOMAINS):
        rows = json.loads(
            build.fetch(f"summedits_{domain}").read_text(encoding="utf-8")
        )
        by_doc: dict[str, list[dict]] = {}
        for r in rows:
            by_doc.setdefault(r["doc"], []).append(
                {
                    "id": f"se-{domain}-{r['id']}",
                    "domain": domain,
                    "doc": r["doc"],
                    "summary": r["summary"],
                    "label": "consistent" if str(r["label"]) == "1" else "inconsistent",
                }
            )
        docs = sorted(by_doc)
        rng.shuffle(docs)
        cut = max(1, round(len(docs) * 0.4))
        dev.extend(build._balanced(by_doc, docs[:cut], 25, rng))
        holdout.extend(build._balanced(by_doc, docs[cut:], 38, rng))
    return dev, holdout


class TestEarlierSplitsDoNotMove:
    def test_faithfulness_dev_and_second_split_equal_the_pre_confirm_builder(
        self, synthetic
    ):
        # Catches a new rng call (or a reordered one) added to build_faithfulness
        # while carving the confirm pool: later draws would shift.
        dev, holdout = legacy_faithfulness(random.Random(f"{build.SEED}:faithfulness"))

        got = built("faithfulness", synthetic)

        assert got["dev"] == dev
        assert got["holdout"] == holdout

    def test_intent_makes_the_same_takes_and_returns_the_leftover(self, synthetic):
        # Catches a confirm_pool that is not what both takes left: a class loses
        # 3 + 4 rows and the pool must hold the other 17 of the 24.
        got = built("intent", synthetic)

        assert all(len(v) == 24 - 3 - 4 for v in got["confirm_pool"].values())
        taken = {r["id"] for r in got["dev"] + got["holdout"]}
        assert not taken & {r["id"] for v in got["confirm_pool"].values() for r in v}

    def test_the_pools_are_not_written_by_main_as_extra_files(self, synthetic):
        # Catches a pool placed in "extra": main writes every "extra" item to disk.
        for task in TASKS:
            assert "confirm_pool" not in built(task, synthetic)["extra"]
            assert "confirm_pool" in built(task, synthetic)


class TestFaithfulnessDraw:
    def test_it_is_deterministic(self, synthetic):
        # Catches a draw that depends on set or dict order, or on the clock.
        assert draw("faithfulness") == draw("faithfulness")

    def test_its_documents_are_dev_pool_documents_and_not_the_second_splits(
        self, synthetic
    ):
        # Catches a pool built from the wrong side of the 40/60 document cut:
        # confirm documents must come from the dev pool, never the release pool.
        got = built("faithfulness", synthetic)
        rows = draw("faithfulness")

        pool_docs = {r["doc"] for v in got["confirm_pool"].values() for r in v}
        held_docs = {r["doc"] for r in got["holdout"]}
        assert {r["doc"] for r in rows} <= pool_docs
        assert not pool_docs & held_docs
        assert not {r["doc"] for r in rows} & held_docs

    def test_no_row_is_a_dev_row_or_a_second_split_row(self, synthetic):
        # Catches a pool that still contains the rows dev took.
        got = built("faithfulness", synthetic)
        rows = draw("faithfulness")

        earlier = {r["id"] for r in got["dev"]} | {r["id"] for r in got["holdout"]}
        assert not {r["id"] for r in rows} & earlier
        assert len({r["id"] for r in rows}) == len(rows)

    def test_every_domain_has_20_consistent_and_20_inconsistent_rows(self, synthetic):
        # Catches a quota drift (the design: 20/20 per domain, 320 rows).
        rows = draw("faithfulness")

        assert len(rows) == 320
        counts = Counter((r["domain"], r["label"]) for r in rows)
        assert set(counts.values()) == {20}
        assert len(counts) == 16

    def test_no_document_gives_more_than_5_rows_of_one_label(self, synthetic):
        # Catches a missing cap: one long-document domain would fill with it.
        rows = draw("faithfulness")

        assert max(Counter((r["doc"], r["label"]) for r in rows).values()) <= 5

    def test_the_draw_is_round_robin_over_documents_not_first_come(self, synthetic):
        # Catches a draw that fills document by document: with 8 dev documents
        # and a quota of 20 the round-robin touches every document, 2 or 3 rows each.
        rows = [r for r in draw("faithfulness") if r["domain"] == "news"]

        per_doc = Counter(r["doc"] for r in rows if r["label"] == "consistent")
        assert len(per_doc) == 8
        assert set(per_doc.values()) <= {2, 3}

    def test_an_unmet_quota_stops_the_build_instead_of_shrinking_it(self):
        # Catches a silent shortfall: 3 documents x cap 5 = 15 < 20.
        rows = [
            {
                "id": f"x{d}-{k}",
                "doc": f"d{d}",
                "domain": "news",
                "summary": "s",
                "label": ["consistent", "inconsistent"][k % 2],
            }
            for d in range(3)
            for k in range(40)
        ]

        with pytest.raises(AssertionError, match="news/consistent"):
            build.build_confirm("faithfulness", {"news": rows}, random.Random(0))

    def test_four_documents_at_the_cap_make_exactly_the_quota(self):
        # Catches an off-by-one in the cap: 4 documents x 5 = 20 is enough.
        rows = [
            {
                "id": f"x{d}-{k}",
                "doc": f"d{d}",
                "domain": "news",
                "summary": "s",
                "label": ["consistent", "inconsistent"][k % 2],
            }
            for d in range(4)
            for k in range(40)
        ]

        out = build.build_confirm("faithfulness", {"news": rows}, random.Random(0))

        assert Counter(r["doc"] for r in out if r["label"] == "consistent") == {
            f"d{d}": 5 for d in range(4)
        }

    def test_the_input_pool_order_does_not_change_the_draw(self, synthetic):
        # Catches a draw that depends on the order the builder happened to leave.
        pool = built("faithfulness", synthetic)["confirm_pool"]
        shuffled = {k: random.Random(5).sample(v, len(v)) for k, v in pool.items()}

        assert draw("faithfulness", pool=shuffled) == draw("faithfulness", pool=pool)

    def test_a_different_confirm_seed_changes_confirm_rows_but_not_dev(self, synthetic):
        # Catches a confirm draw that shares the dev generator: changing one
        # must not move the other.
        first = built("faithfulness", synthetic)
        other = draw("faithfulness", seed="another-seed")

        assert {r["id"] for r in other} != {r["id"] for r in draw("faithfulness")}
        assert built("faithfulness", synthetic)["dev"] == first["dev"]

    def test_rows_carry_the_dev_canary_and_keep_their_fields(self, synthetic):
        # Catches a confirm file without the leak canary (the dev rule).
        rows = draw("faithfulness")

        assert {r["canary"] for r in rows} == {build.CANARY}
        assert {"id", "domain", "doc", "summary", "label"} <= set(rows[0])


class TestIntentDraw:
    def test_ten_rows_per_class_from_the_leftover_only(self, synthetic):
        # Catches a pool that includes dev or second-split rows, or a wrong quota.
        got = built("intent", synthetic)
        rows = draw("intent")

        assert set(Counter(r["label"] for r in rows).values()) == {10}
        taken = {r["id"] for r in got["dev"] + got["holdout"]}
        assert not {r["id"] for r in rows} & taken

    def test_it_is_deterministic_and_has_its_own_generator(self, synthetic):
        # Catches a shared generator: another seed must pick other rows, same count.
        assert draw("intent") == draw("intent")
        other = draw("intent", seed="x")
        assert {r["id"] for r in other} != {r["id"] for r in draw("intent")}
        assert len(other) == len(draw("intent"))

    def test_the_pool_order_does_not_change_the_draw(self, synthetic):
        pool = built("intent", synthetic)["confirm_pool"]
        reordered = {k: list(reversed(v)) for k, v in pool.items()}

        assert draw("intent", pool=reordered) == draw("intent", pool=pool)

    def test_a_class_short_of_rows_stops_the_build(self):
        # Catches a silent shortfall: 9 rows cannot give 10.
        pool = {"c": [{"id": f"b{i}", "text": "t", "label": "c"} for i in range(9)]}

        with pytest.raises(AssertionError, match="only 9 left"):
            build.build_confirm("intent", pool, random.Random(0))

    def test_rows_carry_the_canary(self, synthetic):
        assert {r["canary"] for r in draw("intent")} == {build.CANARY}

    def test_a_task_without_a_confirm_split_is_refused(self):
        with pytest.raises(ValueError, match="attributes"):
            build.build_confirm("attributes", {}, random.Random(0))


def test_rows_sha256_is_the_hash_write_jsonl_returns(tmp_path):
    # Catches drift between the in-memory hash (used on the second split, which
    # must never be written) and the file hash the manifest holds.
    rows = [{"id": "a", "text": "é\n ü"}, {"id": "b", "n": 3}]

    assert build.rows_sha256(rows) == build.write_jsonl(tmp_path / "x.jsonl", rows)


# ── main --confirm ───────────────────────────────────────────────────────


def tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def scratch(synthetic, tmp_path):
    manifest = tmp_path / "manifest.json"
    make_manifest(manifest)
    return synthetic, manifest


def run(data: Path, manifest: Path, *extra: str) -> int:
    return build.main(
        ["--confirm", "--data-dir", str(data), "--manifest", str(manifest), *extra]
    )


class TestMainConfirm:
    def test_it_writes_only_the_confirm_files_and_the_manifest(self, scratch, capsys):
        # Catches a run that rewrites dev.jsonl, an adjudicated file, the second
        # split or an extra file: the real data dir must stay byte-identical.
        data, manifest = scratch
        before = tree(data)

        assert run(data, manifest) == 0

        after = tree(data)
        new = set(after) - set(before)
        assert new == {"faithfulness/confirm.jsonl", "intent/confirm.jsonl"}
        assert all(after[k] == before[k] for k in before)
        assert not any("holdout" in k for k in after)

    def test_it_never_prints_or_stores_a_second_split_row(self, scratch, capsys):
        # Catches a leak of the release split through stdout or any written file.
        data, manifest = scratch
        held = {
            r["id"]
            for t in TASKS
            for r in build.BUILDERS[t](random.Random(f"{build.SEED}:{t}"))["holdout"]
        }

        run(data, manifest)

        written = {
            json.loads(line)["id"]
            for t in TASKS
            for line in (data / t / "confirm.jsonl").read_text().splitlines()
        }
        assert not written & held
        # Whole-token search: "se-news-1-1" must not match inside "se-news-1-10".
        words = set(
            re.findall(r"[\w-]+", capsys.readouterr().out + manifest.read_text())
        )
        assert not words & held

    def test_only_the_confirm_keys_of_the_manifest_change(self, scratch):
        # Catches a rewrite of built_at, sources, seed or the old hashes.
        data, manifest = scratch
        old = json.loads(manifest.read_text())

        run(data, manifest)

        new = json.loads(manifest.read_text())
        for task in TASKS:
            assert set(new["tasks"][task]["hashes"]) - set(
                old["tasks"][task]["hashes"]
            ) == {"confirm"}
            assert set(new["tasks"][task]["counts"]) - set(
                old["tasks"][task]["counts"]
            ) == {"confirm"}
            del new["tasks"][task]["hashes"]["confirm"]
            del new["tasks"][task]["counts"]["confirm"]
        assert new == old
        assert new["built_at"] == "2026-10-24T20:54:29+00:00"

    def test_the_recorded_hash_is_the_files_hash_and_the_counts_match(self, scratch):
        data, manifest = scratch

        run(data, manifest)

        got = json.loads(manifest.read_text())["tasks"]
        for task in TASKS:
            path = data / task / "confirm.jsonl"
            assert got[task]["hashes"]["confirm"] == build.sha256_prefix(path)
            assert got[task]["counts"]["confirm"]["rows"] == len(
                path.read_text().splitlines()
            )
        assert got["faithfulness"]["counts"]["confirm"]["by_domain"] == dict.fromkeys(
            DOMAINS, 40
        )

    def test_a_changed_dev_hash_refuses_and_writes_nothing(self, scratch):
        # Catches a build whose dev rows no longer match the manifest: refuse, exit
        # nonzero, no file and no manifest change.
        data, manifest = scratch
        meta = json.loads(manifest.read_text())
        meta["tasks"]["faithfulness"]["hashes"]["dev"] = "0" * 16
        manifest.write_text(json.dumps(meta, indent=2) + "\n")
        before, mtext = tree(data), manifest.read_text()

        with pytest.raises(SystemExit, match="faithfulness: the rebuilt dev split"):
            run(data, manifest)

        assert tree(data) == before
        assert manifest.read_text() == mtext

    def test_a_changed_second_split_hash_refuses_and_writes_nothing(self, scratch):
        # The same for the release split, which is compared from memory.
        data, manifest = scratch
        meta = json.loads(manifest.read_text())
        meta["tasks"]["intent"]["hashes"]["holdout"] = "f" * 16
        manifest.write_text(json.dumps(meta, indent=2) + "\n")
        before = tree(data)

        with pytest.raises(SystemExit, match="rebuilt holdout split"):
            run(data, manifest)

        assert tree(data) == before

    def test_one_bad_task_stops_the_whole_run_before_any_write(self, scratch):
        # Catches a loop that writes the first task and refuses the second.
        data, manifest = scratch
        meta = json.loads(manifest.read_text())
        meta["tasks"]["intent"]["hashes"]["dev"] = "0" * 16  # intent runs second
        manifest.write_text(json.dumps(meta, indent=2) + "\n")

        with pytest.raises(SystemExit):
            run(data, manifest)

        assert not (data / "faithfulness" / "confirm.jsonl").exists()

    def test_it_never_downloads_a_missing_source(self, scratch, monkeypatch):
        # Catches a fetch() fallthrough: a scratch dir without sources must refuse.
        data, manifest = scratch
        (data / "sources" / "summedits_news.json").unlink()

        def no_network(*_a, **_k):
            raise AssertionError("--confirm tried to download a source")

        monkeypatch.setattr(urllib.request, "urlopen", no_network)

        with pytest.raises(SystemExit, match="never downloads"):
            run(data, manifest)

    def test_a_second_run_is_idempotent(self, scratch):
        # The reviewer re-runs it: same bytes out, same manifest.
        data, manifest = scratch
        run(data, manifest)
        files, text = tree(data), manifest.read_text()

        run(data, manifest)

        assert tree(data) == files
        assert manifest.read_text() == text

    def test_a_recorded_confirm_hash_that_the_draw_no_longer_reproduces_refuses(
        self, scratch
    ):
        # Catches a changed confirm recipe after the hash was committed.
        data, manifest = scratch
        run(data, manifest)
        meta = json.loads(manifest.read_text())
        meta["tasks"]["intent"]["hashes"]["confirm"] = "0" * 16
        manifest.write_text(json.dumps(meta, indent=2) + "\n")
        before = tree(data)

        with pytest.raises(SystemExit, match="not reproducible"):
            run(data, manifest)

        assert tree(data) == before

    def test_task_selects_one_task(self, scratch):
        data, manifest = scratch

        run(data, manifest, "--task", "faithfulness")

        assert (data / "faithfulness" / "confirm.jsonl").exists()
        assert not (data / "intent" / "confirm.jsonl").exists()
        assert (
            "confirm"
            not in json.loads(manifest.read_text())["tasks"]["intent"]["hashes"]
        )

    def test_attributes_has_no_confirm_split(self, scratch):
        data, manifest = scratch

        with pytest.raises(SystemExit):
            run(data, manifest, "--task", "attributes")

    def test_data_dir_and_manifest_are_only_for_confirm_and_come_together(
        self, scratch
    ):
        # Catches a scratch run that would overwrite the real manifest or data.
        data, manifest = scratch
        with pytest.raises(SystemExit):
            build.main(["--data-dir", str(data)])
        with pytest.raises(SystemExit):
            build.main(["--confirm", "--data-dir", str(data)])
        with pytest.raises(SystemExit):
            build.main(["--confirm", "--manifest", str(manifest)])
        assert not (data / "intent" / "dev.jsonl").exists()

    def test_the_real_manifest_round_trips_byte_for_byte_through_the_writer(self):
        # Catches a manifest writer that reformats the file: the only diff of the
        # real run must be the confirm entries.
        if not REAL_MANIFEST.exists():
            pytest.skip("no real manifest")
        text = REAL_MANIFEST.read_text()

        assert json.dumps(json.loads(text), indent=2, ensure_ascii=False) + "\n" == text


def test_the_confirm_recipe_constants_are_the_designs():
    # Catches a drifted generator string or quota: DESIGN-confirm.md section 1
    # fixes them, and a change would draw other rows than the ones pre-registered.
    assert build.CONFIRM_RNG == {
        "intent": "20261020:intent:confirm",
        "faithfulness": "20261020:faithfulness:confirm",
    }
    assert build.CONFIRM_INTENT_PER_CLASS == 10
    assert build.CONFIRM_FAITHFULNESS_PER_DOMAIN_LABEL == 20
    assert build.CONFIRM_FAITHFULNESS_CAP_PER_DOC_LABEL == 5


def test_the_cap_binds_when_documents_are_few():
    # 3 documents with a cap of 7 would give 21 >= 20 rows; the design's cap of 5
    # gives 15 and must stop the build. Catches a cap raised above the design's.
    rows = [
        {
            "id": f"x{d}-{k}",
            "doc": f"d{d}",
            "domain": "news",
            "summary": "s",
            "label": ["consistent", "inconsistent"][k % 2],
        }
        for d in range(3)
        for k in range(40)
    ]

    with pytest.raises(AssertionError, match="under the cap of 5"):
        build.build_confirm("faithfulness", {"news": rows}, random.Random(0))


def test_the_pool_never_reaches_the_extra_files():
    # Catches a pool moved into "extra" (main writes every extra item to a file).
    # Checked on the source: the builders' extra dicts hold the labels or nothing.
    src = _BUILD.read_text()

    assert '"extra": {"labels": labels},' in src
    assert '"extra": {},\n        "strata": "domain",' in src


def test_a_plain_rebuild_keeps_the_manifests_confirm_entries(
    scratch, monkeypatch, tmp_path
):
    # Catches a plain (non --confirm) build that rewrites tasks[task] and drops the
    # confirm hash and counts while confirm.jsonl is still on disk.
    data, manifest = scratch
    run(data, manifest)
    before = json.loads(manifest.read_text())
    home = tmp_path / "home"
    home.mkdir()
    (home / "manifest.json").write_text(manifest.read_text())
    monkeypatch.setattr(build, "HERE", home)

    assert build.main(["--task", "intent", "--task", "faithfulness"]) == 0

    after = json.loads((home / "manifest.json").read_text())
    for task in TASKS:
        assert (
            after["tasks"][task]["hashes"]["confirm"]
            == before["tasks"][task]["hashes"]["confirm"]
        )
        assert (
            after["tasks"][task]["counts"]["confirm"]
            == before["tasks"][task]["counts"]["confirm"]
        )
        assert (
            after["tasks"][task]["hashes"]["dev"]
            == before["tasks"][task]["hashes"]["dev"]
        )
