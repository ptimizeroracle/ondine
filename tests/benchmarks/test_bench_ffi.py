"""FFI boundary benchmarks — per-row vs bulk path.

Measures the cost of per-row Python<->Rust crossings for evidence and
knowledge-store bulk ingest. Baselines guided the optimization gate:
the in-memory per-row path was ~320ms for 10K chunks; the on-disk path
was ~1.7s. A single-transaction bulk API with cached prepared statements
delivers ~7.5x on disk and ~1.5x in memory, and this file retains the
baselines so regressions are visible.

Arrow zero-copy was evaluated and skipped: once the per-row overhead is
eliminated, the residual cost is dominated by SQLite inserts, not by
PyString→String conversion, so the added complexity was not justified
(see design RFC for the full reasoning).
"""

from __future__ import annotations

import json
import uuid

import pytest

pytest.importorskip("pytest_benchmark")

from ondine import _engine


def _make_chunks(n: int) -> list[tuple[str, str, str, str]]:
    """Generate N (chunk_id, text, source, metadata_json) tuples."""
    return [
        (
            str(uuid.uuid4()),
            f"Chunk body number {i}. " + ("lorem ipsum " * 20),
            f"doc_{i % 100}.md",
            json.dumps({"page": i % 50, "idx": i}),
        )
        for i in range(n)
    ]


@pytest.mark.benchmark(group="ffi-bulk-ingest")
def test_bench_store_chunk_10k_baseline(benchmark):
    """Baseline: 10K sequential store_chunk calls (JSON + N FFI crossings)."""
    chunks = _make_chunks(10_000)

    def _run():
        db = _engine.EvidenceDB(":memory:")
        for cid, text, source, meta in chunks:
            db.store_chunk(cid, text, source, meta)
        return db.chunk_count()

    result = benchmark(_run)
    assert result == 10_000


@pytest.mark.benchmark(group="ffi-bulk-ingest")
def test_bench_store_chunk_1k_baseline(benchmark):
    """Smaller baseline for quick iteration (1K chunks)."""
    chunks = _make_chunks(1_000)

    def _run():
        db = _engine.EvidenceDB(":memory:")
        for cid, text, source, meta in chunks:
            db.store_chunk(cid, text, source, meta)
        return db.chunk_count()

    result = benchmark(_run)
    assert result == 1_000


@pytest.mark.benchmark(group="ffi-bulk-ingest")
def test_bench_store_chunks_batch_10k(benchmark):
    """A1: bulk path — single FFI call, one txn, cached prepared stmts."""
    chunks = _make_chunks(10_000)

    def _run():
        db = _engine.EvidenceDB(":memory:")
        db.store_chunks_batch(chunks)
        return db.chunk_count()

    result = benchmark(_run)
    assert result == 10_000


@pytest.mark.benchmark(group="ffi-bulk-ingest")
def test_bench_store_chunks_batch_1k(benchmark):
    chunks = _make_chunks(1_000)

    def _run():
        db = _engine.EvidenceDB(":memory:")
        db.store_chunks_batch(chunks)
        return db.chunk_count()

    result = benchmark(_run)
    assert result == 1_000


@pytest.mark.benchmark(group="ffi-ondisk-ingest")
def test_bench_store_chunk_10k_ondisk_baseline(benchmark, tmp_path):
    chunks = _make_chunks(10_000)

    def _run():
        db_path = str(tmp_path / f"baseline_{uuid.uuid4().hex}.db")
        db = _engine.EvidenceDB(db_path)
        for cid, text, source, meta in chunks:
            db.store_chunk(cid, text, source, meta)
        return db.chunk_count()

    result = benchmark.pedantic(_run, rounds=3, iterations=1)
    assert result == 10_000


@pytest.mark.benchmark(group="ffi-ondisk-ingest")
def test_bench_store_chunks_batch_10k_ondisk(benchmark, tmp_path):
    chunks = _make_chunks(10_000)

    def _run():
        db_path = str(tmp_path / f"batch_{uuid.uuid4().hex}.db")
        db = _engine.EvidenceDB(db_path)
        db.store_chunks_batch(chunks)
        return db.chunk_count()

    result = benchmark.pedantic(_run, rounds=3, iterations=1)
    assert result == 10_000


def test_bulk_ingest_meaningfully_faster_than_single_ondisk(tmp_path):
    """Regression gate: on disk the bulk path must stay materially
    faster than the per-row path for 10K chunks.

    Wall-clock benchmarks are noisy in shared-runner CI, so the gate is
    intentionally conservative: we warm OS/SQLite caches with an
    untimed run per path, take the minimum of ``runs`` timed runs (best
    reflects the fast path, least sensitive to noisy neighbours), and
    require only a 3x speedup. Local measurement shows ~7.5x; a 3x
    floor still catches an accidental loss of the single-transaction
    optimisation without flaking on slow CI.
    """
    import time

    chunks = _make_chunks(10_000)
    rows = len(chunks)
    runs = 3

    def _time_single() -> float:
        path = str(tmp_path / f"single_{uuid.uuid4().hex}.db")
        db = _engine.EvidenceDB(path)
        start = time.perf_counter()
        for cid, text, source, meta in chunks:
            db.store_chunk(cid, text, source, meta)
        elapsed = time.perf_counter() - start
        assert db.chunk_count() == rows
        return elapsed

    def _time_bulk() -> float:
        path = str(tmp_path / f"bulk_{uuid.uuid4().hex}.db")
        db = _engine.EvidenceDB(path)
        start = time.perf_counter()
        db.store_chunks_batch(chunks)
        elapsed = time.perf_counter() - start
        assert db.chunk_count() == rows
        return elapsed

    # Warm caches (untimed).
    _time_single()
    _time_bulk()

    # Alternate paths across rounds to share noise symmetrically.
    single_times: list[float] = []
    bulk_times: list[float] = []
    for i in range(runs):
        if i % 2 == 0:
            single_times.append(_time_single())
            bulk_times.append(_time_bulk())
        else:
            bulk_times.append(_time_bulk())
            single_times.append(_time_single())

    single_best = min(single_times)
    bulk_best = min(bulk_times)
    speedup = single_best / bulk_best
    assert speedup >= 3.0, (
        f"bulk ingest speedup regressed: {speedup:.2f}x "
        f"(single_best={single_best:.3f}s, bulk_best={bulk_best:.3f}s, "
        f"single_all={[round(t, 3) for t in single_times]}, "
        f"bulk_all={[round(t, 3) for t in bulk_times]})"
    )


@pytest.mark.benchmark(group="ffi-bulk-query")
def test_bench_query_chunks_limit_100_baseline(benchmark):
    """Query returning up to 100 results (JSON serialize + parse)."""
    db = _engine.EvidenceDB(":memory:")
    for cid, text, source, meta in _make_chunks(500):
        db.store_chunk(cid, text, source, meta)

    def _run():
        raw = db.query_chunks("lorem ipsum chunk body", 100)
        return json.loads(raw)

    results = benchmark(_run)
    assert isinstance(results, list)
