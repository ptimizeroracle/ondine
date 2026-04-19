"""Tests for ResponseCache — session-scoped, crash-safe LLM response store.

Every test pins a concrete regression that would silently cost the
user money (re-invoking the LLM for already-processed rows) or
corrupt resume semantics.

The SUT is ``SqliteResponseCache`` against a real on-disk SQLite
file. Never mocked — the whole point of this module is crash
atomicity, which only a real backend can prove.
"""

from __future__ import annotations

import os
import subprocess
import sys
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from ondine.adapters.response_cache import (
    CachedResponse,
    CachedRowMetadata,
    SqliteResponseCache,
)

# ── helpers ───────────────────────────────────────────────────────────


def _resp(text: str = "hi", cost: str = "0.001") -> CachedResponse:
    return CachedResponse(
        text=text,
        tokens_in=5,
        tokens_out=10,
        model="gpt-4o-mini",
        cost=Decimal(cost),
        latency_ms=42.0,
        metadata={"finish": "stop"},
    )


def _meta(row_index: int, row_id: str | None = None) -> CachedRowMetadata:
    return CachedRowMetadata(row_index=row_index, row_id=row_id, custom={})


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "responses.db"


@pytest.fixture
def cache(cache_path: Path) -> SqliteResponseCache:
    c = SqliteResponseCache(cache_path)
    yield c
    c.close()


# ── regression #1: resume re-invokes LLM for completed rows ───────────


def test_iter_completed_yields_appended_rows(cache: SqliteResponseCache) -> None:
    """Regression: if iter_completed drops rows, resume re-invokes
    the LLM for them — the user pays twice. This is the load-bearing
    property of the whole module."""
    session = uuid4()
    cache.append(session, 0, _resp("first"), _meta(0))
    cache.append(session, 1, _resp("second"), _meta(1))

    rows = list(cache.iter_completed(session))
    assert [r[0] for r in rows] == [0, 1]
    assert rows[0][1].text == "first"
    assert rows[1][1].text == "second"


def test_iter_completed_preserves_decimal_cost(cache: SqliteResponseCache) -> None:
    """Regression: cost must round-trip as Decimal, not float.
    Float conversion silently drops precision — accumulated across
    50k rows that drifts visible dollars off the billed total."""
    session = uuid4()
    cache.append(session, 0, _resp(cost="0.00042"), _meta(0))

    ((_, resp, _),) = list(cache.iter_completed(session))
    assert isinstance(resp.cost, Decimal)
    assert resp.cost == Decimal("0.00042")


# ── regression #2: ordering drives resume filter correctness ──────────


def test_iter_completed_returns_rows_ordered_by_row_index(
    cache: SqliteResponseCache,
) -> None:
    """Regression: rows can be appended out of order by concurrent
    workers (row 5 finishes before row 3). Resume must see them
    sorted by row_index so the output dataframe reassembles in the
    user's original row order."""
    session = uuid4()
    for idx in [5, 1, 9, 3, 0]:
        cache.append(session, idx, _resp(f"r{idx}"), _meta(idx))

    indices = [r[0] for r in cache.iter_completed(session)]
    assert indices == [0, 1, 3, 5, 9]


def test_last_processed_row_returns_max_index(cache: SqliteResponseCache) -> None:
    """Regression: the resume filter uses ``last_processed_row`` to
    decide what to skip. If it returns the *count* instead of the
    *max index*, sparse (out-of-order) appends miss rows and the
    LLM is re-invoked for already-done work."""
    session = uuid4()
    for idx in [5, 1, 9]:
        cache.append(session, idx, _resp(), _meta(idx))

    assert cache.last_processed_row(session) == 9


def test_last_processed_row_empty_session_returns_minus_one(
    cache: SqliteResponseCache,
) -> None:
    """Regression: fresh session must report -1 (sentinel for "no
    rows"). Returning 0 would cause the resume filter to skip row
    0 on a fresh run."""
    assert cache.last_processed_row(uuid4()) == -1


# ── regression #3: sessions must not leak across scopes ───────────────


def test_sessions_are_isolated(cache: SqliteResponseCache) -> None:
    """Regression: two concurrent pipelines sharing a cache directory
    would overwrite each other's completion history if scoping were
    broken. Verified by writing to session A and observing nothing
    from session B."""
    session_a = uuid4()
    session_b = uuid4()
    cache.append(session_a, 0, _resp("A-row-0"), _meta(0))

    assert list(cache.iter_completed(session_b)) == []
    assert cache.last_processed_row(session_b) == -1
    ((_, resp_a, _),) = list(cache.iter_completed(session_a))
    assert resp_a.text == "A-row-0"


def test_clear_is_session_scoped(cache: SqliteResponseCache) -> None:
    """Regression: ``clear(session_a)`` must not touch session_b.
    A naive ``DELETE FROM responses`` without a WHERE clause would
    nuke concurrent pipelines' state."""
    session_a = uuid4()
    session_b = uuid4()
    cache.append(session_a, 0, _resp(), _meta(0))
    cache.append(session_b, 0, _resp(), _meta(0))

    cache.clear(session_a)

    assert cache.last_processed_row(session_a) == -1
    assert cache.last_processed_row(session_b) == 0


def test_clear_is_idempotent(cache: SqliteResponseCache) -> None:
    """Regression: calling clear twice must not raise. Executor
    calls clear on both the success path and the cleanup path —
    double-clear happens in the wild."""
    session = uuid4()
    cache.clear(session)  # nothing there
    cache.clear(session)  # still nothing


# ── regression #4: retry idempotency ──────────────────────────────────


def test_reappend_same_row_overwrites(cache: SqliteResponseCache) -> None:
    """Regression: a worker that retries a row after a transient
    failure must not create a duplicate cache entry. Resume would
    then emit the row twice and the output dataframe would have
    the wrong length."""
    session = uuid4()
    cache.append(session, 3, _resp("first-try"), _meta(3))
    cache.append(session, 3, _resp("retry-result"), _meta(3))

    rows = list(cache.iter_completed(session))
    assert len(rows) == 1
    assert rows[0][1].text == "retry-result"


# ── regression #5: metadata round-trip ────────────────────────────────


def test_metadata_roundtrips_custom_dict(cache: SqliteResponseCache) -> None:
    """Regression: ``RowMetadata.custom`` carries batch-membership
    flags (``is_batch``, ``batch_size``) that the disaggregator reads
    on resume. Losing them breaks batched-prompt reassembly."""
    session = uuid4()
    meta = CachedRowMetadata(
        row_index=0,
        row_id="external-id",
        custom={"is_batch": True, "batch_size": 8},
    )
    cache.append(session, 0, _resp(), meta)

    ((_, _, got),) = list(cache.iter_completed(session))
    assert got.row_id == "external-id"
    assert got.custom == {"is_batch": True, "batch_size": 8}


# ── regression #6: crash atomicity (the whole motivation) ─────────────


def test_append_survives_process_kill(tmp_path: Path) -> None:
    """Regression: the predecessor — a gzipped-JSON checkpoint
    rewritten every 500 rows — could be truncated mid-write by a
    SIGKILL, losing every response written after the last
    successful rewrite. With row-level atomic appends + WAL, a
    SIGKILL between appends leaves *every completed row* intact.

    We prove this end-to-end with a real subprocess: spawn a worker
    that writes 100 rows and then calls ``os._exit(9)`` (no cleanup,
    no atexit, no flush-on-close). Reopen the cache in the parent
    and verify all 100 rows survived.
    """
    db_path = tmp_path / "crash.db"
    session_id = str(uuid4())

    worker = f"""
import os, sys
from decimal import Decimal
from uuid import UUID
from ondine.adapters.response_cache import (
    SqliteResponseCache, CachedResponse, CachedRowMetadata,
)
cache = SqliteResponseCache({str(db_path)!r})
session = UUID({session_id!r})
for i in range(100):
    cache.append(
        session, i,
        CachedResponse(
            text=f"row-{{i}}",
            tokens_in=1, tokens_out=1,
            model="m",
            cost=Decimal("0.001"),
            latency_ms=1.0,
            metadata={{}},
        ),
        CachedRowMetadata(row_index=i, row_id=None, custom={{}}),
    )
# Hard kill — no close(), no atexit, no Python cleanup path.
os._exit(9)
"""

    proc = subprocess.run(
        [sys.executable, "-c", worker],
        env={**os.environ, "PYTHONPATH": os.getcwd()},
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 9, (
        f"worker did not hard-exit: {proc.returncode}\n{proc.stderr.decode()}"
    )

    # Parent reopens the DB. If append were not crash-atomic, we'd
    # see <100 rows (truncated WAL, half-written transaction, etc).
    cache = SqliteResponseCache(db_path)
    try:
        session = UUID(session_id)
        rows = list(cache.iter_completed(session))
        assert len(rows) == 100
        assert cache.last_processed_row(session) == 99
    finally:
        cache.close()


# ── regression #7: resume after reopen ───────────────────────────────


def test_reopen_sees_previously_appended_rows(tmp_path: Path) -> None:
    """Regression: resume happens in a *new process*. Data must be
    durable across the close/reopen boundary, not just within a
    single process's memory."""
    db_path = tmp_path / "reopen.db"
    session = uuid4()

    a = SqliteResponseCache(db_path)
    a.append(session, 0, _resp("persisted"), _meta(0))
    a.close()

    b = SqliteResponseCache(db_path)
    try:
        rows = list(b.iter_completed(session))
        assert len(rows) == 1
        assert rows[0][1].text == "persisted"
    finally:
        b.close()
