"""Append-only cache of completed LLM responses, scoped by session.

Purpose
-------
Pipelines resume after crash without re-invoking the LLM for rows
already processed. Every completed response is persisted row-by-row
via an atomic write — so even ``kill -9`` mid-run leaves the cache
in a consistent state.

This is the storage the checkpoint system *used to* put inline in
``ExecutionContext.intermediate_data["completed_responses"]`` as a
growing list inside a gzipped JSON blob. That approach rewrote the
entire blob every checkpoint window (O(N²) IO) and was not
crash-atomic. This module replaces that.

Public contract
---------------
Callers see list-like semantics, scoped by ``session_id``:

* ``append(session_id, row_index, response, metadata)`` —
  persist one completed response. Crash-safe on return: a
  subsequent process opening the same cache sees the row.
* ``iter_completed(session_id)`` — yield ``(row_index, response,
  metadata)`` in ascending ``row_index`` order. Used by resume to
  rebuild ``ResponseBatch`` objects without re-calling the LLM.
* ``last_processed_row(session_id)`` — highest ``row_index`` ever
  appended for this session, or -1 if none. Drives the resume
  filter in the pipeline.
* ``clear(session_id)`` — remove everything for a session. Called
  by the executor on successful completion.

The backend (SQLite, file, memory) is irrelevant to callers. That
is the whole point: storage choice is not part of the caller's
cognitive load.

Invariants
----------
1. ``append`` is atomic with respect to process death. After
   control returns, the row is durable on disk (``fsync`` equivalent
   handled by backend). No partial rows.
2. Rows are uniquely keyed by ``(session_id, row_index)``.
   Re-appending the same key overwrites — this handles the
   idempotency case where a worker retries a row after a transient
   failure.
3. ``iter_completed`` yields rows strictly ordered by
   ``row_index`` ascending, regardless of insertion order.
4. Sessions are isolated: operations on session A never observe or
   mutate rows of session B.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    from uuid import UUID


@dataclass(frozen=True)
class CachedRowMetadata:
    """Minimal, serializable snapshot of a row's identity.

    Mirrors ``ondine.core.models.RowMetadata`` but without importing
    the rest of the model graph. ``custom`` stays a plain dict; the
    backend round-trips it through JSON.
    """

    row_index: int
    row_id: str | None
    custom: dict[str, object]


@dataclass(frozen=True)
class CachedResponse:
    """Minimal, serializable snapshot of an LLM response.

    Everything needed to rebuild a ``ResponseBatch`` on resume.
    ``structured_result`` is *not* cached: it is a transient
    parsed-object reference that the disaggregator rebuilds from
    ``text`` if needed. Persisting it would require pickling
    arbitrary Pydantic models — too much surface area for the
    resume-correctness guarantee we need.
    """

    text: str
    tokens_in: int
    tokens_out: int
    model: str
    cost: Decimal
    latency_ms: float
    metadata: dict[str, object]


class ResponseCache(ABC):
    """Abstract backend. Implementations: ``SqliteResponseCache``,
    ``InMemoryResponseCache`` (for tests)."""

    @abstractmethod
    def append(
        self,
        session_id: UUID,
        row_index: int,
        response: CachedResponse,
        metadata: CachedRowMetadata,
    ) -> None:
        """Persist one row. Durable on return."""

    @abstractmethod
    def iter_completed(
        self, session_id: UUID
    ) -> Iterable[tuple[int, CachedResponse, CachedRowMetadata]]:
        """Yield ``(row_index, response, metadata)`` ascending by
        ``row_index``. Empty iterator if no rows."""

    @abstractmethod
    def last_processed_row(self, session_id: UUID) -> int:
        """Highest ``row_index`` for session, or -1 if empty."""

    @abstractmethod
    def clear(self, session_id: UUID) -> None:
        """Remove all rows for session. Idempotent."""

    @abstractmethod
    def close(self) -> None:
        """Release resources. Idempotent."""


# ──────────────────────────────────────────────────────────────────────
# SQLite-backed implementation
# ──────────────────────────────────────────────────────────────────────


_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    session_id  TEXT    NOT NULL,
    row_index   INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    tokens_in   INTEGER NOT NULL,
    tokens_out  INTEGER NOT NULL,
    model       TEXT    NOT NULL,
    cost        TEXT    NOT NULL,   -- Decimal stored as string for exact round-trip
    latency_ms  REAL    NOT NULL,
    metadata    TEXT    NOT NULL,   -- json
    row_id      TEXT,
    custom      TEXT    NOT NULL,   -- json
    PRIMARY KEY (session_id, row_index)
);
"""


class SqliteResponseCache(ResponseCache):
    """SQLite implementation using WAL for crash atomicity.

    Why SQLite:
      * Single-file durability; survives scp/zip/restart.
      * WAL + synchronous=NORMAL makes each INSERT crash-safe
        without paying a full fsync per row.
      * PRIMARY KEY (session_id, row_index) + INSERT OR REPLACE
        gives us free idempotency for retried rows.

    Why a lock:
      * SQLite connections are not thread-safe by default. A pipeline
        has one cache instance across many worker tasks. A coarse
        ``threading.Lock`` around writes is cheaper than
        ``check_same_thread=False`` plus external locking, and
        correct under the expected workload (append-heavy, one
        writer at a time in practice).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self._path,
            isolation_level=None,  # autocommit — each INSERT is its own txn
            check_same_thread=False,
        )
        # WAL survives a hard kill with every committed txn intact.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # Multiple processes/connections may share the same responses.db
        # (e.g. pipelined streaming spins one sub-pipeline per chunk, each
        # opening its own connection). Wait up to 5s for the writer lock
        # instead of erroring with "database is locked".
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(_SCHEMA)

    def append(
        self,
        session_id: UUID,
        row_index: int,
        response: CachedResponse,
        metadata: CachedRowMetadata,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO responses
                (session_id, row_index, text, tokens_in, tokens_out, model,
                 cost, latency_ms, metadata, row_id, custom)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(session_id),
                    row_index,
                    response.text,
                    response.tokens_in,
                    response.tokens_out,
                    response.model,
                    str(response.cost),
                    response.latency_ms,
                    json.dumps(response.metadata),
                    metadata.row_id,
                    json.dumps(metadata.custom),
                ),
            )

    def iter_completed(
        self, session_id: UUID
    ) -> Iterable[tuple[int, CachedResponse, CachedRowMetadata]]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT row_index, text, tokens_in, tokens_out, model,
                       cost, latency_ms, metadata, row_id, custom
                FROM responses
                WHERE session_id = ?
                ORDER BY row_index ASC
                """,
                (str(session_id),),
            )
            rows = cur.fetchall()
        for (
            row_index,
            text,
            tokens_in,
            tokens_out,
            model,
            cost,
            latency_ms,
            metadata,
            row_id,
            custom,
        ) in rows:
            yield (
                row_index,
                CachedResponse(
                    text=text,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    model=model,
                    cost=Decimal(cost),
                    latency_ms=latency_ms,
                    metadata=json.loads(metadata),
                ),
                CachedRowMetadata(
                    row_index=row_index,
                    row_id=row_id,
                    custom=json.loads(custom),
                ),
            )

    def last_processed_row(self, session_id: UUID) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT MAX(row_index) FROM responses WHERE session_id = ?",
                (str(session_id),),
            )
            (result,) = cur.fetchone()
        return -1 if result is None else int(result)

    def clear(self, session_id: UUID) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM responses WHERE session_id = ?",
                (str(session_id),),
            )

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None  # type: ignore[assignment]
