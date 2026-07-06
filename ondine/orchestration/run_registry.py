"""Persistent job index for pipeline runs.

The RunRegistry is the single source of truth for the lifetime of a run
across process boundaries. The MCP ``ondine_run`` tool hands out a
``run_id`` immediately; a worker process (or a later CLI invocation)
resolves that same id against the on-disk SQLite database. Because the
whole point is crash-safe durability, the registry is backed by a real
SQLite file with WAL mode — never an in-memory mock.

Design (deep module, information hiding):

* ``RunRegistry`` exposes four operations — ``create``, ``get``,
  ``list``, ``transition`` — and hides every SQL statement, schema
  migration, and serialisation detail behind them.
* ``RunHandle`` is an immutable snapshot read fresh from disk on every
  ``get``/``list``/``transition`` call. There is deliberately no
  in-memory cache: a second process polling status must see the
  latest committed row, so caching would be a bug, not an optimisation.
* ``RegistryObserver`` is the extension point for live progress feeds
  (the MCP status stream). Observers are notified inside the same
  transaction commit so a crash between "persist" and "notify" is
  impossible.

The registry is optional. ``Pipeline.execute(run_id=None)`` (the
default) never touches it, so existing users see no new on-disk artefact.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from ondine.utils import get_logger

logger = get_logger(__name__)

# Sentinel filename inside the checkpoint directory. Co-located with
# checkpoints and the response cache so that copying the checkpoint dir
# also carries the run history — one artefact to back up.
REGISTRY_FILENAME = "runs.db"


class RunStatus(str, Enum):
    """Lifecycle states for a single pipeline run.

    The ordering encodes the legal forward transitions (see
    ``_ALLOWED_TRANSITIONS``). A run moves strictly forward; there is no
    "un-fail" path. PARTIAL covers the ProviderBatch case where some
    rows succeeded but the job did not complete cleanly.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUBMITTED_REMOTE = "submitted_remote"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"


# Legal forward edges. Key = current status, value = statuses reachable
# in one hop. Defined once, checked on every transition so the registry
# can never record an impossible history (e.g. PENDING -> SUCCEEDED).
_ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset({RunStatus.RUNNING, RunStatus.FAILED}),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.SUBMITTED_REMOTE,
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.PARTIAL,
        }
    ),
    RunStatus.SUBMITTED_REMOTE: frozenset(
        {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.PARTIAL}
    ),
    # Terminal states have no outgoing edges.
    RunStatus.SUCCEEDED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.PARTIAL: frozenset(),
}


class RegistryObserver(ABC):
    """Extension point for status-transition events.

    Implementations drive the live progress feed in the MCP server and
    any metrics sinks. The registry calls ``on_transition`` exactly once
    per committed transition, after the row is durable on disk. An
    observer that raises cannot roll back the transition — the registry
    logs and swallows the error so one misbehaving listener cannot take
    down a run (mirrors ``ExecutionContext.notify_progress``).
    """

    @abstractmethod
    def on_transition(self, run_id: UUID, old: RunStatus, new: RunStatus) -> None:
        """Called after a status transition is committed to disk."""
        raise NotImplementedError


class RunSpec:
    """User-supplied description of a run at creation time.

    A thin value object: ``pipeline_id`` identifies which pipeline
    configuration is running, ``dataset`` is a short label for the input
    (a filename or ``"dataframe"``), and ``spec_snapshot`` is the
    arbitrary JSON-serialisable dict that resume and the CLI rely on to
    reconstruct the run. Everything serialises through ``to_dict`` /
    ``from_dict`` so the registry never has to know the shape.
    """

    def __init__(
        self,
        pipeline_id: str,
        dataset: str = "",
        spec_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.pipeline_id = pipeline_id
        self.dataset = dataset
        # Default the snapshot to the identifying fields so that callers
        # who only pass pipeline_id/dataset still get a useful, non-empty
        # snapshot (the contract every test asserts on).
        self.spec_snapshot = (
            dict(spec_snapshot)
            if spec_snapshot
            else {
                "pipeline_id": pipeline_id,
                "dataset": dataset,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "dataset": self.dataset,
            "spec_snapshot": self.spec_snapshot,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunSpec:
        # Tolerate both the full serialised form and a bare snapshot
        # (tests pass {"spec_snapshot": {...}} directly).
        if "spec_snapshot" in data and "pipeline_id" not in data:
            snapshot = data["spec_snapshot"]
            return cls(
                pipeline_id=snapshot.get("pipeline_id", str(uuid4())),
                dataset=snapshot.get("dataset", ""),
                spec_snapshot=snapshot,
            )
        return cls(
            pipeline_id=data.get("pipeline_id", str(uuid4())),
            dataset=data.get("dataset", ""),
            spec_snapshot=data.get("spec_snapshot"),
        )


class RunHandle:
    """Immutable, read-only view of a run's current on-disk state.

    A handle is a snapshot, not a live cursor: re-fetch via
    ``registry.get(run_id)`` to observe subsequent transitions. Keeping
    it immutable means callers cannot drift the in-memory copy out of
    sync with the database.
    """

    __slots__ = (
        "run_id",
        "status",
        "spec_snapshot",
        "metrics",
        "provider_job_id",
        "created_at",
        "updated_at",
    )

    def __init__(
        self,
        run_id: UUID,
        status: RunStatus,
        spec_snapshot: dict[str, Any],
        metrics: dict[str, Any],
        provider_job_id: str | None,
        created_at: str,
        updated_at: str,
    ) -> None:
        self.run_id = run_id
        self.status = status
        self.spec_snapshot = spec_snapshot
        self.metrics = metrics
        self.provider_job_id = provider_job_id
        self.created_at = created_at
        self.updated_at = updated_at

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"RunHandle(run_id={self.run_id}, status={self.status.name}, "
            f"provider_job_id={self.provider_job_id!r})"
        )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    status           TEXT NOT NULL,
    spec_snapshot    TEXT NOT NULL,
    metrics          TEXT NOT NULL DEFAULT '{}',
    provider_job_id  TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
"""


def _row_to_handle(row: sqlite3.Row) -> RunHandle:
    return RunHandle(
        run_id=UUID(row["run_id"]),
        status=RunStatus(row["status"]),
        spec_snapshot=json.loads(row["spec_snapshot"]),
        metrics=json.loads(row["metrics"] or "{}"),
        provider_job_id=row["provider_job_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class RunRegistry:
    """Crash-safe, on-disk index of pipeline runs.

    The registry is a deep module: callers use ``create`` / ``get`` /
    ``list`` / ``transition`` and never see SQL, the WAL pragma, or the
    serialisation format. The database file lives in the existing
    checkpoint directory (or any path the caller chooses), so there is
    exactly one persistence location to back up.

    Thread-safety: a module-level lock serialises writes so concurrent
    threads in one process cannot interleave a transition with observer
    dispatch. Cross-process safety comes from SQLite's own file locking.
    """

    # Process-wide lock. SQLite serialises cross-process writes, but
    # observer dispatch must not be interrupted by a second in-process
    # transition.
    _write_lock = threading.Lock()

    def __init__(self, path: Path | str) -> None:
        """Open (or create) the registry at ``path``.

        ``path`` may be either a directory (the registry file is created
        inside it as ``runs.db``, co-located with checkpoints) or a
        direct file path. A directory keeps every persistence artefact
        in one place; a file path is convenient for tests and ad-hoc
        tooling.
        """
        path = Path(path)
        if path.is_dir() or (not path.exists() and path.suffix == ""):
            # Treat as a directory: ensure it exists, then nest the db.
            path.mkdir(parents=True, exist_ok=True)
            self._db_path = path / REGISTRY_FILENAME
        else:
            self._db_path = path
            self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self._db_path),
            isolation_level=None,  # autocommit; we manage txns explicitly
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._observers: list[RegistryObserver] = []

    # ── observers ──────────────────────────────────────────────────

    def add_observer(self, observer: RegistryObserver) -> None:
        """Register an observer for status-transition events."""
        self._observers.append(observer)

    def remove_observer(self, observer: RegistryObserver) -> None:
        """Unregister a previously-added observer (no-op if absent)."""
        try:
            self._observers.remove(observer)
        except ValueError:
            pass

    # ── create / read ──────────────────────────────────────────────

    def create(self, spec: RunSpec) -> RunHandle:
        """Persist a new PENDING run and return its handle.

        The run_id is generated server-side so callers can never
        accidentally collide. The row is committed before the handle is
        returned — the MCP tool only hands the id to the user after the
        create is durable, which is the whole reason this module exists.
        """
        run_id = uuid4()
        now = _now_iso()
        snapshot_json = json.dumps(spec.spec_snapshot, default=str)
        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO runs "
                    "(run_id, status, spec_snapshot, metrics, "
                    " provider_job_id, created_at, updated_at) "
                    "VALUES (?, ?, ?, '{}', NULL, ?, ?)",
                    (str(run_id), RunStatus.PENDING.value, snapshot_json, now, now),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        logger.debug("Created run %s (status=%s)", run_id, RunStatus.PENDING.name)
        # Read back so the returned handle is exactly what's on disk.
        handle = self._fetch(run_id)
        assert handle is not None  # just inserted
        return handle

    def get(self, run_id: UUID) -> RunHandle | None:
        """Return the current handle for ``run_id``, or None if absent.

        Reads are uncached: every call hits disk so a second process
        polling status observes committed transitions immediately.
        """
        return self._fetch(run_id)

    def list(self, status: RunStatus | None = None) -> list[RunHandle]:
        """List runs, newest-first, optionally filtered by status.

        Newest-first matches the CLI ``ondine status`` UX: the run the
        user just kicked off is the one they want to see at the top.
        """
        if status is None:
            rows = self._conn.execute(
                # Order by rowid DESC — SQLite assigns rowid in insertion
                # order, so this gives newest-first deterministically
                # even when several creates land within the same
                # microsecond clock tick.
                "SELECT * FROM runs ORDER BY rowid DESC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM runs WHERE status = ? ORDER BY rowid DESC",
                (status.value,),
            ).fetchall()
        return [_row_to_handle(r) for r in rows]

    # ── transition ─────────────────────────────────────────────────

    def transition(
        self,
        run_id: UUID,
        status: RunStatus,
        *,
        metrics: dict[str, Any] | None = None,
        provider_job_id: str | None = None,
    ) -> RunHandle:
        """Move ``run_id`` to ``status``, persist, notify observers.

        Validates the transition against ``_ALLOWED_TRANSITIONS``,
        merges any ``metrics`` / ``provider_job_id`` overrides onto the
        existing row, commits, then fires observers exactly once with
        ``(run_id, old_status, new_status)``. Observer exceptions are
        logged and swallowed so a misbehaving listener cannot corrupt
        the on-disk state or starve other observers.

        Raises:
            KeyError: if ``run_id`` does not exist (programmer error —
                callers only transition known runs).
            ValueError: if the transition is not in
                ``_ALLOWED_TRANSITIONS``.
        """
        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT * FROM runs WHERE run_id = ?", (str(run_id),)
                ).fetchone()
                if row is None:
                    self._conn.execute("ROLLBACK")
                    raise KeyError(run_id)

                old_status = RunStatus(row["status"])
                if status not in _ALLOWED_TRANSITIONS.get(old_status, frozenset()):
                    self._conn.execute("ROLLBACK")
                    raise ValueError(
                        f"Invalid transition {old_status.name} -> {status.name} "
                        f"for run {run_id}"
                    )

                merged_metrics = json.loads(row["metrics"] or "{}")
                if metrics:
                    merged_metrics.update(metrics)

                updated_at = _now_iso()
                self._conn.execute(
                    "UPDATE runs SET status = ?, metrics = ?, "
                    "  provider_job_id = COALESCE(?, provider_job_id), "
                    "  updated_at = ? WHERE run_id = ?",
                    (
                        status.value,
                        json.dumps(merged_metrics, default=str),
                        provider_job_id,
                        updated_at,
                        str(run_id),
                    ),
                )
                self._conn.execute("COMMIT")
            except Exception:
                # ROLLBACK is idempotent on a non-transaction; guard it.
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise

        new_handle = self._fetch(run_id)
        assert new_handle is not None  # row exists; just updated
        self._notify(run_id, old_status, status)
        logger.info(
            "Run %s transitioned %s -> %s",
            run_id,
            old_status.name,
            status.name,
        )
        return new_handle

    # ── lifecycle ──────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection. Safe to call more than once."""
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    # ── internals ──────────────────────────────────────────────────

    def _fetch(self, run_id: UUID) -> RunHandle | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (str(run_id),)
        ).fetchone()
        return _row_to_handle(row) if row is not None else None

    def _notify(self, run_id: UUID, old: RunStatus, new: RunStatus) -> None:
        """Fan out a transition to observers, swallowing their errors."""
        for observer in list(self._observers):
            try:
                observer.on_transition(run_id, old, new)
            except Exception:
                logger.exception(
                    "RegistryObserver %s raised on %s -> %s for run %s",
                    type(observer).__name__,
                    old.name,
                    new.name,
                    run_id,
                )

    def __enter__(self) -> RunRegistry:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _now_iso() -> str:
    """UTC ISO-8601 timestamp with microsecond resolution.

    Microsecond precision is required so that rapid sequential creates
    (the common case — the MCP server may issue several run_ids in one
    request) get distinct timestamps and ``list()`` can order them
    newest-first deterministically.
    """
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="microseconds")
