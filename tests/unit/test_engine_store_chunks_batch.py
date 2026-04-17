"""Behavioural tests for the bulk chunk ingest FFI path.

Each test targets a specific regression:

- ``test_store_chunks_batch_inserts_all_rows``: regressions in loop/txn
  commit would drop rows silently; count must equal input.
- ``test_store_chunks_batch_preserves_text_and_source``: catches column
  mis-ordering in the prepared statement.
- ``test_store_chunks_batch_roundtrip_via_query_chunks``: catches any
  divergence between the new bulk path and the single-row path
  (they must populate the same tables, including the FTS index).
- ``test_store_chunks_batch_empty_list_is_noop``: empty input must not
  crash and must not leave partial state.
- ``test_store_chunks_batch_duplicate_id_replaces``: matches the
  INSERT OR REPLACE semantics of ``store_chunk`` — regression would be
  silent UNIQUE-constraint failure.
"""

from __future__ import annotations

import json
import uuid

import pytest

from ondine import _engine


@pytest.fixture
def db() -> _engine.EvidenceDB:
    return _engine.EvidenceDB(":memory:")


def _chunk(i: int) -> tuple[str, str, str, str]:
    return (
        str(uuid.uuid4()),
        f"chunk text {i}",
        f"doc_{i}.md",
        json.dumps({"idx": i}),
    )


def test_store_chunks_batch_inserts_all_rows(db: _engine.EvidenceDB) -> None:
    chunks = [_chunk(i) for i in range(250)]

    db.store_chunks_batch(chunks)

    assert db.chunk_count() == 250


def test_store_chunks_batch_preserves_text_and_source(
    db: _engine.EvidenceDB,
) -> None:
    cid = str(uuid.uuid4())
    db.store_chunks_batch(
        [(cid, "organic cereals contain whole grains", "pantry.pdf", "{}")]
    )

    raw = db.query_chunks("organic cereals whole grains", 1)
    results = json.loads(raw)

    assert len(results) == 1
    assert results[0][0] == cid  # chunk_id
    assert results[0][1] == "organic cereals contain whole grains"
    assert results[0][2] == "pantry.pdf"


def test_store_chunks_batch_roundtrip_matches_single_path(
    db: _engine.EvidenceDB,
) -> None:
    """Bulk and single paths must be indistinguishable at query time.

    If the bulk path forgets the FTS index update, this test fails.
    """
    db_single = _engine.EvidenceDB(":memory:")
    db_bulk = _engine.EvidenceDB(":memory:")
    chunks = [_chunk(i) for i in range(50)]

    for cid, text, source, meta in chunks:
        db_single.store_chunk(cid, text, source, meta)
    db_bulk.store_chunks_batch(chunks)

    # Same query returns same hit count and same chunk_ids.
    single_ids = {
        row[0] for row in json.loads(db_single.query_chunks("chunk text", 50))
    }
    bulk_ids = {row[0] for row in json.loads(db_bulk.query_chunks("chunk text", 50))}

    assert bulk_ids == single_ids
    assert len(bulk_ids) > 0  # sanity — search actually returns hits


def test_store_chunks_batch_empty_list_is_noop(db: _engine.EvidenceDB) -> None:
    db.store_chunks_batch([])

    assert db.chunk_count() == 0


def test_store_chunks_batch_duplicate_id_replaces(
    db: _engine.EvidenceDB,
) -> None:
    """INSERT OR REPLACE semantics — second insert wins, count stays 1.

    Also guards against orphaned FTS5 rows: when the primary-key upsert
    changes the rowid (as with INSERT OR REPLACE), the old FTS entry at
    the previous rowid must be removed, otherwise the old text stays
    searchable.
    """
    cid = str(uuid.uuid4())
    db.store_chunks_batch([(cid, "alpha quokka", "a.pdf", "{}")])
    db.store_chunks_batch([(cid, "bravo narwhal", "a.pdf", "{}")])

    assert db.chunk_count() == 1
    new_hits = json.loads(db.query_chunks("narwhal", 5))
    assert len(new_hits) == 1
    assert new_hits[0][1] == "bravo narwhal"

    # Old text must not remain searchable via the FTS index.
    stale_hits = json.loads(db.query_chunks("quokka", 5))
    assert stale_hits == []


def test_store_chunk_single_path_also_clears_stale_fts(
    db: _engine.EvidenceDB,
) -> None:
    """The per-row store_chunk had the same FTS-orphan bug. Guard
    against regression here by re-using the single-row API."""
    cid = str(uuid.uuid4())
    db.store_chunk(cid, "alpha quokka", "a.pdf", "{}")
    db.store_chunk(cid, "bravo narwhal", "a.pdf", "{}")

    assert db.chunk_count() == 1
    assert json.loads(db.query_chunks("quokka", 5)) == []
    hits = json.loads(db.query_chunks("narwhal", 5))
    assert len(hits) == 1
    assert hits[0][1] == "bravo narwhal"
