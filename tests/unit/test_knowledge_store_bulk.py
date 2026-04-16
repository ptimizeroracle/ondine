"""Tests for KnowledgeStore bulk ingest path.

Each test targets a specific regression in the bulk-ingest integration:

- ``test_ingest_documents_many_chunks_uses_bulk_path``: if the facade
  forgets to route through ``store_chunks_batch`` for multi-chunk
  ingests, this test catches it via a wrapper-level counter.
- ``test_ingest_documents_result_equivalent_to_single_path``: the
  user-visible outcome (chunks searchable, correct count) must match
  the pre-existing per-chunk behaviour.
- ``test_ingest_documents_empty_list_stores_nothing``: edge case that
  previously would still create no rows; ensures bulk path respects it.
"""

from __future__ import annotations

from ondine.knowledge.loader import Document
from ondine.knowledge.store import KnowledgeStore


def _docs(n: int) -> list[Document]:
    return [
        Document(
            text=f"passage {i} about chunk content " * 5,
            source=f"doc_{i}.md",
            metadata={"idx": i},
        )
        for i in range(n)
    ]


class _CountingDB:
    """Proxy over the real Rust DB that counts which path was used."""

    def __init__(self, inner):
        self._inner = inner
        self.single_calls = 0
        self.batch_calls = 0
        self.batch_total_rows = 0

    def store_chunk(self, *args, **kwargs):
        self.single_calls += 1
        return self._inner.store_chunk(*args, **kwargs)

    def store_chunks_batch(self, chunks):
        self.batch_calls += 1
        self.batch_total_rows += len(chunks)
        return self._inner.store_chunks_batch(chunks)

    def query_chunks(self, q, limit):
        return self._inner.query_chunks(q, limit)

    def chunk_count(self):
        return self._inner.chunk_count()


def test_ingest_documents_many_chunks_uses_bulk_path() -> None:
    store = KnowledgeStore(":memory:", embedder=None, reranker=None)
    # Disable the embedder so the ingest path doesn't try to load
    # sentence-transformers (not guaranteed in CI environments).
    store._embedder = None
    store._db = _CountingDB(store._db)

    store.ingest_documents(_docs(5))

    counter: _CountingDB = store._db  # type: ignore[assignment]
    assert counter.batch_calls >= 1
    assert counter.batch_total_rows >= 5
    # Single-path must not be used for multi-chunk ingests.
    assert counter.single_calls == 0


def test_ingest_documents_result_equivalent_to_single_path() -> None:
    """Behavioural equivalence: bulk and hypothetical single path must
    leave the KB in the same searchable state."""
    store = KnowledgeStore(":memory:", embedder=None, reranker=None)
    store._embedder = None

    count = store.ingest_documents(_docs(8))

    assert count == store.chunk_count
    assert count >= 8  # at least one chunk per doc
    hits = store.search("passage about chunk content", limit=20)
    # Hits should cover multiple sources from different docs.
    sources = {h.source for h in hits}
    assert len(sources) >= 3


def test_ingest_documents_empty_list_stores_nothing() -> None:
    store = KnowledgeStore(":memory:", embedder=None, reranker=None)

    count = store.ingest_documents([])

    assert count == 0
    assert store.chunk_count == 0
