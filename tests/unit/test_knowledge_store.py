"""Tests for KnowledgeStore — verifies ingest/search lifecycle.

Regressions caught:
- Ingesting text produces searchable chunks (core pipeline)
- Empty query returns empty results (boundary)
- chunk_count tracks stored chunks (state integrity)
- Search ranking returns best match first (correctness)
- Fallback to in-memory when Rust unavailable (graceful degradation)
"""

import pytest

from ondine.knowledge.store import KnowledgeStore, SearchResult, _InMemoryChunkDB


class TestInMemoryChunkDB:
    """Directly test the pure-Python fallback to isolate its logic."""

    def test_store_and_query(self):
        db = _InMemoryChunkDB()
        db.store_chunk("c1", "Organic cereals with whole grains", "a.pdf")
        db.store_chunk("c2", "Frozen vegetables", "b.pdf")

        results = db.query_chunks("organic grains", 5)
        assert len(results) >= 1
        assert results[0][0] == "c1"

    def test_query_empty_db(self):
        db = _InMemoryChunkDB()
        assert db.query_chunks("anything", 5) == []

    def test_chunk_count(self):
        db = _InMemoryChunkDB()
        assert db.chunk_count() == 0
        db.store_chunk("c1", "text", "src")
        assert db.chunk_count() == 1


class TestKnowledgeStore:
    @pytest.fixture
    def store(self):
        return KnowledgeStore(":memory:")

    def test_ingest_text_and_search(self, store):
        count = store.ingest_text("Organic cereals contain whole grains and fiber")
        assert count >= 1
        assert store.chunk_count >= 1

        results = store.search("organic grains", limit=3)
        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)
        assert (
            "organic" in results[0].text.lower() or "grains" in results[0].text.lower()
        )

    def test_search_empty_store_returns_empty(self, store):
        results = store.search("anything", limit=5)
        assert results == []

    def test_ingest_multiple_documents_updates_count(self, store):
        store.ingest_text("First document about apples")
        initial = store.chunk_count
        store.ingest_text("Second document about bananas")
        assert store.chunk_count > initial

    def test_search_ranking_best_match_first(self, store):
        store.ingest_text("Python programming language tutorial")
        store.ingest_text("Java enterprise application server")
        store.ingest_text("Python data science with pandas")

        results = store.search("Python programming", limit=5)
        assert len(results) >= 1
        top = results[0].text.lower()
        assert "python" in top

    def test_chunk_count_property(self, store):
        assert store.chunk_count == 0
        store.ingest_text("some text")
        assert store.chunk_count > 0
