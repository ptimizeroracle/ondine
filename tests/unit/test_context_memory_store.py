"""Tests for InMemoryContextStore fallback."""

import pytest

from ondine.context.memory_store import InMemoryContextStore
from ondine.context.protocol import EvidenceRecord


@pytest.fixture
def store():
    return InMemoryContextStore()


class TestInMemoryStore:
    def test_store_and_retrieve(self, store):
        record = EvidenceRecord(text="Test claim text", source_ref="test.csv")
        cid = store.store(record)
        retrieved = store.retrieve(cid)
        assert retrieved is not None
        assert retrieved.text == "Test claim text"

    def test_duplicate_increments_support(self, store):
        record = EvidenceRecord(text="Repeated claim", claim_id="fixed-id")
        store.store(record)
        store.store(record)
        results = store.search("Repeated claim")
        assert len(results) >= 1
        assert results[0].support_count == 2

    def test_search_empty_store(self, store):
        results = store.search("anything")
        assert results == []

    def test_grounding_stores_result(self, store):
        results = store.ground(
            "Organic cereals from Italy",
            ["Organic cereals from Italy are popular."],
            threshold=0.3,
        )
        assert len(results) == 1
        assert results[0].grounded is True
        retrieved = store.retrieve(results[0].claim_id)
        assert retrieved is not None

    def test_grounding_below_threshold(self, store):
        results = store.ground(
            "Quantum computing advances",
            ["Weather is nice today."],
            threshold=0.3,
        )
        assert len(results) == 0

    def test_contradictions(self, store):
        id_a = store.store(EvidenceRecord(text="X is A"))
        id_b = store.store(EvidenceRecord(text="X is B"))
        store.add_contradiction(id_a, id_b)
        assert id_b in store.get_contradictions(id_a)
        assert id_a in store.get_contradictions(id_b)

    def test_close_clears_state(self, store):
        store.store(EvidenceRecord(text="Some data"))
        store.close()
        assert store.search("data") == []
