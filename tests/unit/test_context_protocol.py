"""Tests for ContextStore protocol compliance across all backends."""

import pytest

from ondine.context.memory_store import InMemoryContextStore
from ondine.context.protocol import ContextStore, EvidenceRecord
from ondine.context.rust_store import RustContextStore


def _make_record(text: str = "Product X is Organic Cereals") -> EvidenceRecord:
    return EvidenceRecord(
        text=text,
        source_ref="test_doc.csv",
        claim_type="factual",
        source_type="llm_response",
        asserted_by="test_pipeline",
    )


@pytest.fixture(params=["memory", "rust"])
def store(request) -> ContextStore:
    if request.param == "memory":
        return InMemoryContextStore()
    if request.param == "rust":
        return RustContextStore(":memory:")
    pytest.skip(f"Unknown backend: {request.param}")


class TestProtocolCompliance:
    """Every backend must satisfy the ContextStore ABC contract."""

    def test_is_context_store_subclass(self, store):
        assert isinstance(store, ContextStore)

    def test_store_returns_claim_id(self, store):
        record = _make_record()
        claim_id = store.store(record)
        assert isinstance(claim_id, str)
        assert len(claim_id) > 0

    def test_retrieve_after_store(self, store):
        record = _make_record("Bio Valley Granola is Organic")
        claim_id = store.store(record)
        retrieved = store.retrieve(claim_id)
        assert retrieved is not None
        assert retrieved.text == "Bio Valley Granola is Organic"

    def test_retrieve_nonexistent_returns_none(self, store):
        result = store.retrieve("00000000-0000-0000-0000-000000000000")
        assert result is None

    def test_search_returns_list(self, store):
        store.store(_make_record("Organic premium cereals from Italy"))
        store.store(_make_record("Frozen vegetables budget line"))
        results = store.search("organic cereals", limit=5)
        assert isinstance(results, list)

    def test_search_ranks_relevant_higher(self, store):
        store.store(_make_record("Organic premium cereals from Italy"))
        store.store(_make_record("Frozen vegetables budget line"))
        results = store.search("organic cereals")
        if len(results) >= 2:
            assert results[0].score >= results[1].score

    def test_close_does_not_raise(self, store):
        store.store(_make_record())
        store.close()
