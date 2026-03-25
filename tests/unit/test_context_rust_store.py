"""Tests for the Rust-backed context store (RustContextStore)."""

import pytest

from ondine.context.protocol import EvidenceRecord
from ondine.context.rust_store import RustContextStore


@pytest.fixture
def store():
    return RustContextStore(":memory:")


class TestRustStoreRoundTrip:
    """Store → retrieve → search round-trip via Rust engine."""

    def test_store_and_retrieve(self, store):
        record = EvidenceRecord(
            text="Product X belongs to Organic Cereals category",
            source_ref="catalogue.csv:row_42",
            claim_type="factual",
            source_type="llm_response",
            asserted_by="pipeline_run_1",
        )
        claim_id = store.store(record)
        retrieved = store.retrieve(claim_id)
        assert retrieved is not None
        assert "Organic Cereals" in retrieved.text

    def test_search_returns_results(self, store):
        store.store(EvidenceRecord(text="Organic premium granola", source_ref="a"))
        store.store(EvidenceRecord(text="Frozen fish sticks", source_ref="b"))
        store.store(EvidenceRecord(text="Organic bio cereal bars", source_ref="c"))

        results = store.search("organic cereals", limit=5)
        assert len(results) >= 1
        assert any(
            "organic" in r.text.lower() or "cereal" in r.text.lower() for r in results
        )

    def test_tfidf_similarity_from_engine(self):
        from ondine._engine import tfidf_similarity

        sim = tfidf_similarity("organic cereals", "organic cereals premium")
        assert 0.0 < sim <= 1.0

    def test_tfidf_identical_is_one(self):
        from ondine._engine import tfidf_similarity

        sim = tfidf_similarity("hello world", "hello world")
        assert abs(sim - 1.0) < 1e-10


class TestRustStoreGrounding:
    """TF-IDF grounding via Rust engine."""

    def test_grounding_above_threshold(self, store):
        results = store.ground(
            response_text="Product X is categorized as Organic Cereals",
            source_sentences=[
                "Product X is categorized as Organic Cereals in the database.",
            ],
            threshold=0.3,
        )
        assert len(results) == 1
        assert results[0].confidence > 0.3
        assert results[0].grounded is True

    def test_grounding_below_threshold_discards(self, store):
        results = store.ground(
            response_text="This product will dominate global markets",
            source_sentences=[
                "Weather patterns in arctic regions have been changing.",
            ],
            threshold=0.3,
        )
        assert len(results) == 0


class TestRustStoreContradiction:
    """Contradiction detection via Rust engine."""

    def test_add_and_get_contradictions(self, store):
        id_a = store.store(EvidenceRecord(text="Product X is Cereals"))
        id_b = store.store(EvidenceRecord(text="Product X is Snacks"))

        store.add_contradiction(id_a, id_b)

        contras_a = store.get_contradictions(id_a)
        assert id_b in contras_a

        contras_b = store.get_contradictions(id_b)
        assert id_a in contras_b
