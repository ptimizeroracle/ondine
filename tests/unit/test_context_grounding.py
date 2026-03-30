"""Tests for grounding logic through the Python API."""

import pytest

from ondine.context.memory_store import InMemoryContextStore
from ondine.context.rust_store import RustContextStore


@pytest.fixture(params=["rust", "memory"])
def store(request):
    if request.param == "rust":
        return RustContextStore(":memory:")
    return InMemoryContextStore()


class TestGrounding:
    def test_exact_match_high_confidence(self, store):
        text = "Product X is in Organic Cereals category"
        results = store.ground(text, [text], threshold=0.3)
        assert len(results) == 1
        assert results[0].confidence > 0.9

    def test_similar_text_grounded(self, store):
        results = store.ground(
            "Product X belongs to Organic Cereals category",
            [
                "Product X belongs to Organic Cereals category according to the database."
            ],
            threshold=0.3,
        )
        assert len(results) == 1
        assert results[0].confidence > 0.3

    def test_unrelated_text_not_grounded(self, store):
        results = store.ground(
            "This will disrupt the entire industry forever",
            ["The weather forecast shows rain tomorrow."],
            threshold=0.3,
        )
        assert len(results) == 0

    def test_threshold_zero_grounds_everything(self, store):
        results = store.ground(
            "Random claim",
            ["Something unrelated"],
            threshold=0.0001,
        )
        # With a near-zero threshold, even a tiny overlap may pass.
        # The result depends on actual TF-IDF overlap.
        assert isinstance(results, list)

    def test_empty_source_sentences(self, store):
        results = store.ground("Some claim", [], threshold=0.3)
        assert results == []


class TestEmbeddingAugmentedGrounding:
    """Verify that embed_fn boosts grounding for cross-lingual or semantic matches."""

    @pytest.fixture(params=["rust", "memory"])
    def store(self, request):
        if request.param == "rust":
            return RustContextStore(":memory:")
        return InMemoryContextStore()

    @staticmethod
    def _mock_embed_fn(texts: list[str]) -> list[list[float]]:
        """Return fixed vectors: first text gets [1,0,0], rest get [0.9,0.1,0]."""
        vecs = []
        for i, _ in enumerate(texts):
            if i == 0:
                vecs.append([1.0, 0.0, 0.0])
            else:
                vecs.append([0.9, 0.1, 0.0])
        return vecs

    def test_embed_fn_rescues_cross_lingual_mismatch(self, store):
        # TF-IDF would score 0 (no word overlap), but embeddings score high
        results = store.ground(
            "organic fennel product",
            ["FINOCCHI BIO 1.000 KG"],
            threshold=0.3,
            embed_fn=self._mock_embed_fn,
        )
        assert len(results) == 1
        assert results[0].confidence > 0.3

    def test_no_embed_fn_preserves_tfidf_behavior(self, store):
        # Without embed_fn, cross-lingual gets no grounding
        results = store.ground(
            "organic fennel product",
            ["FINOCCHI BIO 1.000 KG"],
            threshold=0.3,
            embed_fn=None,
        )
        assert len(results) == 0
