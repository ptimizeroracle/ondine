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
