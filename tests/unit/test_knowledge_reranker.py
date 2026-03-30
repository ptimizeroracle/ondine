"""Tests for CrossEncoderReranker — verifies reranking behavior.

Regressions caught:
- Empty results pass through without error (boundary)
- Results are truncated to top_k (limit enforcement)
- Graceful fallback when sentence-transformers unavailable (no crash)
"""

from ondine.knowledge.reranker import CrossEncoderReranker
from ondine.knowledge.store import SearchResult


def _make_result(text: str, score: float = 0.5) -> SearchResult:
    return SearchResult(chunk_id="id", text=text, source="s", score=score)


class TestCrossEncoderReranker:
    def test_empty_results_returns_empty(self):
        reranker = CrossEncoderReranker(top_k=3)
        assert reranker.rerank("query", []) == []

    def test_truncates_to_top_k_when_model_unavailable(self):
        reranker = CrossEncoderReranker(top_k=2)
        reranker._get_model = lambda: None  # simulate unavailable model

        results = [_make_result(f"text {i}") for i in range(5)]
        reranked = reranker.rerank("query", results)
        assert len(reranked) == 2

    def test_results_preserved_when_model_unavailable(self):
        reranker = CrossEncoderReranker(top_k=10)
        reranker._get_model = lambda: None

        results = [_make_result("hello"), _make_result("world")]
        reranked = reranker.rerank("query", results)
        assert len(reranked) == 2
        assert reranked[0].text == "hello"
