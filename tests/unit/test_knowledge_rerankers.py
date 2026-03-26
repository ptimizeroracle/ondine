"""Tests for Reranker implementations and factory.

Uses mocking to avoid loading real models or calling real APIs.
"""

from unittest.mock import MagicMock, patch

import pytest

from ondine.knowledge.protocols import Reranker
from ondine.knowledge.reranker import (
    CrossEncoderReranker,
    JinaReranker,
    resolve_reranker,
)
from ondine.knowledge.store import SearchResult


def _make_result(text: str, score: float = 0.5) -> SearchResult:
    return SearchResult(chunk_id="id", text=text, source="s", score=score)


class TestCrossEncoderRerankerProtocol:
    def test_satisfies_protocol(self):
        assert isinstance(CrossEncoderReranker(), Reranker)

    def test_default_model_upgraded(self):
        r = CrossEncoderReranker()
        assert "L-12" in r._model_name

    def test_empty_results(self):
        r = CrossEncoderReranker()
        assert r.rerank("q", []) == []

    def test_truncates_when_model_unavailable(self):
        r = CrossEncoderReranker()
        r._get_model = lambda: None
        results = [_make_result(f"t{i}") for i in range(10)]
        out = r.rerank("q", results, top_k=3)
        assert len(out) == 3

    def test_repr(self):
        assert "ms-marco" in repr(CrossEncoderReranker())


class TestJinaReranker:
    def test_satisfies_protocol(self):
        assert isinstance(JinaReranker(), Reranker)

    def test_empty_results(self):
        r = JinaReranker()
        assert r.rerank("q", []) == []

    @patch("litellm.rerank")
    def test_rerank_calls_litellm(self, mock_rerank):
        mock_rerank.return_value = MagicMock(
            results=[
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.80},
            ]
        )

        r = JinaReranker()
        results = [_make_result("apple"), _make_result("banana")]
        out = r.rerank("fruit", results, top_k=2)

        assert len(out) == 2
        assert out[0].score == pytest.approx(0.95)
        assert out[0].text == "banana"  # index 1 comes first

    @patch("litellm.rerank", side_effect=Exception("API error"))
    def test_graceful_fallback_on_error(self, mock_rerank):
        r = JinaReranker()
        results = [_make_result("a"), _make_result("b")]
        out = r.rerank("q", results, top_k=2)
        assert len(out) == 2
        assert out[0].text == "a"  # original order preserved

    def test_repr(self):
        assert "jina" in repr(JinaReranker())


class TestResolveReranker:
    def test_none_returns_none(self):
        assert resolve_reranker(None) is None

    def test_false_returns_none(self):
        assert resolve_reranker(False) is None

    def test_true_returns_cross_encoder(self):
        r = resolve_reranker(True)
        assert isinstance(r, CrossEncoderReranker)

    def test_string_jina_returns_jina(self):
        r = resolve_reranker("jina-reranker-v2")
        assert isinstance(r, JinaReranker)

    def test_string_cross_encoder_returns_cross_encoder(self):
        r = resolve_reranker("cross-encoder/ms-marco-MiniLM-L-12-v2")
        assert isinstance(r, CrossEncoderReranker)

    def test_passthrough_existing_reranker(self):
        class _Custom:
            def rerank(self, query, results, top_k=None):
                return results[: (top_k or 5)]

        custom = _Custom()
        assert resolve_reranker(custom) is custom
