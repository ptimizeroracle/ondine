"""Tests for RAG protocols — verify structural subtyping works.

Any object with the right method signature must satisfy the protocol,
without inheriting from it. This ensures third-party components can
plug in without wrapping.
"""

import pytest

from ondine.knowledge.protocols import (
    Embedder,
    EvalResult,
    QueryTransformer,
    Reranker,
    RetrievalScorer,
)


class _MockEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


class _MockReranker:
    def rerank(self, query: str, results: list, top_k: int | None = None) -> list:
        return results[: (top_k or 5)]


class _MockTransformer:
    def transform(self, query: str) -> list[str]:
        return [query, f"{query} rephrased"]


class _MockScorer:
    def score(self, query: str, answer: str, contexts: list[str]) -> EvalResult:
        return EvalResult(faithfulness=1.0, relevancy=0.9, context_precision=0.8)


class TestProtocolStructuralSubtyping:
    def test_embedder_isinstance(self):
        assert isinstance(_MockEmbedder(), Embedder)

    def test_reranker_isinstance(self):
        assert isinstance(_MockReranker(), Reranker)

    def test_query_transformer_isinstance(self):
        assert isinstance(_MockTransformer(), QueryTransformer)

    def test_retrieval_scorer_isinstance(self):
        assert isinstance(_MockScorer(), RetrievalScorer)

    def test_plain_object_not_embedder(self):
        assert not isinstance(object(), Embedder)

    def test_runtime_checkable_checks_method_names(self):
        # runtime_checkable only verifies method existence, not signatures.
        # Static type checkers (mypy/pyright) catch argument mismatches.
        class _HasEmbed:
            def embed(self, text: str) -> list[float]:
                return [0.1]

        assert isinstance(_HasEmbed(), Embedder)  # method name present → True


class TestEvalResult:
    def test_defaults(self):
        r = EvalResult()
        assert r.faithfulness == 0.0
        assert r.relevancy == 0.0
        assert r.context_precision == 0.0
        assert r.metadata == {}

    def test_custom_values(self):
        r = EvalResult(
            faithfulness=0.95,
            relevancy=0.8,
            context_precision=0.7,
            metadata={"model": "test"},
        )
        assert r.faithfulness == 0.95
        assert r.metadata == {"model": "test"}

    def test_frozen(self):
        r = EvalResult()
        with pytest.raises(AttributeError):
            r.faithfulness = 0.5  # type: ignore[misc]
