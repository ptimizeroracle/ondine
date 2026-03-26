"""Protocols for pluggable RAG components.

Defines structural interfaces (``typing.Protocol``) so that any object
with the right method signatures can be used — no inheritance required.
Third-party embedders, rerankers, or evaluators work out of the box as
long as they satisfy the protocol shape.

All protocols are ``@runtime_checkable`` for use with ``isinstance()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# ── Embedder ──────────────────────────────────────────────────────


@runtime_checkable
class Embedder(Protocol):
    """Encode texts into dense vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


# ── Reranker ──────────────────────────────────────────────────────


@runtime_checkable
class Reranker(Protocol):
    """Re-score search results using a cross-attention model."""

    def rerank(self, query: str, results: list, top_k: int | None = None) -> list: ...


# ── Query Transformer ────────────────────────────────────────────


@runtime_checkable
class QueryTransformer(Protocol):
    """Transform a single query into one or more retrieval queries."""

    def transform(self, query: str) -> list[str]: ...


# ── Evaluation ────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvalResult:
    """Scores produced by a RAG evaluator."""

    faithfulness: float = 0.0
    relevancy: float = 0.0
    context_precision: float = 0.0
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class RetrievalScorer(Protocol):
    """Score a RAG answer for faithfulness / relevancy."""

    def score(self, query: str, answer: str, contexts: list[str]) -> EvalResult: ...
