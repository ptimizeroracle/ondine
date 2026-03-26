"""Knowledge Base module for hybrid retrieval-augmented generation.

Provides document ingestion, semantic chunking, hybrid search (BM25 + dense),
cross-encoder reranking, query transformation, and RAG evaluation — all
backed by the Rust EvidenceDB engine with pluggable Python components.
"""

from ondine.knowledge.protocols import (
    Embedder,
    EvalResult,
    QueryTransformer,
    Reranker,
    RetrievalScorer,
)
from ondine.knowledge.store import KnowledgeStore

__all__ = [
    # Facade
    "KnowledgeStore",
    # Protocols
    "Embedder",
    "Reranker",
    "QueryTransformer",
    "RetrievalScorer",
    "EvalResult",
    # Concrete implementations (lazy)
    "DocumentLoader",
    "SemanticChunker",
    "CrossEncoderReranker",
    "JinaReranker",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "MultiQueryTransformer",
    "HyDETransformer",
    "StepBackTransformer",
    "LLMJudge",
]


def __getattr__(name: str):
    _lazy = {
        "DocumentLoader": ("ondine.knowledge.loader", "DocumentLoader"),
        "SemanticChunker": ("ondine.knowledge.chunker", "SemanticChunker"),
        "CrossEncoderReranker": ("ondine.knowledge.reranker", "CrossEncoderReranker"),
        "JinaReranker": ("ondine.knowledge.reranker", "JinaReranker"),
        "SentenceTransformerEmbedder": (
            "ondine.knowledge.embedders",
            "SentenceTransformerEmbedder",
        ),
        "OpenAIEmbedder": ("ondine.knowledge.embedders", "OpenAIEmbedder"),
        "MultiQueryTransformer": ("ondine.knowledge.query", "MultiQueryTransformer"),
        "HyDETransformer": ("ondine.knowledge.query", "HyDETransformer"),
        "StepBackTransformer": ("ondine.knowledge.query", "StepBackTransformer"),
        "LLMJudge": ("ondine.knowledge.eval", "LLMJudge"),
    }

    if name in _lazy:
        module_path, attr = _lazy[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
