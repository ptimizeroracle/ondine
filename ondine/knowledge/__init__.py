"""Knowledge Base module for hybrid retrieval-augmented generation.

This module implements the full RAG lifecycle: document ingestion, semantic
chunking, embedding, hybrid search (BM25 + dense vectors), cross-encoder
reranking, query transformation, OCR for images and PDFs, and LLM-as-judge
evaluation -- all backed by the Rust ``EvidenceDB`` engine with pluggable
Python components.

Facade
------
- ``KnowledgeStore`` -- The primary entry point.  Wraps the Rust EvidenceDB
  and exposes a deep-module interface: callers interact via ``ingest()`` and
  ``search()`` only; all intermediate steps (chunking, embedding, FTS5/dense
  indexing, query transformation, reranking) are hidden.

Protocols (ABCs)
----------------
- ``Embedder`` -- Embed a list of texts into dense vectors.
- ``Reranker`` -- Re-score candidate chunks against the original query.
- ``QueryTransformer`` -- Expand or rewrite a user query before retrieval.
- ``RetrievalScorer`` -- Score retrieval quality (precision/recall).
- ``OCRProvider`` -- Extract text from images or PDF pages.
- ``EvalResult`` -- Structured output from RAG evaluation.

Embedders
---------
- ``SentenceTransformerEmbedder`` -- Local embedding via sentence-transformers
  (default: ``BAAI/bge-base-en-v1.5``).
- ``OpenAIEmbedder`` -- Remote embedding via the OpenAI API.

Rerankers
---------
- ``CrossEncoderReranker`` -- Local cross-encoder reranking
  (default: ``cross-encoder/ms-marco-MiniLM-L-12-v2``).
- ``JinaReranker`` -- Remote reranking via the Jina AI API.

Query transformers
------------------
- ``MultiQueryTransformer`` -- Generate multiple query reformulations and
  merge their result sets for broader recall.
- ``HyDETransformer`` -- Hypothetical Document Embeddings: generate a
  synthetic answer, embed it, and retrieve by dense similarity.
- ``StepBackTransformer`` -- Step-back prompting: generate a more abstract
  version of the query for better conceptual retrieval.

OCR providers
-------------
- ``VisionOCR`` -- Extract text from images using a vision LLM (e.g.
  ``openai/gpt-4o-mini``).
- ``TesseractOCR`` -- Local OCR via Tesseract.
- ``DocTROCR`` -- Local OCR via the DocTR deep-learning model.

Document handling
-----------------
- ``DocumentLoader`` -- Load documents from files (PDF, DOCX, TXT, HTML,
  Markdown, images) or directories with recursive globbing.
- ``SemanticChunker`` -- Split documents into semantically coherent chunks
  using sentence-boundary detection and configurable overlap.

Evaluation
----------
- ``LLMJudge`` -- LLM-as-judge evaluation of RAG answers for
  faithfulness, relevance, and completeness.

Examples
--------
Basic ingest and search::

    from ondine.knowledge import KnowledgeStore

    kb = KnowledgeStore("knowledge.db")
    kb.ingest("docs/")
    results = kb.search("How does authentication work?", top_k=5)
    for r in results:
        print(f"[{r.score:.2f}] {r.text[:80]}...")

With reranking and query expansion::

    kb = KnowledgeStore(
        "knowledge.db",
        reranker=True,                    # default cross-encoder
        query_transform="multi-query",    # generate multiple reformulations
    )
    kb.ingest("docs/")
    results = kb.search("authentication flow", top_k=5)

With the pipeline builder API::

    from ondine.api import PipelineBuilder
    from ondine.knowledge import KnowledgeStore

    kb = KnowledgeStore("knowledge.db", reranker=True)
    kb.ingest("docs/")

    pipeline = (
        PipelineBuilder.create()
        .from_csv("questions.csv", input_columns=["question"], output_columns=["answer"])
        .with_knowledge_base(kb, top_k=5, rerank=True, query_transform="hyde")
        .with_prompt("Context:\\n{_kb_context}\\n\\nQuestion: {question}\\nAnswer:")
        .with_llm(model="openai/gpt-4o-mini")
        .build()
    )

OCR-enabled ingest for images and PDFs::

    kb = KnowledgeStore("knowledge.db", ocr="vision", extract_pdf_images=True)
    kb.ingest("scanned_docs/")
"""

from ondine.knowledge.protocols import (
    Embedder,
    EvalResult,
    OCRProvider,
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
    "OCRProvider",
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
    "VisionOCR",
    "TesseractOCR",
    "DocTROCR",
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
        "VisionOCR": ("ondine.knowledge.ocr", "VisionOCR"),
        "TesseractOCR": ("ondine.knowledge.ocr", "TesseractOCR"),
        "DocTROCR": ("ondine.knowledge.ocr", "DocTROCR"),
    }

    if name in _lazy:
        module_path, attr = _lazy[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
