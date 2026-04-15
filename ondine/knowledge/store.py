"""KnowledgeStore — unified ingest, embed, and search facade.

Wraps the Rust ``EvidenceDB`` for KB chunk storage and provides a
single high-level interface for the entire knowledge-base lifecycle:
ingest documents -> chunk -> embed -> store -> search.

Callers interact with ``ingest()`` and ``search()`` only; all
intermediate steps (chunking, embedding, FTS5/dense indexing, query
transformation, reranking) are hidden behind the module boundary
(Ousterhout: deep module).
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ondine.knowledge.chunker import Chunk, SemanticChunker
from ondine.knowledge.embedders import resolve_embedder
from ondine.knowledge.loader import Document, DocumentLoader
from ondine.knowledge.query import resolve_query_transform
from ondine.knowledge.reranker import resolve_reranker

if TYPE_CHECKING:
    from pathlib import Path

    from ondine.knowledge.protocols import (
        Embedder,
        OCRProvider,
        QueryTransformer,
        Reranker,
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """A scored KB chunk returned by hybrid search."""

    chunk_id: str
    text: str
    source: str
    score: float
    metadata: dict = field(default_factory=dict)


EmbedFn = Callable[[list[str]], list[list[float]]]


class _EmbedFnAdapter:
    """Adapt a legacy ``EmbedFn`` callable to the ``Embedder`` protocol."""

    def __init__(self, fn: EmbedFn) -> None:
        self._fn = fn

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._fn(texts)

    def __repr__(self) -> str:
        return f"_EmbedFnAdapter({self._fn!r})"


class KnowledgeStore:
    """End-to-end knowledge base: ingest -> chunk -> embed -> hybrid search.

    Uses the Rust ``EvidenceDB`` for storage and retrieval when available,
    falling back to a pure-Python in-memory store otherwise.

    Args:
        db_path: SQLite path or ``":memory:"``.
        chunker: Custom ``SemanticChunker``, or ``None`` for defaults.
        embedder: An ``Embedder``-protocol object, a model-name string,
            or ``None`` for auto-detection (``SentenceTransformerEmbedder``
            with ``BAAI/bge-base-en-v1.5`` if available).
        reranker: A ``Reranker``-protocol object, a model-name string,
            ``True`` for the default cross-encoder, or ``None``/``False``
            to disable.
        query_transform: A ``QueryTransformer``-protocol object, a
            strategy-name string (``"multi-query"``, ``"hyde"``,
            ``"step-back"``), or ``None`` to disable.
        ocr: An ``OCRProvider``-protocol object, a shortcut string
            (``"vision"``, ``"tesseract"``, ``"doctr"``, or a litellm
            model name), or ``None`` to skip image files during ingest.
        extract_pdf_images: When ``True`` and ``ocr`` is configured,
            also extract and OCR embedded images from PDF pages.
        embed_fn: **Deprecated.** Legacy callable ``list[str] ->
            list[list[float]]``. Use ``embedder`` instead.
        embed_model_name: Model-name label stored alongside embeddings.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        chunker: SemanticChunker | None = None,
        embedder: Embedder | str | None = None,
        reranker: Reranker | str | bool | None = None,
        query_transform: QueryTransformer | str | None = None,
        ocr: OCRProvider | str | None = None,
        extract_pdf_images: bool = False,
        embed_fn: EmbedFn | None = None,
        embed_model_name: str = "BAAI/bge-base-en-v1.5",
    ) -> None:
        self._db = self._open_db(db_path)
        self._chunker = chunker or SemanticChunker()
        self._loader = DocumentLoader(ocr=ocr, extract_pdf_images=extract_pdf_images)

        # Resolve embedder — honour legacy embed_fn for backward compat
        if embed_fn is not None:
            warnings.warn(
                "embed_fn is deprecated; use embedder= instead",
                DeprecationWarning,
                stacklevel=2,
            )
            self._embedder: Embedder | None = _EmbedFnAdapter(embed_fn)
        else:
            self._embedder = resolve_embedder(embedder)

        self._embed_model = embed_model_name
        self._reranker: Reranker | None = resolve_reranker(reranker)
        self._query_transform: QueryTransformer | None = resolve_query_transform(
            query_transform
        )

    # ── public API ────────────────────────────────────────────────

    def ingest(self, path: str | Path) -> int:
        """Load, chunk, store, and optionally embed all documents at *path*.

        Returns the number of chunks stored.
        """
        docs = self._loader.load(path)
        return self.ingest_documents(docs)

    def ingest_documents(self, docs: list[Document]) -> int:
        """Chunk, store, and embed pre-loaded ``Document`` objects."""
        count = 0
        for doc in docs:
            chunks = self._chunker.chunk(doc.text, doc.source, doc.metadata)
            for chunk in chunks:
                self._store_chunk(chunk)
                count += 1

        if self._embedder is not None:
            embedded = self._embed_pending()
            logger.info("Embedded %d pending chunks", embedded)

        logger.info("Ingested %d chunks from %d documents", count, len(docs))
        return count

    def ingest_text(
        self, text: str, source: str = "inline", metadata: dict | None = None
    ) -> int:
        """Convenience: chunk and store raw text without file I/O."""
        doc = Document(text=text, source=source, metadata=metadata or {})
        return self.ingest_documents([doc])

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Hybrid search (FTS5 + dense + RRF) with optional query
        transformation and reranking.

        Flow:
        1. If a ``QueryTransformer`` is configured, expand the query
           into multiple retrieval queries.
        2. Run hybrid search for each query variant via Rust.
        3. Deduplicate results across variants.
        4. If a ``Reranker`` is configured, rerank the merged results.
        5. Return top-*limit* ``SearchResult`` objects.
        """
        # Step 1: query expansion
        if self._query_transform is not None:
            queries = self._query_transform.transform(query)
        else:
            queries = [query]

        # Step 2: retrieve for each variant
        seen: dict[str, SearchResult] = {}
        fetch_limit = limit * 3 if len(queries) > 1 else limit * 2

        for q in queries:
            raw = self._query_chunks(q, fetch_limit)
            for r in raw:
                chunk_id = r[0]
                if chunk_id not in seen or r[4] > seen[chunk_id].score:
                    seen[chunk_id] = SearchResult(
                        chunk_id=chunk_id,
                        text=r[1],
                        source=r[2],
                        score=r[4],
                        metadata=json.loads(r[3]) if isinstance(r[3], str) else r[3],
                    )

        # Step 3: collect deduplicated results sorted by score
        results = sorted(seen.values(), key=lambda r: r.score, reverse=True)

        # Step 4: rerank
        if self._reranker is not None and results:
            results = self._reranker.rerank(query, results, top_k=limit)
        else:
            results = results[:limit]

        return results

    @property
    def chunk_count(self) -> int:
        """Number of chunks currently stored."""
        return int(self._db.chunk_count())

    # ── private: Rust-backed or fallback ──────────────────────────

    def _open_db(self, path: str):
        try:
            from ondine import _engine

            return _engine.EvidenceDB(path)
        except ImportError:
            logger.info("Rust engine unavailable; using in-memory fallback")
            return _InMemoryChunkDB()

    def _store_chunk(self, chunk: Chunk) -> None:
        self._db.store_chunk(
            chunk.chunk_id,
            chunk.text,
            chunk.source,
            json.dumps(chunk.metadata),
        )

    def _query_chunks(self, query: str, limit: int):
        result_json = self._db.query_chunks(query, limit)
        if isinstance(result_json, str):
            return json.loads(result_json)
        return result_json

    def _embed_pending(self) -> int:
        if self._embedder is None:
            return 0

        def _callback(texts: list[str]) -> list[list[float]]:
            return self._embedder.embed(texts)  # type: ignore[union-attr]

        return int(self._db.embed_pending_chunks(_callback, self._embed_model))


class _InMemoryChunkDB:
    """Pure-Python fallback when the Rust extension is unavailable.

    Provides the same method signatures as ``EvidenceDB`` for the KB
    subset so callers never need to care which backend is active.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, dict] = {}

    def store_chunk(
        self, chunk_id: str, text: str, source: str, metadata_json: str = "{}"
    ) -> None:
        self._chunks[chunk_id] = {
            "text": text,
            "source": source,
            "metadata": metadata_json,
        }

    def query_chunks(self, question: str, limit: int = 5) -> list[tuple]:
        tokens = set(question.lower().split())
        scored: list[tuple[str, str, str, str, float]] = []
        for cid, data in self._chunks.items():
            overlap = sum(1 for t in tokens if t in data["text"].lower())
            if overlap > 0:
                scored.append(
                    (
                        cid,
                        data["text"],
                        data["source"],
                        data["metadata"],
                        float(overlap),
                    )
                )
        scored.sort(key=lambda x: x[4], reverse=True)
        return scored[:limit]

    def embed_pending_chunks(self, _callback, _model: str) -> int:
        return 0

    def chunk_count(self) -> int:
        return len(self._chunks)
