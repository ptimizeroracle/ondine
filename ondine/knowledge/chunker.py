"""Semantic chunking using sentence-level embedding similarity.

Splits documents into semantically coherent chunks by detecting
breakpoints where consecutive sentence embeddings diverge. This
produces chunks that respect topic boundaries rather than cutting
at arbitrary character counts.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    """A text chunk with provenance metadata and a stable identifier."""

    chunk_id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def _sentence_split(text: str) -> list[str]:
    """Split text into sentences on common delimiters while preserving non-empty results."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


class SemanticChunker:
    """Chunk documents by detecting semantic breakpoints.

    Sentences are embedded, then consecutive similarity is measured.
    A new chunk starts whenever the similarity drops below
    ``breakpoint_percentile`` of all observed inter-sentence similarities.

    If no embedding model is available, falls back to fixed-size
    sentence-count windows — gracefully degrading rather than failing
    (Ousterhout: define errors out of existence).

    Args:
        max_chunk_tokens: Soft upper bound on chunk size in whitespace tokens.
        breakpoint_percentile: Percentile threshold for breakpoint detection
            (0.0–1.0). Lower = more aggressive splitting.
        model_name: ``sentence-transformers`` model used for embedding.
    """

    def __init__(
        self,
        *,
        max_chunk_tokens: int = 512,
        breakpoint_percentile: float = 0.25,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._max_tokens = max_chunk_tokens
        self._percentile = breakpoint_percentile
        self._model_name = model_name
        self._model = None  # lazy-loaded

    def chunk(
        self, text: str, source: str, base_metadata: dict | None = None
    ) -> list[Chunk]:
        """Split *text* into semantic chunks."""
        sentences = _sentence_split(text)
        if not sentences:
            return []

        meta = base_metadata or {}
        embeddings = self._try_embed(sentences)

        if embeddings is not None:
            return self._semantic_split(sentences, embeddings, source, meta)
        return self._fixed_split(sentences, source, meta)

    # ── private ───────────────────────────────────────────────────

    def _try_embed(self, sentences: list[str]):
        """Return sentence embeddings or None if unavailable."""
        try:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
            return self._model.encode(sentences, show_progress_bar=False)
        except ImportError:
            logger.info(
                "sentence-transformers not installed; using fixed-size chunking"
            )
            return None
        except Exception:
            logger.warning(
                "Embedding failed; falling back to fixed-size chunking", exc_info=True
            )
            return None

    def _semantic_split(
        self, sentences: list[str], embeddings, source: str, meta: dict
    ) -> list[Chunk]:
        import numpy as np

        sims: list[float] = []
        for i in range(len(embeddings) - 1):
            a, b = embeddings[i], embeddings[i + 1]
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
            sims.append(cos)

        if not sims:
            return [self._make_chunk(" ".join(sentences), source, meta)]

        threshold = float(np.percentile(sims, self._percentile * 100))

        chunks: list[Chunk] = []
        current: list[str] = [sentences[0]]
        token_count = len(sentences[0].split())

        for i, sim in enumerate(sims):
            next_sent = sentences[i + 1]
            next_tokens = len(next_sent.split())

            if sim < threshold or token_count + next_tokens > self._max_tokens:
                chunks.append(self._make_chunk(" ".join(current), source, meta))
                current = [next_sent]
                token_count = next_tokens
            else:
                current.append(next_sent)
                token_count += next_tokens

        if current:
            chunks.append(self._make_chunk(" ".join(current), source, meta))

        return chunks

    def _fixed_split(
        self, sentences: list[str], source: str, meta: dict
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        current: list[str] = []
        token_count = 0

        for sent in sentences:
            tokens = len(sent.split())
            if token_count + tokens > self._max_tokens and current:
                chunks.append(self._make_chunk(" ".join(current), source, meta))
                current = []
                token_count = 0
            current.append(sent)
            token_count += tokens

        if current:
            chunks.append(self._make_chunk(" ".join(current), source, meta))

        return chunks

    @staticmethod
    def _make_chunk(text: str, source: str, meta: dict) -> Chunk:
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            text=text,
            source=source,
            metadata=meta,
        )
