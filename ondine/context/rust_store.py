"""Rust-backed context store using the compiled ondine._engine module.

This is the default high-performance backend. It uses SQLite + FTS5
for hybrid search (dense + sparse via RRF) all running at native speed.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from ondine.context.protocol import (
    ContextStore,
    EvidenceRecord,
    GroundingResult,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


def _import_engine():
    try:
        from ondine import _engine

        return _engine
    except ImportError:
        raise ImportError(
            "ondine._engine (Rust extension) is not available. "
            "Install with: pip install ondine  (requires Rust toolchain for source builds)"
        )


class RustContextStore(ContextStore):
    """High-performance context store backed by Rust + SQLite/FTS5."""

    def __init__(self, path: str = ":memory:"):
        engine = _import_engine()
        self._db = engine.EvidenceDB(path)
        self._path = path

    def store(self, record: EvidenceRecord) -> str:
        claim_id = record.claim_id or str(uuid.uuid4())
        decision_id = record.metadata.get("decision_id", str(uuid.uuid4()))

        claim_json = json.dumps(
            {
                "claim_id": claim_id,
                "text": record.text,
                "claim_type": record.claim_type.capitalize(),
                "source_type": _map_source_type(record.source_type),
                "source_ref": record.source_ref,
                "asserted_by": record.asserted_by,
                "asserted_during": {
                    "decision_id": decision_id,
                    "phase": record.metadata.get("phase", 0),
                    "round": record.metadata.get("round"),
                },
                "valid_from": None,
                "superseded_by": None,
                "contradiction_of": [],
            }
        )

        return str(self._db.store_claim(claim_json))

    def retrieve(self, claim_id: str) -> EvidenceRecord | None:
        try:
            claim_json = self._db.get_claim(claim_id)
            claim = json.loads(claim_json)
            return EvidenceRecord(
                text=claim["text"],
                source_ref=claim.get("source_ref", ""),
                claim_type=claim.get("claim_type", "Factual").lower(),
                source_type=_reverse_source_type(
                    claim.get("source_type", "LlmResponse")
                ),
                asserted_by=claim.get("asserted_by", ""),
                claim_id=claim["claim_id"],
            )
        except Exception:
            return None

    def search(self, query: str, limit: int = 5) -> list[RetrievalResult]:
        results_json = self._db.query(query, limit)
        results = json.loads(results_json)
        return [
            RetrievalResult(
                text=r["claim"]["text"],
                score=r["relevance_score"],
                claim_id=r["claim"]["claim_id"],
                source_ref=r["claim"].get("source_ref", ""),
                support_count=r.get("support_count", 0),
            )
            for r in results
        ]

    def ground(
        self,
        response_text: str,
        source_sentences: list[str],
        threshold: float = 0.3,
        embed_fn: Callable[..., Any] | None = None,
    ) -> list[GroundingResult]:
        decision_id = str(uuid.uuid4())
        raw_claims = json.dumps(
            [
                {
                    "text": response_text,
                    "claim_type": "factual",
                    "source_location": "llm_output",
                    "doc_id": "source_doc",
                }
            ]
        )
        doc_sentences = json.dumps(
            [
                {
                    "doc_id": "source_doc",
                    "sentences": source_sentences,
                }
            ]
        )

        result_json = self._db.ground_and_store(
            decision_id, raw_claims, doc_sentences, threshold
        )
        grounded = json.loads(result_json)

        # Augment with embedding similarity when an embed_fn is provided
        if embed_fn is not None and source_sentences:
            embed_boost = _best_embedding_similarity(
                embed_fn, response_text, source_sentences
            )
            for g in grounded:
                g["confidence"] = max(g["confidence"], embed_boost)

            # If TF-IDF returned nothing but embeddings exceed threshold,
            # synthesise a result so the claim is not silently dropped.
            if not grounded and embed_boost >= threshold:
                grounded = [
                    {
                        "claim_id": str(uuid.uuid4()),
                        "claim_text": response_text,
                        "source": "embedding",
                        "confidence": embed_boost,
                    }
                ]

        return [
            GroundingResult(
                claim_id=g["claim_id"],
                claim_text=g["claim_text"],
                source=g["source"],
                confidence=g["confidence"],
                grounded=g["confidence"] >= threshold,
            )
            for g in grounded
        ]

    def add_contradiction(self, claim_a_id: str, claim_b_id: str) -> None:
        self._db.add_contradiction(claim_a_id, claim_b_id)

    def get_contradictions(self, claim_id: str) -> list[str]:
        result_json = self._db.get_contradictions(claim_id)
        return list(json.loads(result_json))

    def close(self) -> None:
        self._db = None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _best_embedding_similarity(
    embed_fn: Callable[..., Any],
    response_text: str,
    source_sentences: list[str],
) -> float:
    """Return the best embedding cosine similarity between response and sources."""
    all_texts = [response_text] + list(source_sentences)
    embeddings = embed_fn(all_texts)
    response_emb = embeddings[0]
    best = 0.0
    for source_emb in embeddings[1:]:
        sim = _cosine_similarity(response_emb, source_emb)
        if sim > best:
            best = sim
    return best


def _map_source_type(source_type: str) -> str:
    mapping = {
        "document": "Document",
        "llm_response": "LlmResponse",
        "user_correction": "UserCorrection",
        "external": "External",
    }
    return mapping.get(source_type.lower(), "LlmResponse")


def _reverse_source_type(source_type: str) -> str:
    mapping = {
        "Document": "document",
        "LlmResponse": "llm_response",
        "UserCorrection": "user_correction",
        "External": "external",
    }
    return mapping.get(source_type, "llm_response")
