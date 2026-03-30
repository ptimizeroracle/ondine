"""Rust-backed context store using the compiled ondine._engine module.

This is the default high-performance backend. It uses SQLite + FTS5
for hybrid search (dense + sparse via RRF) all running at native speed.
"""

from __future__ import annotations

import json
import logging
import uuid

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

        return self._db.store_claim(claim_json)

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
        return json.loads(result_json)

    def close(self) -> None:
        self._db = None


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
