"""In-memory context store for testing and fallback.

Uses pure Python TF-IDF (no Rust dependency). Not suitable for production
workloads but useful for unit tests and environments without a Rust toolchain.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from ondine.context.protocol import (
    ContextStore,
    EvidenceRecord,
    GroundingResult,
    RetrievalResult,
)
from ondine.context.text import tfidf_cosine_similarity


@dataclass
class _StoredRecord:
    record: EvidenceRecord
    support_count: int = 1


class InMemoryContextStore(ContextStore):
    """Pure-Python in-memory context store (testing/fallback)."""

    def __init__(self):
        self._records: dict[str, _StoredRecord] = {}
        self._contradictions: dict[str, set[str]] = {}

    def store(self, record: EvidenceRecord) -> str:
        claim_id = record.claim_id or str(uuid.uuid4())
        record.claim_id = claim_id

        if claim_id in self._records:
            self._records[claim_id].support_count += 1
        else:
            self._records[claim_id] = _StoredRecord(record=record)

        return claim_id

    def retrieve(self, claim_id: str) -> EvidenceRecord | None:
        stored = self._records.get(claim_id)
        return stored.record if stored else None

    def search(self, query: str, limit: int = 5) -> list[RetrievalResult]:
        scored = []
        for cid, stored in self._records.items():
            sim = tfidf_cosine_similarity(query, stored.record.text)
            if sim > 0.0:
                scored.append(
                    RetrievalResult(
                        text=stored.record.text,
                        score=sim,
                        claim_id=cid,
                        source_ref=stored.record.source_ref,
                        support_count=stored.support_count,
                    )
                )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:limit]

    def ground(
        self,
        response_text: str,
        source_sentences: list[str],
        threshold: float = 0.3,
    ) -> list[GroundingResult]:
        best_sim = 0.0
        for sentence in source_sentences:
            sim = tfidf_cosine_similarity(response_text, sentence)
            if sim > best_sim:
                best_sim = sim

        if best_sim < threshold:
            return []

        claim_id = str(uuid.uuid4())
        record = EvidenceRecord(
            text=response_text,
            source_ref="grounding",
            claim_type="factual",
            source_type="llm_response",
            asserted_by="grounding_engine",
            claim_id=claim_id,
            confidence=best_sim,
        )
        self.store(record)

        return [
            GroundingResult(
                claim_id=claim_id,
                claim_text=response_text,
                source="grounding",
                confidence=best_sim,
                grounded=True,
            )
        ]

    def add_contradiction(self, claim_a_id: str, claim_b_id: str) -> None:
        self._contradictions.setdefault(claim_a_id, set()).add(claim_b_id)
        self._contradictions.setdefault(claim_b_id, set()).add(claim_a_id)

    def get_contradictions(self, claim_id: str) -> list[str]:
        return list(self._contradictions.get(claim_id, set()))

    def close(self) -> None:
        self._records.clear()
        self._contradictions.clear()
