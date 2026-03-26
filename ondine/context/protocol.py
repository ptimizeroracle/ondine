"""Abstract protocol for pluggable context stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class EvidenceRecord:
    """A single piece of evidence to store or that was retrieved."""

    text: str
    source_ref: str = ""
    claim_type: str = "factual"
    source_type: str = "llm_response"
    asserted_by: str = "pipeline"
    claim_id: str | None = None
    confidence: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A result returned from a context store query."""

    text: str
    score: float
    claim_id: str = ""
    source_ref: str = ""
    support_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class GroundingResult:
    """Result of grounding an LLM response against source text."""

    claim_id: str
    claim_text: str
    source: str
    confidence: float
    grounded: bool = True


class ContextStore(ABC):
    """Abstract base class for context store backends.

    All backends must implement store, retrieve, search, and close.
    Optionally implement ground() and add_contradiction() for full
    anti-hallucination pipeline support.
    """

    @abstractmethod
    def store(self, record: EvidenceRecord) -> str:
        """Store an evidence record. Returns a claim_id string."""
        ...

    @abstractmethod
    def retrieve(self, claim_id: str) -> EvidenceRecord | None:
        """Retrieve a single evidence record by ID."""
        ...

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> list[RetrievalResult]:
        """Search for relevant evidence. Returns scored results."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the store."""
        ...

    def ground(
        self,
        response_text: str,
        source_sentences: list[str],
        threshold: float = 0.3,
    ) -> list[GroundingResult]:
        """Ground an LLM response against source sentences via TF-IDF.

        Default implementation returns empty (no grounding).
        Override in backends that support it.
        """
        return []

    def add_contradiction(self, claim_a_id: str, claim_b_id: str) -> None:
        """Flag two claims as contradictory.

        Default is a no-op. Override in backends that support it.
        """

    def get_contradictions(self, claim_id: str) -> list[str]:
        """Get IDs of claims that contradict the given claim.

        Default returns empty. Override in backends that support it.
        """
        return []
