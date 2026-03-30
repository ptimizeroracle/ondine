"""Pluggable context store for Ondine's anti-hallucination pipeline.

The context store system provides persistent evidence storage, retrieval,
grounding verification, contradiction detection, confidence scoring, and
evidence priming. Every LLM response processed by the pipeline can be
stored as an evidence record, then retrieved later to ground future
responses and detect contradictions.

Backends
--------
- ``RustContextStore`` -- High-performance default backed by compiled Rust
  with SQLite/FTS5.  Supports full-text search, TF-IDF + optional dense
  embedding grounding, and contradiction tracking.
- ``ZepContextStore`` -- Cloud-hosted knowledge graph via Zep Cloud API.
- ``InMemoryContextStore`` -- Pure-Python fallback for testing and
  prototyping (no persistence).

Core data classes
-----------------
- ``EvidenceRecord`` -- A single piece of evidence with text, source
  reference, claim type, confidence, and arbitrary metadata.
- ``RetrievalResult`` -- A scored result returned from a context store
  search, including support count for contradiction-aware ranking.
- ``GroundingResult`` -- The outcome of grounding an LLM claim against
  source sentences, carrying a confidence score and grounded/ungrounded flag.

Key capabilities
----------------
- **Evidence storage & retrieval** -- Store validated LLM outputs as
  evidence records and retrieve them by claim ID or semantic search.
- **Grounding verification** -- Verify new LLM responses against source
  sentences using TF-IDF cosine similarity or dense embeddings.  Claims
  below the confidence threshold are flagged as ungrounded.
- **Contradiction detection** -- Flag pairs of claims as contradictory
  and query the contradiction graph for a given claim.
- **Confidence scoring** -- Each evidence record carries an optional
  confidence float that downstream stages can use for filtering.
- **Evidence priming** -- Prior validated evidence is injected into
  prompts before LLM inference so the model can leverage earlier answers
  for consistency (see ``EvidenceRetrievalStage``).

Examples
--------
Standalone usage::

    from ondine.context import RustContextStore, EvidenceRecord

    store = RustContextStore("evidence.db")
    claim_id = store.store(EvidenceRecord(
        text="Paris is the capital of France.",
        source_ref="geography-101",
        claim_type="factual",
        confidence=0.95,
    ))
    results = store.search("capital of France", limit=3)
    store.close()

With the pipeline builder API::

    from ondine.api import PipelineBuilder
    from ondine.context import RustContextStore

    pipeline = (
        PipelineBuilder.create()
        .from_csv("data.csv", input_columns=["question"], output_columns=["answer"])
        .with_prompt("Answer: {question}")
        .with_llm(model="openai/gpt-4o-mini")
        .with_context_store(RustContextStore("evidence.db"))
        .with_grounding(threshold=0.3)
        .with_evidence_priming(query_columns=["question"], top_k=3)
        .build()
    )
"""

from ondine.context.protocol import (
    ContextStore,
    EvidenceRecord,
    GroundingResult,
    RetrievalResult,
)

__all__ = [
    "ContextStore",
    "EvidenceRecord",
    "GroundingResult",
    "RetrievalResult",
    "RustContextStore",
    "ZepContextStore",
    "InMemoryContextStore",
]


def __getattr__(name: str):
    if name == "RustContextStore":
        from ondine.context.rust_store import RustContextStore

        return RustContextStore
    if name == "ZepContextStore":
        from ondine.context.zep_store import ZepContextStore

        return ZepContextStore
    if name == "InMemoryContextStore":
        from ondine.context.memory_store import InMemoryContextStore

        return InMemoryContextStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
