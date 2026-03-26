"""Pluggable context store for Ondine's anti-hallucination pipeline.

Provides a protocol (ABC) and multiple backends:
- RustContextStore: High-performance default backed by compiled Rust + SQLite/FTS5
- ZepContextStore: Cloud-hosted knowledge graph via Zep Cloud API
- InMemoryContextStore: Pure-Python fallback for testing
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
