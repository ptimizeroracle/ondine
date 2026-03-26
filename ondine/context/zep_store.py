"""Zep Cloud-backed context store.

Adapted from Parallax's zep_memory.py. Implements the ContextStore protocol
using Zep Cloud's knowledge graph API for storage and hybrid search.

Requires: pip install ondine[zep]
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from ondine.context.protocol import (
    ContextStore,
    EvidenceRecord,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

_client_cache: dict[str, Any] = {}
_created_graphs: set[str] = set()


def _get_client(api_key: str | None = None) -> Any | None:
    """Lazily create and cache a Zep client."""
    key = api_key or os.environ.get("ZEP_API_KEY")
    if not key:
        return None

    if key in _client_cache:
        return _client_cache[key]

    try:
        from zep_cloud.client import Zep

        client = Zep(api_key=key)
        _client_cache[key] = client
        logger.info("Zep Cloud client initialized")
        return client
    except ImportError:
        logger.warning(
            "zep-cloud package not installed — install with: pip install ondine[zep]"
        )
        return None
    except Exception as e:
        logger.debug("Zep client init failed: %s", e)
        return None


def _ensure_graph(client: Any, graph_id: str) -> None:
    """Create the graph if it doesn't exist yet (idempotent per session)."""
    if graph_id in _created_graphs:
        return
    try:
        client.graph.create(graph_id=graph_id)
        logger.info("Zep: created graph %s", graph_id)
    except Exception as e:
        if "already exists" in str(e).lower() or "409" in str(e):
            logger.debug("Zep: graph %s already exists", graph_id)
        else:
            logger.warning("Zep: graph create for %s: %s", graph_id, e)
    _created_graphs.add(graph_id)


class ZepContextStore(ContextStore):
    """Context store backed by Zep Cloud knowledge graph.

    Each pipeline run can use its own graph_id, or share one for
    cross-run memory. Zep automatically extracts entities and
    relationships from stored text.
    """

    def __init__(
        self,
        graph_id: str | None = None,
        api_key: str | None = None,
    ):
        self._api_key = api_key
        self._graph_id = graph_id or str(uuid.uuid4())
        self._client = _get_client(api_key)
        if self._client:
            _ensure_graph(self._client, self._graph_id)
        self._local_records: dict[str, EvidenceRecord] = {}

    @property
    def available(self) -> bool:
        return self._client is not None

    def store(self, record: EvidenceRecord) -> str:
        claim_id = record.claim_id or str(uuid.uuid4())
        record.claim_id = claim_id

        self._local_records[claim_id] = record

        if self._client:
            try:
                formatted = f"[{record.claim_type.upper()}] {record.text}"
                self._client.graph.add(
                    graph_id=self._graph_id,
                    type="text",
                    data=formatted,
                )
            except Exception as e:
                logger.debug("Zep store failed for %s: %s", claim_id, e)

        return claim_id

    def retrieve(self, claim_id: str) -> EvidenceRecord | None:
        return self._local_records.get(claim_id)

    def search(self, query: str, limit: int = 5) -> list[RetrievalResult]:
        if not self._client:
            return []

        try:
            results = self._client.graph.search(
                graph_id=self._graph_id,
                query=query[:400],
                scope="edges",
                limit=limit,
                reranker="cross_encoder",
            )

            parsed = []
            if results and hasattr(results, "edges") and results.edges:
                for edge in results.edges:
                    parsed.append(
                        RetrievalResult(
                            text=getattr(edge, "fact", "") or "",
                            score=getattr(edge, "score", 0.0),
                            source_ref=getattr(edge, "name", ""),
                        )
                    )
            if results and hasattr(results, "nodes") and results.nodes:
                for node in results.nodes:
                    parsed.append(
                        RetrievalResult(
                            text=getattr(node, "summary", "")
                            or getattr(node, "name", ""),
                            score=getattr(node, "score", 0.0),
                            source_ref=getattr(node, "name", ""),
                        )
                    )

            return parsed[:limit]
        except Exception as e:
            logger.debug("Zep search failed: %s", e)
            return []

    def close(self) -> None:
        self._local_records.clear()
        self._client = None
