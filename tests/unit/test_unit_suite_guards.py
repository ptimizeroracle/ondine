"""The unit suite's own guarantees, asserted rather than assumed.

The guard in conftest is invisible when it works and invisible when it
silently stops working — a test that forgot to inject an embedder would just
start downloading models again, and nothing would say so until the day
huggingface.co was unreachable (#221).
"""

import pytest

from ondine.knowledge.chunker import SemanticChunker
from ondine.knowledge.embedders import SentenceTransformerEmbedder
from ondine.knowledge.reranker import CrossEncoderReranker


def test_loading_a_real_embedding_model_is_refused():
    """Reaching the real model means a test forgot to inject a fake."""
    with pytest.raises(RuntimeError, match="must not load models"):
        SentenceTransformerEmbedder()._load()


def test_chunker_does_not_embed():
    """SemanticChunker loads its own model; injection cannot reach it."""
    assert SemanticChunker()._try_embed(["one sentence", "another"]) is None


def test_reranker_reports_no_model():
    """Falls back to truncation, the same path taken when unavailable."""
    assert CrossEncoderReranker()._get_model() is None
