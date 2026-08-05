"""Unit-suite guarantees that hold no matter what a test forgets to inject."""

import pytest

from ondine.knowledge.chunker import SemanticChunker
from ondine.knowledge.embedders import SentenceTransformerEmbedder
from ondine.knowledge.reranker import CrossEncoderReranker


def _refuse_to_load(*_args, **_kwargs):
    raise RuntimeError(
        "Unit tests must not load models. Inject a fake instead — "
        "see DeterministicEmbedder in tests/conftest.py (#221)."
    )


@pytest.fixture(autouse=True)
def _no_model_downloads(monkeypatch):
    """Stop any unit test from loading a model, however it got there.

    Injecting a fake embedder covers ``KnowledgeStore``, but not every
    component that decides on its own to load one. ``SemanticChunker`` builds
    its own ``SentenceTransformer`` regardless of the store's embedder, and
    anything added later could do the same. A test that forgets to inject
    would silently start downloading again, and the suite would only say so on
    the day huggingface.co is unreachable — which cost 17 failures and 57
    minutes of retry backoff on #220 (#221).

    Patched at the load boundary rather than through ``HF_HUB_OFFLINE``.
    Environment variables cannot be scoped within a process: huggingface_hub
    reads that one into a module constant at import time, so setting it for
    tests/unit baked it in for the whole interpreter, and the CI step that
    runs the tree in one process (``pytest --cov``) carried it into
    tests/verification — where ``test_claim_36_knowledge_store_search`` embeds
    with a real model on purpose. ``monkeypatch`` undoes these attributes after
    each test, so the guard genuinely stops at this directory's edge.

    The two fallbacks are the ones the code already takes when a model is
    unavailable, so behaviour here is the documented degraded path rather than
    something invented for tests. The embedder instead raises, because reaching
    it means a test expected real embeddings and should say so loudly.
    """
    monkeypatch.setattr(SentenceTransformerEmbedder, "_load", _refuse_to_load)
    monkeypatch.setattr(SemanticChunker, "_try_embed", lambda self, sentences: None)
    monkeypatch.setattr(CrossEncoderReranker, "_get_model", lambda self: None)
