"""Unit-suite guarantees that hold no matter what a test forgets to inject."""

import pytest


@pytest.fixture(autouse=True)
def _no_model_downloads(monkeypatch):
    """Forbid the unit suite from reaching Hugging Face.

    Injecting a fake embedder covers ``KnowledgeStore``, but not every
    component that can decide on its own to load a model — ``SemanticChunker``
    builds its own ``SentenceTransformer``, and so would anything added later.
    A test that forgets to inject would silently start downloading again, and
    the suite would only reveal it on the day the network is down.

    Left alone that is expensive: when huggingface.co was unreachable, 17
    tests failed and the job spent ~57 minutes in HF's retry backoff before
    giving up (#221). Offline mode turns any such attempt into an immediate
    failure that the caller's existing fallback handles, so behaviour stops
    depending on whether the network happens to be up.

    Deliberately per-test rather than per-session. A session fixture that sets
    ``os.environ`` leaks: the CI step that runs the whole tree in one process
    (``pytest --cov``) would carry offline mode out of tests/unit and into
    tests/verification, where ``test_claim_36_knowledge_store_search`` embeds
    with a real model on purpose. ``monkeypatch`` undoes it after each test, so
    the guard stops exactly at the edge of this directory.

    Real models belong in the integration and verification suites, which are
    allowed to be slow and online.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
