"""Unit-suite guarantees that hold no matter what a test forgets to inject."""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _no_model_downloads():
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

    Real models belong in the integration suite, which is allowed to be slow
    and online. This fixture is scoped to unit tests only.
    """
    previous = {
        name: os.environ.get(name)
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
