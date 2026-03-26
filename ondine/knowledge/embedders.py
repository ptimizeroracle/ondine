"""Concrete ``Embedder`` implementations and factory.

Satisfies the ``Embedder`` protocol via structural subtyping — no
inheritance from the protocol class is needed.

Two built-in implementations:
* ``SentenceTransformerEmbedder`` — local, open-source, GPU-optional.
* ``OpenAIEmbedder`` — API-based via *litellm* (works for OpenAI,
  Cohere, Azure, and any litellm-supported provider).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ondine.knowledge.protocols import Embedder

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


class SentenceTransformerEmbedder:
    """Wrap any ``sentence-transformers`` model as an ``Embedder``.

    The model is loaded lazily on first ``embed()`` call so that
    import-time stays fast and GPU memory is only allocated when needed.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def _load(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)
        logger.info("Loaded embedding model: %s", self._model_name)
        return self._model

    def __repr__(self) -> str:
        return f"SentenceTransformerEmbedder({self._model_name!r})"


class OpenAIEmbedder:
    """API-based embedder using *litellm* for provider-agnostic access.

    Works with any model litellm supports for embeddings:
    ``text-embedding-3-small``, ``text-embedding-ada-002``,
    ``cohere/embed-english-v3.0``, etc.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        import litellm

        kwargs: dict = {"model": self._model, "input": texts}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions

        response = litellm.embedding(**kwargs)
        return [item["embedding"] for item in response.data]

    def __repr__(self) -> str:
        return f"OpenAIEmbedder({self._model!r})"


# ── Factory ───────────────────────────────────────────────────────

_SHORTCUT_MAP: dict[str, type] = {
    "bge": SentenceTransformerEmbedder,
    "minilm": SentenceTransformerEmbedder,
    "openai": OpenAIEmbedder,
}


def resolve_embedder(spec: Embedder | str | None) -> Embedder | None:
    """Resolve an embedder specification to a concrete instance.

    Accepts:
    * ``None`` → auto-detect (``SentenceTransformerEmbedder`` if
      ``sentence-transformers`` is installed, else ``None``).
    * A string model name → ``SentenceTransformerEmbedder(model_name)``
      for local models, ``OpenAIEmbedder(model)`` if the string
      contains ``"openai/"`` or ``"text-embedding"``.
    * An existing ``Embedder``-compatible object → returned as-is.
    """
    if spec is None:
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            logger.info("sentence-transformers unavailable; embeddings disabled")
            return None

    if isinstance(spec, str):
        if "text-embedding" in spec or spec.startswith("openai/"):
            return OpenAIEmbedder(model=spec.removeprefix("openai/"))
        return SentenceTransformerEmbedder(model_name=spec)

    return spec  # already an Embedder-compatible object
