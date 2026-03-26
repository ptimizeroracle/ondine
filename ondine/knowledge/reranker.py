"""Reranker implementations and factory.

Satisfies the ``Reranker`` protocol via structural subtyping.

Built-in implementations:
* ``CrossEncoderReranker`` — local cross-encoder via *sentence-transformers*.
* ``JinaReranker`` — API-based via *litellm* rerank endpoint.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ondine.knowledge.protocols import Reranker

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-12-v2"


class CrossEncoderReranker:
    """Rerank search results using a local cross-encoder model.

    The model is loaded lazily on first ``rerank()`` call.
    Falls back to truncation when the model cannot be loaded.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        top_k: int = 5,
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        self._model = None

    def rerank(self, query: str, results: list, top_k: int | None = None) -> list:
        k = top_k if top_k is not None else self._top_k
        if not results:
            return results

        model = self._get_model()
        if model is None:
            return results[:k]

        pairs = [(query, r.text) for r in results]
        scores = model.predict(pairs, show_progress_bar=False)

        scored = sorted(
            zip(scores, results, strict=False), key=lambda x: x[0], reverse=True
        )
        return [replace(r, score=float(s)) for s, r in scored[:k]]

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
            return self._model
        except ImportError:
            logger.info("sentence-transformers not installed; reranking disabled")
            return None
        except Exception:
            logger.warning(
                "Failed to load cross-encoder; reranking disabled", exc_info=True
            )
            return None

    def __repr__(self) -> str:
        return f"CrossEncoderReranker({self._model_name!r})"


class JinaReranker:
    """API-based reranker using *litellm*'s rerank endpoint.

    Works with Jina, Cohere, or any provider litellm supports for
    reranking. Requires a ``JINA_API_KEY`` or equivalent env var.
    """

    def __init__(
        self,
        model: str = "jina-reranker-v2-base-multilingual",
        top_k: int = 5,
        *,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._top_k = top_k
        self._api_key = api_key

    def rerank(self, query: str, results: list, top_k: int | None = None) -> list:
        k = top_k if top_k is not None else self._top_k
        if not results:
            return results

        try:
            import litellm

            documents = [r.text for r in results]
            kwargs: dict = {
                "model": self._model,
                "query": query,
                "documents": documents,
                "top_n": k,
            }
            if self._api_key:
                kwargs["api_key"] = self._api_key

            response = litellm.rerank(**kwargs)

            reranked = []
            for item in response.results:
                idx = item["index"]
                score = item["relevance_score"]
                reranked.append(replace(results[idx], score=float(score)))
            return reranked[:k]
        except Exception:
            logger.warning(
                "Jina rerank failed; returning original order", exc_info=True
            )
            return results[:k]

    def __repr__(self) -> str:
        return f"JinaReranker({self._model!r})"


# ── Factory ───────────────────────────────────────────────────────


def resolve_reranker(spec: Reranker | str | bool | None) -> Reranker | None:
    """Resolve a reranker specification to a concrete instance.

    Accepts:
    * ``None`` or ``False`` → no reranking.
    * ``True`` → default ``CrossEncoderReranker``.
    * A string model name → ``CrossEncoderReranker(model_name)`` for
      local models, ``JinaReranker(model)`` if the string contains
      ``"jina"``.
    * An existing ``Reranker``-compatible object → returned as-is.
    """
    if spec is None or spec is False:
        return None

    if spec is True:
        return CrossEncoderReranker()

    if isinstance(spec, str):
        if "jina" in spec.lower():
            return JinaReranker(model=spec)
        return CrossEncoderReranker(model_name=spec)

    return spec  # already a Reranker-compatible object
