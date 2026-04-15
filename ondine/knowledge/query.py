"""Query transformation implementations and factory.

Satisfies the ``QueryTransformer`` protocol via structural subtyping.

Built-in implementations:
* ``MultiQueryTransformer`` — generate N rephrasings to improve recall.
* ``HyDETransformer`` — generate a hypothetical answer for embedding.
* ``StepBackTransformer`` — generate a broader version of the query.

All LLM calls go through *litellm* for provider agnosticity.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ondine.knowledge.protocols import QueryTransformer

logger = logging.getLogger(__name__)


class MultiQueryTransformer:
    """Generate N rephrasings of the original query.

    The retrieval stage unions results across all rephrasings,
    improving recall for ambiguous or under-specified queries.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        n: int = 3,
        *,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._n = n
        self._api_key = api_key

    def transform(self, query: str) -> list[str]:
        try:
            import litellm

            prompt = (
                f"Generate {self._n} different rephrasings of this search query. "
                f"Each rephrasing should approach the topic from a different angle "
                f"to maximize retrieval recall. Return ONLY a JSON array of strings.\n\n"
                f"Query: {query}"
            )
            kwargs: dict = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            }
            if self._api_key:
                kwargs["api_key"] = self._api_key

            response = litellm.completion(**kwargs)
            text = response.choices[0].message.content.strip()

            text = text.removeprefix("```json").removesuffix("```").strip()
            variants = json.loads(text)
            if isinstance(variants, list):
                return [query] + [str(v) for v in variants]
        except Exception:
            logger.warning(
                "Multi-query transform failed; using original", exc_info=True
            )

        return [query]

    def __repr__(self) -> str:
        return f"MultiQueryTransformer(model={self._model!r}, n={self._n})"


class HyDETransformer:
    """Hypothetical Document Embeddings (HyDE).

    Generates a hypothetical answer to the query, then uses that
    answer *as* the retrieval query. The hypothesis is semantically
    closer to relevant passages than the raw question.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        *,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key

    def transform(self, query: str) -> list[str]:
        try:
            import litellm

            prompt = (
                "Write a short, factual paragraph that directly answers the "
                "following question. This will be used for document retrieval, "
                "so be specific and use relevant terminology.\n\n"
                f"Question: {query}"
            )
            kwargs: dict = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
            if self._api_key:
                kwargs["api_key"] = self._api_key

            response = litellm.completion(**kwargs)
            hypothesis = response.choices[0].message.content.strip()
            return [query, hypothesis]
        except Exception:
            logger.warning("HyDE transform failed; using original", exc_info=True)

        return [query]

    def __repr__(self) -> str:
        return f"HyDETransformer(model={self._model!r})"


class StepBackTransformer:
    """Step-back prompting for query generalization.

    Generates a more general/abstract version of the query so
    retrieval can surface broader context that the specific query
    might miss.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        *,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key

    def transform(self, query: str) -> list[str]:
        try:
            import litellm

            prompt = (
                "Given the following specific question, generate a more general, "
                "abstract 'step-back' question that covers the broader topic. "
                "Return ONLY the step-back question, nothing else.\n\n"
                f"Specific question: {query}"
            )
            kwargs: dict = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
            if self._api_key:
                kwargs["api_key"] = self._api_key

            response = litellm.completion(**kwargs)
            step_back = response.choices[0].message.content.strip()
            return [query, step_back]
        except Exception:
            logger.warning("Step-back transform failed; using original", exc_info=True)

        return [query]

    def __repr__(self) -> str:
        return f"StepBackTransformer(model={self._model!r})"


# ── Factory ───────────────────────────────────────────────────────

_STRATEGY_MAP: dict[str, type] = {
    "multi-query": MultiQueryTransformer,
    "multi_query": MultiQueryTransformer,
    "hyde": HyDETransformer,
    "step-back": StepBackTransformer,
    "step_back": StepBackTransformer,
    "stepback": StepBackTransformer,
}


def resolve_query_transform(
    spec: QueryTransformer | str | None,
) -> QueryTransformer | None:
    """Resolve a query-transform specification to a concrete instance.

    Accepts:
    * ``None`` → no transformation.
    * A strategy name (``"multi-query"``, ``"hyde"``, ``"step-back"``)
      → default-configured instance.
    * An existing ``QueryTransformer``-compatible object → returned as-is.
    """
    if spec is None:
        return None

    if isinstance(spec, str):
        cls = _STRATEGY_MAP.get(spec.lower().strip())
        if cls is None:
            raise ValueError(
                f"Unknown query-transform strategy {spec!r}. "
                f"Available: {', '.join(sorted(_STRATEGY_MAP))}"
            )
        return cls()  # type: ignore[no-any-return]

    return spec  # type: ignore[no-any-return]  # already a QueryTransformer-compatible object
