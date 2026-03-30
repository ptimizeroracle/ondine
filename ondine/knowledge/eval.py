"""RAG evaluation via LLM-as-judge and factory.

Satisfies the ``RetrievalScorer`` protocol via structural subtyping.

Built-in implementation:
* ``LLMJudge`` — uses *litellm* to score faithfulness, relevancy,
  and context precision of a RAG answer.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from ondine.knowledge.protocols import EvalResult

if TYPE_CHECKING:
    from ondine.knowledge.protocols import RetrievalScorer

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
You are an expert RAG evaluator. Score the following answer on three dimensions.

**Query**: {query}

**Retrieved Contexts**:
{contexts}

**Answer**: {answer}

Score each dimension from 0.0 to 1.0:
1. **faithfulness**: Is the answer fully supported by the retrieved contexts? \
(1.0 = every claim is grounded, 0.0 = fabricated)
2. **relevancy**: Does the answer directly address the query? \
(1.0 = perfectly on-topic, 0.0 = irrelevant)
3. **context_precision**: Are the retrieved contexts relevant to the query? \
(1.0 = all contexts are useful, 0.0 = none are useful)

Return ONLY a JSON object: {{"faithfulness": float, "relevancy": float, "context_precision": float}}
"""


class LLMJudge:
    """LLM-as-judge for RAG evaluation.

    Uses *litellm* for provider-agnostic access. Any chat model works:
    ``openai/gpt-4o-mini``, ``anthropic/claude-3-haiku``, etc.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._temperature = temperature

    def score(self, query: str, answer: str, contexts: list[str]) -> EvalResult:
        try:
            import litellm

            contexts_str = "\n---\n".join(
                f"[{i + 1}] {c}" for i, c in enumerate(contexts)
            )
            prompt = _JUDGE_PROMPT.format(
                query=query, answer=answer, contexts=contexts_str
            )

            kwargs: dict = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self._temperature,
            }
            if self._api_key:
                kwargs["api_key"] = self._api_key

            response = litellm.completion(**kwargs)
            text = response.choices[0].message.content.strip()

            text = text.removeprefix("```json").removesuffix("```").strip()
            data = json.loads(text)

            return EvalResult(
                faithfulness=float(data.get("faithfulness", 0.0)),
                relevancy=float(data.get("relevancy", 0.0)),
                context_precision=float(data.get("context_precision", 0.0)),
                metadata={"model": self._model, "raw": data},
            )
        except Exception:
            logger.warning("LLM judge scoring failed", exc_info=True)
            return EvalResult(metadata={"error": "scoring_failed"})

    def __repr__(self) -> str:
        return f"LLMJudge(model={self._model!r})"


# ── Factory ───────────────────────────────────────────────────────


def resolve_scorer(
    spec: RetrievalScorer | str | bool | None,
) -> RetrievalScorer | None:
    """Resolve a scorer specification to a concrete instance.

    Accepts:
    * ``None`` or ``False`` → no evaluation.
    * ``True`` → default ``LLMJudge()``.
    * A string model name → ``LLMJudge(model=spec)``.
    * An existing ``RetrievalScorer``-compatible object → returned as-is.
    """
    if spec is None or spec is False:
        return None

    if spec is True:
        return LLMJudge()

    if isinstance(spec, str):
        return LLMJudge(model=spec)

    return spec  # already a RetrievalScorer-compatible object
