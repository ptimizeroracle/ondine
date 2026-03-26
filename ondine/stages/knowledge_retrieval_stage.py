"""Pipeline stage that retrieves KB context for each row before prompt formatting.

Inserted between DataLoaderStage and PromptFormatterStage, this stage
augments every row with a ``_kb_context`` column containing the top-k
relevant chunks from the knowledge base. The PromptFormatterStage then
auto-injects this context into the prompt template.

Optionally runs LLM-as-judge evaluation after the LLM produces an
answer, adding ``_kb_eval_*`` columns to the output.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from ondine.core.data_container import DataContainer
from ondine.core.models import CostEstimate, ValidationResult
from ondine.stages.pipeline_stage import PipelineStage

if TYPE_CHECKING:
    from ondine.knowledge.store import KnowledgeStore


class KnowledgeRetrievalStage(PipelineStage[DataContainer, DataContainer]):
    """Enrich each row with top-k knowledge base chunks.

    For every row, builds a query from the configured input columns,
    runs hybrid search against the ``KnowledgeStore`` (which now
    handles query transformation and reranking internally via its
    protocol components), and writes the concatenated text into a
    ``_kb_context`` column.

    Args:
        store: The ingested ``KnowledgeStore`` to search against.
        query_columns: Which input columns to concatenate as the search query.
        top_k: Number of chunks to retrieve per row.
        reranker: Optional legacy reranker (now prefer configuring the
            reranker directly on the ``KnowledgeStore``).
        context_separator: String used to join multiple chunk texts.
        evaluate: Run LLM-as-judge scoring on RAG answers.
        eval_model: LLM model for the judge.
    """

    def __init__(
        self,
        store: KnowledgeStore,
        query_columns: list[str],
        *,
        top_k: int = 3,
        reranker: Any | None = None,
        context_separator: str = "\n\n---\n\n",
        evaluate: bool = False,
        eval_model: str = "openai/gpt-4o-mini",
    ) -> None:
        super().__init__("KnowledgeRetrieval")
        self._store = store
        self._query_columns = query_columns
        self._top_k = top_k
        self._reranker = reranker
        self._separator = context_separator
        self._evaluate = evaluate
        self._eval_model = eval_model

    def process(self, input_data: DataContainer, context: Any) -> DataContainer:
        """Augment each row with ``_kb_context``."""
        from ondine.adapters.containers import DictListContainer

        rows: list[dict] = []
        for row in input_data:
            row_dict = dict(row)
            query = " ".join(
                str(row_dict.get(c, "")) for c in self._query_columns
            ).strip()

            if query:
                results = self._store.search(query, limit=self._top_k * 2)
                if self._reranker and results:
                    results = self._reranker.rerank(query, results)
                else:
                    results = results[: self._top_k]

                row_dict["_kb_context"] = self._separator.join(r.text for r in results)
            else:
                row_dict["_kb_context"] = ""

            rows.append(row_dict)

        columns = list(input_data.columns)
        if "_kb_context" not in columns:
            columns.append("_kb_context")

        self.logger.info(
            "Augmented %d rows with KB context (top_k=%d)", len(rows), self._top_k
        )
        return DictListContainer(data=rows, columns=columns)

    def evaluate_answers(
        self, results_data: DataContainer, query_column: str, answer_column: str
    ) -> DataContainer:
        """Post-LLM evaluation: score answers against their retrieved contexts.

        Adds ``_kb_eval_faithfulness``, ``_kb_eval_relevancy``, and
        ``_kb_eval_context_precision`` columns.
        """
        if not self._evaluate:
            return results_data

        from ondine.adapters.containers import DictListContainer
        from ondine.knowledge.eval import LLMJudge

        judge = LLMJudge(model=self._eval_model)
        rows: list[dict] = []

        for row in results_data:
            row_dict = dict(row)
            query = str(row_dict.get(query_column, ""))
            answer = str(row_dict.get(answer_column, ""))
            context_text = str(row_dict.get("_kb_context", ""))
            contexts = [c.strip() for c in context_text.split("---") if c.strip()]

            if query and answer and contexts:
                result = judge.score(query, answer, contexts)
                row_dict["_kb_eval_faithfulness"] = result.faithfulness
                row_dict["_kb_eval_relevancy"] = result.relevancy
                row_dict["_kb_eval_context_precision"] = result.context_precision
            else:
                row_dict["_kb_eval_faithfulness"] = None
                row_dict["_kb_eval_relevancy"] = None
                row_dict["_kb_eval_context_precision"] = None

            rows.append(row_dict)

        columns = list(results_data.columns)
        for col in (
            "_kb_eval_faithfulness",
            "_kb_eval_relevancy",
            "_kb_eval_context_precision",
        ):
            if col not in columns:
                columns.append(col)

        self.logger.info(
            "Evaluated %d rows with LLM judge (%s)", len(rows), self._eval_model
        )
        return DictListContainer(data=rows, columns=columns)

    def validate_input(self, input_data: DataContainer) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        if len(input_data) == 0:
            result.add_error("Input container is empty")
        missing = set(self._query_columns) - set(input_data.columns)
        if missing:
            result.add_error(f"Query columns not in container: {missing}")
        return result

    def estimate_cost(self, input_data: DataContainer) -> CostEstimate:
        return CostEstimate(
            total_cost=Decimal("0"),
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            rows=len(input_data),
        )
