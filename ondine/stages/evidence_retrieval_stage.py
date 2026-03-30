"""Pipeline stage that retrieves prior evidence for each row before prompt formatting.

Inserted between DataLoaderStage and PromptFormatterStage, this stage
augments every row with a ``_evidence_context`` column containing the top-k
relevant evidence records from the context store (with attribution), and
a ``_evidence_count`` column showing how many records matched.

The PromptFormatterStage auto-injects ``_evidence_context`` into prompts so
the LLM can leverage prior validated answers for consistency.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from ondine.core.data_container import DataContainer
from ondine.core.models import CostEstimate, ValidationResult
from ondine.stages.pipeline_stage import PipelineStage

if TYPE_CHECKING:
    from ondine.context.protocol import ContextStore


class EvidenceRetrievalStage(PipelineStage[DataContainer, DataContainer]):
    """Enrich each row with prior evidence from the context store.

    For every row, builds a query from the configured input columns,
    searches the ``ContextStore``, filters by ``min_score``, and writes
    the formatted evidence (with relevance scores and source refs) into
    ``_evidence_context``.  A parallel ``_evidence_count`` column tracks
    how many results were above threshold — zero means the store had
    nothing useful and evidence priming is effectively a no-op for that row.

    Args:
        store: The ``ContextStore`` instance to search against.
        query_columns: Which input columns to concatenate as the search query.
        top_k: Maximum evidence records to retrieve per row.
        min_score: Minimum relevance score (0-1) to include a result.
            Results below this are discarded to avoid injecting noise.
    """

    def __init__(
        self,
        store: ContextStore,
        query_columns: list[str],
        *,
        top_k: int = 3,
        min_score: float = 0.1,
    ) -> None:
        super().__init__("EvidenceRetrieval")
        self._store = store
        self._query_columns = query_columns
        self._top_k = top_k
        self._min_score = min_score

    def process(self, input_data: DataContainer, context: Any) -> DataContainer:
        """Augment each row with ``_evidence_context`` and ``_evidence_count``."""
        from ondine.adapters.containers import DictListContainer

        rows: list[dict] = []
        total_evidence_found = 0

        for row in input_data:
            row_dict = dict(row)
            query = " ".join(
                str(row_dict.get(c, "")) for c in self._query_columns
            ).strip()

            if query:
                results = self._store.search(query, limit=self._top_k)
                # Filter by min_score to avoid injecting low-relevance noise
                relevant = [r for r in results if r.score >= self._min_score]

                if relevant:
                    # Format with attribution so LLM knows provenance
                    parts = []
                    for r in relevant:
                        source = f" (source: {r.source_ref})" if r.source_ref else ""
                        parts.append(f"[score={r.score:.2f}]{source} {r.text}")
                    row_dict["_evidence_context"] = "\n---\n".join(parts)
                else:
                    row_dict["_evidence_context"] = ""

                row_dict["_evidence_count"] = len(relevant)
                total_evidence_found += len(relevant)
            else:
                row_dict["_evidence_context"] = ""
                row_dict["_evidence_count"] = 0

            rows.append(row_dict)

        columns = list(input_data.columns)
        for col in ("_evidence_context", "_evidence_count"):
            if col not in columns:
                columns.append(col)

        if total_evidence_found == 0:
            self.logger.info(
                "Evidence store returned no relevant results for %d rows "
                "(store may be empty on first run — evidence accumulates across runs)",
                len(rows),
            )
        else:
            self.logger.info(
                "Augmented %d rows with %d total evidence records (top_k=%d, min_score=%.2f)",
                len(rows),
                total_evidence_found,
                self._top_k,
                self._min_score,
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
