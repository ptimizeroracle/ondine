"""Front-door facade: enrich() — the one-call path to LLM dataset enrichment.

enrich() is a thin wrapper over QuickPipeline: it accepts the smallest common
set of arguments, builds a pipeline, runs it once, and returns the enriched
DataFrame. Anything beyond the named parameters is forwarded through an
explicit allowlist so callers get a clean error rather than silent getattr
magic. This is L5 — it owns no logic, only orchestration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ondine.api.quick import QuickPipeline

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from decimal import Decimal
    from pathlib import Path

    import pandas as pd

# **options keys that map 1:1 to QuickPipeline.create parameters.
# Anything not listed here is rejected so the surface stays auditable.
_ALLOWED_OPTIONS: frozenset[str] = frozenset(
    {
        "temperature",
        "max_tokens",
        "batch_size",
        "concurrency",
        "provider",
    }
)


def enrich(
    data: str | Path | pd.DataFrame,
    prompt: str,
    output_columns: list[str] | str | None = None,
    *,
    model: str = "gpt-4o-mini",
    schema: Any | None = None,
    budget: float | Decimal | None = None,  # noqa: F821 — Decimal imported lazily
    **options: Any,
) -> pd.DataFrame:
    """Enrich a dataset with a single LLM call per row.

    Args:
        data: CSV/Excel/Parquet/JSON file path or an in-memory DataFrame.
        prompt: Prompt template with {placeholders} matching input columns.
        output_columns: Output column name(s). Defaults to ``["output"]``.
        model: Model identifier (default ``gpt-4o-mini``); provider auto-detected.
        schema: Optional Pydantic model enabling native structured output.
        budget: Optional maximum cost cap in USD.
        **options: Recognized tuning knobs — ``temperature``, ``max_tokens``,
            ``batch_size``, ``concurrency``, ``provider``. Unknown keys raise
            TypeError.

    Returns:
        The enriched DataFrame (input columns plus output columns).

    Raises:
        TypeError: If an unrecognized keyword argument is passed.
        ValueError: If the data, prompt, or budget is invalid (raised by
            the underlying pipeline).
    """

    # Reject anything outside the allowlist (including reserved internal names
    # like max_budget) before forwarding, so the contract is explicit.
    unexpected = set(options) - _ALLOWED_OPTIONS
    if unexpected:
        raise TypeError(
            f"enrich() got unexpected keyword argument(s): {sorted(unexpected)}. "
            f"Allowed: {sorted(_ALLOWED_OPTIONS)}."
        )

    pipeline = QuickPipeline.create(
        data=data,
        prompt=prompt,
        output_columns=output_columns,
        model=model,
        max_budget=budget,
        **options,
    )

    # Schema is the one knob QuickPipeline.create doesn't expose; inject it via
    # the same metadata channel that PipelineBuilder.with_structured_output uses
    # (read by Pipeline.execute at the LLM-invocation stage).
    if schema is not None:
        pipeline.specifications.metadata["structured_output_model"] = schema

    result = pipeline.execute()
    return result.to_pandas()
