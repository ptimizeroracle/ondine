"""
enrich() — single-function front door for Ondine.

Thin facade over QuickPipeline: give it data, a prompt, and the columns
you want back, get an enriched DataFrame. This is the lowest-friction
entry point for the common case (bulk LLM enrichment of a table).

Design: pure facade — no logic of its own. All behaviour, defaults, and
production plumbing come from QuickPipeline / PipelineBuilder beneath it.
"""

from __future__ import annotations

from decimal import Decimal  # noqa: TC003
from pathlib import Path  # noqa: TC003
from typing import Any  # noqa: TC003

import pandas as pd  # noqa: TC002

from ondine.api.pipeline_builder import PipelineBuilder
from ondine.api.quick import QuickPipeline

# Explicit allowlist: kwargs map 1:1 to QuickPipeline.create / PipelineBuilder
# options. We deliberately do NOT forward unknown kwargs via **kwargs, so a
# typo fails loudly instead of silently being ignored.
_ALLOWED_OPTIONS = frozenset(
    {
        "provider",
        "temperature",
        "max_tokens",
        "batch_size",
        "concurrency",
    }
)


def enrich(
    data: str | Path | pd.DataFrame,
    prompt: str,
    output_columns: list[str] | str | None = None,
    *,
    model: str = "gpt-4o-mini",
    budget: float | Decimal | str | None = None,
    schema: Any | None = None,
    **options: Any,
) -> pd.DataFrame:
    """Run an LLM over a table and return the enriched DataFrame.

    Input columns are read from ``{placeholders}`` in *prompt*; the LLM's
    answers land in *output_columns*. Schema enforcement, checkpointing,
    cost tracking, retries, and adaptive concurrency are on by default.

    Args:
        data: CSV/Excel/Parquet/JSON path or an in-memory DataFrame.
        prompt: Prompt template with ``{column}`` placeholders for each
            input column.
        output_columns: Column(s) to populate. Defaults to ``["output"]``.
        model: LiteLLM model string (e.g. ``"gpt-4o-mini"``,
            ``"claude-3-5-sonnet"``, ``"ollama/qwen3"``). Provider is
            auto-detected from the name.
        budget: Hard USD cap for the run; the pipeline halts at the limit.
        schema: Optional Pydantic model for structured output.
        **options: Forwarded to QuickPipeline — ``provider``,
            ``temperature``, ``max_tokens``, ``batch_size``, ``concurrency``.

    Returns:
        The enriched pandas DataFrame (input columns + output columns).

    Raises:
        TypeError: If an option key is not in the allowlist.
        ValueError: If no placeholders are found or input columns are absent.

    Example:
        >>> from ondine import enrich
        >>> df = enrich(
        ...     "reviews.csv",
        ...     "Classify the tone of: {review}",
        ...     output_columns=["sentiment"],
        ...     model="gpt-4o-mini",
        ...     budget=5.0,
        ... )
    """
    bad = set(options) - _ALLOWED_OPTIONS
    if bad:
        raise TypeError(
            f"enrich() got unexpected option(s): {sorted(bad)}. "
            f"Allowed: {sorted(_ALLOWED_OPTIONS)}."
        )

    # QuickPipeline owns all the smart defaults (provider auto-detect,
    # batch/concurrency sizing, parser selection, retries). We build it
    # first, then layer structured output on top by reconstructing the
    # builder from the resulting specifications — single code path, no fork.
    pipeline = QuickPipeline.create(
        data=data,
        prompt=prompt,
        model=model,
        output_columns=output_columns,
        max_budget=budget,
        **options,
    )

    if schema is not None:
        pipeline = (
            PipelineBuilder.from_specifications(pipeline.specifications)
            .with_structured_output(schema)
            .build()
        )

    result = pipeline.execute()
    return result.to_pandas()
