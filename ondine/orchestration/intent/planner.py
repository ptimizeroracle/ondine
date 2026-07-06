"""LLM-drafted pipeline planning (intent layer).

This module exposes :func:`plan` — a front-door that uses a single
structured LLM call to turn a goal + a sample of tabular data into a
fully-formed :class:`~ondine.core.specifications.PipelineSpecifications`.

Design notes
------------
This is a *deep module* (Ousterhout, *A Philosophy of Software Design*):

* The interface is one function with four obvious arguments.
* The implementation hides: sample extraction, the instructor schema,
  mapping the LLM's draft onto the validated spec tree, defensive
  column checks, and the YAML preview used for approval-by-inspection.

The LLM client is the only non-deterministic collaborator, so it is an
injectable boundary (``llm_client``). Callers normally leave it at
``None`` and let :func:`plan` build the real
:class:`~ondine.adapters.llm_client.UnifiedLiteLLMClient`; tests inject
a fake. There is no agent loop, no re-planning, and no execution: a
:class:`Plan` is an immutable handoff that the user inspects and then
feeds to :meth:`Plan.build` (which routes through the existing
:class:`~ondine.api.pipeline_builder.PipelineBuilder`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ondine.core.specifications import (
    DatasetSpec,
    DataSourceType,
    LLMSpec,
    PipelineSpecifications,
    ProcessingSpec,
    PromptSpec,
)

if TYPE_CHECKING:
    import pandas as pd

    from ondine.adapters.llm_client import LLMClient
    from ondine.api.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum rows sampled from the input data and shown to the drafting LLM.
#: Bounded so a million-row frame never inflates the planning prompt.
_SAMPLE_ROWS = 50

#: Default model used when the caller does not pass one. Mirrors QuickPipeline.
_DEFAULT_MODEL = "openai/gpt-4o-mini"


# ---------------------------------------------------------------------------
# LLM output schema (instructor-shaped)
# ---------------------------------------------------------------------------


class _DraftSpec(BaseModel):
    """Structured payload the drafting LLM must return.

    Field names are deliberate and few: every field maps 1:1 to a knob on
    :class:`PipelineSpecifications` that a non-expert cannot be expected to
    know how to set. Anything the user *can* reasonably set themselves
    (budget, model, rate limits, concurrency) is intentionally absent —
    those are the caller's constraints, not the LLM's guess.
    """

    input_columns: list[str] = Field(
        ..., min_length=1, description="Existing input columns to feed the prompt."
    )
    output_columns: list[str] = Field(
        ..., min_length=1, description="New columns the LLM should produce."
    )
    prompt_template: str = Field(
        ..., min_length=1, description="Prompt template using {column} placeholders."
    )
    system_message: str | None = Field(
        default=None, description="Optional system message."
    )
    response_format: str = Field(
        default="raw", description="'raw' (single column) or 'json' (multi-column)."
    )
    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Sampling temperature."
    )
    rationale: str = Field(
        default="", description="One-sentence rationale for these choices."
    )


# ---------------------------------------------------------------------------
# Public value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """An LLM-drafted pipeline awaiting human approval.

    A :class:`Plan` is an immutable handoff: it carries the drafted
    :class:`PipelineSpecifications` plus the provenance (goal + rationale)
    so a reviewer can sanity-check *why* each choice was made. Call
    :meth:`preview_yaml` to inspect the configuration, edit it, or pass it
    through :class:`~ondine.config.config_loader.ConfigLoader`; call
    :meth:`build` once you are happy.

    The planner never executes anything — building a real
    :class:`~ondine.api.pipeline.Pipeline` is a one-liner the caller owns.
    """

    specifications: PipelineSpecifications
    goal: str
    rationale: str = ""

    def preview_yaml(self) -> str:
        """Return the drafted spec as YAML for approval-by-inspection.

        The output round-trips through
        :class:`~ondine.config.config_loader.ConfigLoader.from_yaml`, so a
        reviewer can edit it on disk and reload without ever touching
        Python.
        """
        import yaml  # local import: yaml is a hard runtime dep but we keep
        # the module importable for environments that only need plan().

        payload = self.specifications.model_dump(mode="json")
        return yaml.dump(payload, default_flow_style=False, sort_keys=False)

    def build(self) -> Pipeline:
        """Materialise the drafted spec into a ready-to-run Pipeline.

        Routes through the standard
        :class:`~ondine.api.pipeline_builder.PipelineBuilder` so the planner
        introduces no new execution path.
        """
        from ondine.api.pipeline_builder import PipelineBuilder

        return PipelineBuilder.from_specifications(self.specifications).build()


# ---------------------------------------------------------------------------
# Sample extraction
# ---------------------------------------------------------------------------


def _to_pandas(data: Any) -> pd.DataFrame:
    """Coerce CSV/Excel/Parquet paths or frames to a pandas DataFrame.

    Mirrors :meth:`QuickPipeline._load_data` but lives here so the planner
    stays a single self-contained entry point.
    """
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        return data
    # Polars frame — cheap to convert.
    if hasattr(data, "to_pandas") and not isinstance(data, str):
        return data.to_pandas()

    if not isinstance(data, str | Path):
        raise ValueError(
            f"Unsupported data type {type(data).__name__}; "
            "expected DataFrame or a file path."
        )

    path = Path(data)  # type: ignore[arg-type]
    if not path.exists():
        raise ValueError(f"Data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(
        f"Unsupported file type: {suffix}. "
        "Supported: .csv, .xlsx, .xls, .parquet, .json"
    )


def _sample_records(df: pd.DataFrame, n: int = _SAMPLE_ROWS) -> list[dict[str, Any]]:
    """Return up to ``n`` rows from ``df`` as JSON-friendly records.

    Truncates long string cells so the drafting prompt stays compact even
    when the data contains essays or base64 blobs.
    """
    head = df.head(n)
    records: list[dict[str, Any]] = []
    for _, row in head.iterrows():
        record: dict[str, Any] = {}
        for col, value in row.items():
            if isinstance(value, str):
                record[str(col)] = value[:200]
            else:
                record[str(col)] = value
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Draft -> validated spec
# ---------------------------------------------------------------------------


def _draft_to_spec(
    draft: _DraftSpec,
    df: pd.DataFrame,
    *,
    model: str,
    budget: Decimal,
) -> PipelineSpecifications:
    """Validate the LLM draft against the real data and build a spec.

    This is where defensive checks live: a hallucinated column name or an
    in/out overlap is a programmer-visible error here, not a cryptic
    Pydantic failure at execution time.
    """
    data_cols = set(df.columns.astype(str))

    missing = [c for c in draft.input_columns if c not in data_cols]
    if missing:
        raise ValueError(
            f"LLM drafted input_columns {missing} not present in data. "
            f"Available columns: {sorted(data_cols)}"
        )

    overlap = set(draft.input_columns) & set(draft.output_columns)
    if overlap:
        raise ValueError(
            f"input_columns and output_columns overlap: {sorted(overlap)}. "
            "output_columns must be new columns."
        )

    # response_format sanity: multi-column outputs need JSON parsing.
    fmt = draft.response_format.lower()
    if fmt not in {"raw", "json"}:
        raise ValueError(
            f"LLM drafted unsupported response_format {fmt!r}; "
            "expected 'raw' or 'json'."
        )
    if fmt == "raw" and len(draft.output_columns) > 1:
        # Auto-promote to JSON so multi-column output is parseable.
        fmt = "json"

    prompt_spec = PromptSpec(
        template=draft.prompt_template,
        system_message=draft.system_message,
        response_format=fmt,
        json_fields=list(draft.output_columns) if fmt == "json" else None,
    )

    dataset_spec = DatasetSpec(
        source_type=DataSourceType.DATAFRAME,
        input_columns=list(draft.input_columns),
        output_columns=list(draft.output_columns),
    )

    llm_spec = LLMSpec(
        model=model,
        temperature=draft.temperature,
    )

    processing_spec = ProcessingSpec(
        max_budget=budget,
    )

    return PipelineSpecifications(
        dataset=dataset_spec,
        prompt=prompt_spec,
        llm=llm_spec,
        processing=processing_spec,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


_PLANNER_SYSTEM = (
    "You are an expert data-engineering assistant. Given a small sample of "
    "tabular data and a user's goal, you draft the configuration for an LLM "
    "batch-processing pipeline. You choose which existing columns to feed "
    "into the prompt, which new columns the model should produce, the prompt "
    "template itself, and the response format. You never invent column "
    "names that are not in the provided sample. The prompt template MUST "
    "reference each input column with {column_name} placeholders."
)


def _build_user_prompt(
    goal: str, columns: list[str], sample: list[dict[str, Any]]
) -> str:
    """Assemble the user-side drafting prompt.

    The prompt is the entire contract with the drafting LLM, so it is
    built explicitly (no f-string soup hidden in three helpers): goal,
    schema of the available data, and a few concrete rows.
    """
    sample_json = json.dumps(sample, default=str, indent=2)
    return (
        f"Goal:\n{goal}\n\n"
        f"Available columns:\n{json.dumps(columns)}\n\n"
        f"Sample rows (truncated, up to {len(sample)}):\n{sample_json}\n\n"
        "Draft a pipeline configuration that achieves the goal using only "
        "the available columns. Return the draft in the required structured "
        "shape."
    )


# ---------------------------------------------------------------------------
# LLM client construction
# ---------------------------------------------------------------------------


def _default_client(model: str) -> LLMClient:
    """Build the real LLM client used for drafting when none is injected."""
    from ondine.adapters.llm_client import create_llm_client

    spec = LLMSpec(model=model, temperature=0.0)
    return create_llm_client(spec)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plan(
    data: Any,
    goal: str,
    *,
    budget: Decimal | float | str,
    model: str = _DEFAULT_MODEL,
    llm_client: LLMClient | None = None,
) -> Plan:
    """Draft a :class:`PipelineSpecifications` from data + a natural-language goal.

    This is the intent-layer front door (ARCHITECTURE_PROPOSAL §5). It makes
    exactly one structured LLM call to decide prompt template, input/output
    columns, and response format, then returns a :class:`Plan` carrying a
    fully-validated spec. Nothing is executed.

    Args:
        data: Input data — a pandas/polars DataFrame or a path to a
            CSV/Excel/Parquet/JSON file. A small sample (≤ 50 rows) is
            shown to the drafting LLM.
        goal: What you want the pipeline to do, in plain language.
        budget: Maximum spend in USD. Mandatory — there is no sensible
            default for "how much money should this cost".
        model: LiteLLM-format model string used both to draft the spec and
            as the model the drafted pipeline will run on.
        llm_client: Optional :class:`~ondine.adapters.llm_client.LLMClient`
            used for the single drafting call. Leave ``None`` to build the
            real client; inject a fake in tests.

    Returns:
        A :class:`Plan` holding the drafted spec. Call
        :meth:`Plan.preview_yaml` to inspect it, then :meth:`Plan.build`
        to get a runnable :class:`~ondine.api.pipeline.Pipeline`.

    Raises:
        ValueError: If ``goal`` is empty, ``budget`` is non-positive, the
            data cannot be loaded, or the LLM drafts columns that are not
            in the data / overlap each other.

    Example:
        >>> from decimal import Decimal
        >>> from ondine import plan
        >>> drafted = plan(
        ...     data="reviews.csv",
        ...     goal="Extract sentiment and a one-line summary per review",
        ...     budget=Decimal("5.0"),
        ... )
        >>> print(drafted.preview_yaml())      # inspect / edit
        >>> pipeline = drafted.build()         # then run
    """
    # --- Validate the caller's hard constraints before spending a token. ---
    cleaned_goal = (goal or "").strip()
    if not cleaned_goal:
        raise ValueError("goal must be a non-empty description of the task.")

    budget_dec = Decimal(str(budget))
    if budget_dec <= 0:
        raise ValueError(f"budget must be positive, got {budget}.")

    # --- Coerce input data and pull a bounded sample. ---
    df = _to_pandas(data)
    if df.empty:
        raise ValueError("Cannot plan against an empty dataset.")

    columns = [str(c) for c in df.columns]
    sample = _sample_records(df)

    # --- Make the single structured drafting call. ---
    client = llm_client or _default_client(model)
    response = client.structured_invoke(
        prompt=_build_user_prompt(cleaned_goal, columns, sample),
        output_cls=_DraftSpec,
        system_message=_PLANNER_SYSTEM,
    )

    draft = response.structured_result
    if draft is None:  # defensive: clients must populate structured_result
        raise ValueError(
            "LLM client did not return a structured_result; "
            "ensure structured_invoke populates the parsed Pydantic object."
        )

    # --- Validate the draft against the real data and assemble the spec. ---
    specifications = _draft_to_spec(draft, df, model=model, budget=budget_dec)

    return Plan(
        specifications=specifications,
        goal=cleaned_goal,
        rationale=draft.rationale,
    )
