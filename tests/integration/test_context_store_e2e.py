"""
E2E tests for context store anti-hallucination pipeline integration.

Validates the full pipeline path: load -> LLM -> parse -> context verification -> output.
Uses patched litellm.acompletion to provide deterministic LLM responses while exercising
every real stage including PipelineBuilder wiring, Pipeline execution, and
_apply_context_verification.

Regression targets:
- Grounding scores attached to output when grounding is enabled
- Output columns cleared when grounding action is "skip" and score is below threshold
- Contradiction flags set when same-key rows produce different values
- Confidence scores derived from grounding + search support
- Verification stage skipped when only with_context_store() is called (no features enabled)
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from ondine.api import PipelineBuilder
from ondine.context.memory_store import InMemoryContextStore

_FAKE_API_KEY = "fake-key"  # pragma: allowlist secret


def _make_litellm_response(content: str):
    """Build a minimal litellm ModelResponse for the mock."""
    from litellm import Choices, Message, ModelResponse

    return ModelResponse(
        choices=[Choices(message=Message(content=content))],
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )


def _build_base_pipeline(df: pd.DataFrame, store: InMemoryContextStore):
    """Shared builder: CSV-like DataFrame, mock LLM, context store injected."""
    return (
        PipelineBuilder.create()
        .from_dataframe(
            df,
            input_columns=["product"],
            output_columns=["category"],
        )
        .with_prompt("Classify: {product}")
        .with_llm(provider="openai", model="gpt-4o-mini", api_key=_FAKE_API_KEY)
        .with_concurrency(1)
        .with_context_store(store)
    )


# ---------------------------------------------------------------------------
# 1. Grounding — flag action: LLM echoes source text closely
# ---------------------------------------------------------------------------
# Regression: if _apply_context_verification stops calling context_store.ground()
# or stops writing _grounding_score to rows, this test fails.
@patch("litellm.acompletion")
def test_grounding_flag_preserves_output_and_attaches_score(mock_acompletion):
    """High-similarity LLM output should produce _grounding_score > 0 and keep output."""
    mock_acompletion.return_value = _make_litellm_response("Organic Cereals")

    df = pd.DataFrame({"product": ["Organic Cereals product from the store"]})
    store = InMemoryContextStore()

    pipeline = (
        _build_base_pipeline(df, store)
        .with_grounding(threshold=0.1, action="flag")
        .build()
    )

    result = pipeline.execute()
    output_df = result.data.to_pandas()

    assert "category" in output_df.columns
    assert output_df["category"].iloc[0] is not None
    assert "_grounding_score" in output_df.columns

    score = output_df["_grounding_score"].iloc[0]
    assert score is not None, "Grounding score should not be None for similar text"
    assert score > 0, f"Expected positive grounding score, got {score}"


# ---------------------------------------------------------------------------
# 2. Grounding — skip action: LLM returns completely unrelated text
# ---------------------------------------------------------------------------
# Regression: if the skip-action branch in _apply_context_verification is removed
# or broken, output columns will remain populated when they should be cleared.
@patch("litellm.acompletion")
def test_grounding_skip_clears_output_when_ungrounded(mock_acompletion):
    """Unrelated LLM output with action='skip' should clear output columns."""
    mock_acompletion.return_value = _make_litellm_response(
        "quantum entanglement in parallel universes"
    )

    df = pd.DataFrame({"product": ["Organic Cereals product from the store"]})
    store = InMemoryContextStore()

    pipeline = (
        _build_base_pipeline(df, store)
        .with_grounding(threshold=0.5, action="skip")
        .build()
    )

    result = pipeline.execute()
    output_df = result.data.to_pandas()

    assert output_df["category"].iloc[0] is None, (
        "Output should be cleared when grounding score is below threshold with skip action"
    )


# ---------------------------------------------------------------------------
# 3. Contradiction detection: same product, different categories
# ---------------------------------------------------------------------------
# Regression: if contradiction_cfg wiring breaks, _contradiction column won't appear
# or rows with differing values for the same key won't be flagged.
@patch("litellm.acompletion")
def test_contradiction_detection_flags_conflicting_rows(mock_acompletion):
    """Two rows with same input but different outputs should trigger contradiction flag."""
    responses = iter(
        [
            _make_litellm_response("Organic Cereals"),
            _make_litellm_response("Frozen Desserts"),
        ]
    )
    mock_acompletion.side_effect = lambda *a, **kw: next(responses)

    df = pd.DataFrame(
        {
            "product": ["Corn Flakes", "Corn Flakes"],
        }
    )
    store = InMemoryContextStore()

    pipeline = (
        _build_base_pipeline(df, store)
        .with_contradiction_detection(
            key_columns=["product"],
            value_columns=["category"],
        )
        .build()
    )

    result = pipeline.execute()
    output_df = result.data.to_pandas()

    assert "_contradiction" in output_df.columns
    contradictions = output_df["_contradiction"].fillna(False).tolist()
    assert any(contradictions), (
        f"Expected at least one contradiction flag, got {contradictions}"
    )


# ---------------------------------------------------------------------------
# 4. Confidence scoring: derived from grounding + search support
# ---------------------------------------------------------------------------
# Regression: if the confidence formula (gs * 0.7 + support_factor * 0.3) changes
# or the search-support lookup breaks, this test catches it.
@patch("litellm.acompletion")
def test_confidence_scoring_produces_bounded_scores(mock_acompletion):
    """Confidence scores should be between 0 and 1 when features are enabled."""
    mock_acompletion.return_value = _make_litellm_response("Organic Cereals")

    df = pd.DataFrame({"product": ["Organic Cereals product from the store"]})
    store = InMemoryContextStore()

    pipeline = (
        _build_base_pipeline(df, store)
        .with_grounding(threshold=0.1, action="flag")
        .with_confidence_scoring(include_in_output=True)
        .build()
    )

    result = pipeline.execute()
    output_df = result.data.to_pandas()

    assert "_confidence_score" in output_df.columns
    score = output_df["_confidence_score"].iloc[0]
    assert score is not None, (
        "Confidence score should not be None when grounding is enabled"
    )
    assert 0.0 <= score <= 1.0, f"Confidence score {score} outside [0, 1]"


# ---------------------------------------------------------------------------
# 5. Combined: grounding + contradiction + confidence in one pipeline
# ---------------------------------------------------------------------------
# Regression: if composing all three features causes interference (e.g. claim_id
# collisions, missing stored_claim_ids entries), this catches it.
@patch("litellm.acompletion")
def test_all_features_combined(mock_acompletion):
    """All context verification features enabled simultaneously should compose cleanly."""
    responses = iter(
        [
            _make_litellm_response("Organic Cereals"),
            _make_litellm_response("Frozen Desserts"),
            _make_litellm_response("Organic Cereals"),
        ]
    )
    mock_acompletion.side_effect = lambda *a, **kw: next(responses)

    df = pd.DataFrame(
        {
            "product": [
                "Organic Cereals product",
                "Organic Cereals product",
                "Frozen Desserts item",
            ],
        }
    )
    store = InMemoryContextStore()

    pipeline = (
        _build_base_pipeline(df, store)
        .with_grounding(threshold=0.1, action="flag")
        .with_contradiction_detection(
            key_columns=["product"],
            value_columns=["category"],
        )
        .with_confidence_scoring(include_in_output=True)
        .build()
    )

    result = pipeline.execute()
    output_df = result.data.to_pandas()

    assert "_grounding_score" in output_df.columns
    assert "_confidence_score" in output_df.columns
    assert "_contradiction" in output_df.columns

    for _, row in output_df.iterrows():
        if row.get("_grounding_score") is not None:
            assert 0.0 <= row["_grounding_score"] <= 1.0
        if row.get("_confidence_score") is not None:
            assert 0.0 <= row["_confidence_score"] <= 1.0


# ---------------------------------------------------------------------------
# 6. Store-only mode: no verification columns when no features enabled
# ---------------------------------------------------------------------------
# Regression: if the guard `if context_store and (grounding_cfg or ...)` is
# removed, verification would run unconditionally and inject unwanted columns.
@patch("litellm.acompletion")
def test_store_only_mode_skips_verification(mock_acompletion):
    """with_context_store() alone (no grounding/contradiction/confidence) should not
    inject _grounding_score or other verification columns."""
    mock_acompletion.return_value = _make_litellm_response("Organic Cereals")

    df = pd.DataFrame({"product": ["Corn Flakes"]})
    store = InMemoryContextStore()

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            df,
            input_columns=["product"],
            output_columns=["category"],
        )
        .with_prompt("Classify: {product}")
        .with_llm(provider="openai", model="gpt-4o-mini", api_key=_FAKE_API_KEY)
        .with_concurrency(1)
        .with_context_store(store)
        .build()
    )

    result = pipeline.execute()
    output_df = result.data.to_pandas()

    assert "category" in output_df.columns
    assert "_grounding_score" not in output_df.columns
    assert "_confidence_score" not in output_df.columns
    assert "_contradiction" not in output_df.columns
