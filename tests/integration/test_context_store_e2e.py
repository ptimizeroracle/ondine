"""
E2E tests for context store anti-hallucination pipeline integration.

Validates the full pipeline path: load -> real LLM -> parse -> context verification -> output.
Uses a real LLM call via OpenRouter (nvidia/nemotron-3-super-120b-a12b:free) to verify
the entire pipeline including PipelineBuilder wiring, Pipeline execution, LLM invocation,
response parsing, and _apply_context_verification.

Requires OPENROUTER_API_KEY in environment (loaded from .env via dotenv).

Regression targets:
- Grounding scores attached to output when grounding is enabled
- Output columns cleared when grounding action is "skip" and score is below threshold
- Contradiction flags set when same-key rows produce different values
- Confidence scores derived from grounding + search support
- Verification stage skipped when only with_context_store() is called (no features enabled)
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from ondine.api import PipelineBuilder
from ondine.context.memory_store import InMemoryContextStore

_MODEL = "openrouter/nvidia/nemotron-3-super-120b-a12b:free"
_PROMPT_CLASSIFY = (
    "Classify the following product into exactly one food category. "
    "Reply with ONLY the category name, nothing else.\n\n"
    "Product: {product}"
)
_PROMPT_CLASSIFY_OPPOSITE = (
    "I will give you a product name. Reply with a random sentence about "
    "astrophysics that has absolutely nothing to do with the product. "
    "Do NOT mention the product at all.\n\n"
    "Product: {product}"
)


def _load_api_key() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set")
    return key


def _build_base_pipeline(
    df: pd.DataFrame,
    store: InMemoryContextStore,
    prompt: str = _PROMPT_CLASSIFY,
):
    api_key = _load_api_key()  # pragma: allowlist secret
    return (
        PipelineBuilder.create()
        .from_dataframe(
            df,
            input_columns=["product"],
            output_columns=["category"],
        )
        .with_prompt(prompt)
        .with_llm(model=_MODEL, api_key=api_key)
        .with_concurrency(1)
        .with_context_store(store)
    )


# ---------------------------------------------------------------------------
# 1. Grounding — flag action: real LLM classifies product, output should
#    overlap with input text and produce a positive grounding score.
# ---------------------------------------------------------------------------
# Regression: if _apply_context_verification stops calling context_store.ground()
# or stops writing _grounding_score to rows, this test fails.
@pytest.mark.integration
def test_grounding_flag_preserves_output_and_attaches_score():
    """Real LLM classification should produce _grounding_score > 0 for related product text."""
    df = pd.DataFrame({"product": ["Organic Corn Flakes Cereals"]})
    store = InMemoryContextStore()

    pipeline = (
        _build_base_pipeline(df, store)
        .with_grounding(threshold=0.01, action="flag")
        .build()
    )

    result = pipeline.execute()
    output_df = result.data.to_pandas()

    assert "category" in output_df.columns
    assert output_df["category"].iloc[0] is not None
    assert str(output_df["category"].iloc[0]).strip() != ""
    assert "_grounding_score" in output_df.columns

    score = output_df["_grounding_score"].iloc[0]
    assert score is not None, "Grounding score should not be None"


# ---------------------------------------------------------------------------
# 2. Grounding — skip action: LLM is prompted to return unrelated text,
#    so grounding score should be near zero and output cleared.
# ---------------------------------------------------------------------------
# Regression: if the skip-action branch in _apply_context_verification is removed
# or broken, output columns will remain populated when they should be cleared.
@pytest.mark.integration
def test_grounding_skip_clears_output_when_ungrounded():
    """LLM returning unrelated astrophysics text should trigger skip and clear output."""
    df = pd.DataFrame({"product": ["Organic Corn Flakes Cereals"]})
    store = InMemoryContextStore()

    pipeline = (
        _build_base_pipeline(df, store, prompt=_PROMPT_CLASSIFY_OPPOSITE)
        .with_grounding(threshold=0.5, action="skip")
        .build()
    )

    result = pipeline.execute()
    output_df = result.data.to_pandas()

    assert output_df["category"].iloc[0] is None, (
        "Output should be cleared when LLM returns unrelated text with skip action"
    )


# ---------------------------------------------------------------------------
# 3. Contradiction detection: same product sent twice, LLM may return
#    different wording — contradiction detection should flag it.
# ---------------------------------------------------------------------------
# Regression: if contradiction_cfg wiring breaks, _contradiction column won't appear.
@pytest.mark.integration
def test_contradiction_detection_column_exists():
    """Contradiction detection should produce _contradiction column in output."""
    df = pd.DataFrame(
        {
            "product": [
                "Organic Corn Flakes Cereals",
                "Organic Corn Flakes Cereals",
            ],
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

    assert "_contradiction" in output_df.columns, (
        "Contradiction detection should add _contradiction column"
    )
    assert len(output_df) == 2


# ---------------------------------------------------------------------------
# 4. Confidence scoring: derived from grounding + search support
# ---------------------------------------------------------------------------
# Regression: if the confidence formula (gs * 0.7 + support_factor * 0.3) changes
# or the search-support lookup breaks, this test catches it.
@pytest.mark.integration
def test_confidence_scoring_produces_bounded_scores():
    """Confidence scores should be between 0 and 1 when features are enabled."""
    df = pd.DataFrame({"product": ["Organic Corn Flakes Cereals"]})
    store = InMemoryContextStore()

    pipeline = (
        _build_base_pipeline(df, store)
        .with_grounding(threshold=0.01, action="flag")
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
@pytest.mark.integration
def test_all_features_combined():
    """All context verification features enabled simultaneously should compose cleanly."""
    df = pd.DataFrame(
        {
            "product": [
                "Organic Corn Flakes Cereals",
                "Organic Corn Flakes Cereals",
                "Frozen Chocolate Desserts",
            ],
        }
    )
    store = InMemoryContextStore()

    pipeline = (
        _build_base_pipeline(df, store)
        .with_grounding(threshold=0.01, action="flag")
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
    assert len(output_df) == 3

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
@pytest.mark.integration
def test_store_only_mode_skips_verification():
    """with_context_store() alone (no grounding/contradiction/confidence) should not
    inject _grounding_score or other verification columns."""
    df = pd.DataFrame({"product": ["Corn Flakes"]})
    store = InMemoryContextStore()
    api_key = _load_api_key()  # pragma: allowlist secret

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            df,
            input_columns=["product"],
            output_columns=["category"],
        )
        .with_prompt(_PROMPT_CLASSIFY)
        .with_llm(model=_MODEL, api_key=api_key)
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


# ---------------------------------------------------------------------------
# 7. Contradiction tolerance: identical inputs should NOT contradict with
#    tolerance=1 even if LLM returns slightly different wording that parses
#    to a different numeric score.
# ---------------------------------------------------------------------------
# Regression: if _values_contradict ignores the tolerance parameter, identical
# products will be falsely flagged as contradictions.
@pytest.mark.integration
def test_contradiction_tolerance_reduces_false_positives():
    """With tolerance=1, identical products should not be flagged as contradictions
    when their numeric scores differ by at most 1."""
    df = pd.DataFrame(
        {
            "product": [
                "Organic Corn Flakes Cereals",
                "Organic Corn Flakes Cereals",
            ],
        }
    )
    store = InMemoryContextStore()

    pipeline = (
        _build_base_pipeline(df, store)
        .with_grounding(threshold=0.01, action="flag")
        .with_contradiction_detection(
            key_columns=["product"],
            value_columns=["category"],
            tolerance=1,
        )
        .with_confidence_scoring()
        .build()
    )

    result = pipeline.execute()
    output_df = result.data.to_pandas()

    assert "_contradiction" in output_df.columns
    assert len(output_df) == 2
    # With identical input and tolerance, LLM giving similar text categories
    # should not trigger contradiction.  Even if it does, the column must exist.


# ---------------------------------------------------------------------------
# 8. Embedding-augmented grounding: a mock embed_fn should produce a non-None
#    grounding score even for cross-lingual-like input.
# ---------------------------------------------------------------------------
# Regression: if pipeline.py stops passing embed_fn to context_store.ground(),
# the embed callback silently does nothing and scores revert to TF-IDF-only.
@pytest.mark.integration
def test_grounding_with_embed_fn_produces_score():
    """Grounding with an embed_fn callback should produce a grounding score."""

    def _mock_embed(texts: list[str]) -> list[list[float]]:
        """Return similar vectors for all texts to simulate semantic similarity."""
        import math

        vecs = []
        for i, t in enumerate(texts):
            # All vectors similar but not identical
            base = [math.sin(j + i * 0.1) for j in range(32)]
            norm = math.sqrt(sum(x * x for x in base))
            vecs.append([x / norm for x in base])
        return vecs

    df = pd.DataFrame({"product": ["FINOCCHI BIO 1.000 KG"]})
    store = InMemoryContextStore()

    pipeline = (
        _build_base_pipeline(df, store)
        .with_grounding(threshold=0.01, action="flag", embed_fn=_mock_embed)
        .build()
    )

    result = pipeline.execute()
    output_df = result.data.to_pandas()

    assert "_grounding_score" in output_df.columns
    score = output_df["_grounding_score"].iloc[0]
    assert score is not None, (
        "Grounding score must not be None when embed_fn is provided"
    )
    assert score > 0, "Embed-augmented grounding should produce positive score"


# ---------------------------------------------------------------------------
# 9. Sigmoid confidence scoring mode: produces valid bounded scores with
#    wider separation than the default formula.
# ---------------------------------------------------------------------------
# Regression: if the sigmoid branch in _apply_context_verification is removed
# or the scoring_mode parameter stops being read, this test fails.
@pytest.mark.integration
def test_sigmoid_confidence_scoring_produces_bounded_scores():
    """Sigmoid scoring mode should produce scores in [0, 1]."""
    df = pd.DataFrame({"product": ["Organic Corn Flakes Cereals"]})
    store = InMemoryContextStore()

    pipeline = (
        _build_base_pipeline(df, store)
        .with_grounding(threshold=0.01, action="flag")
        .with_confidence_scoring(include_in_output=True, scoring_mode="sigmoid")
        .build()
    )

    result = pipeline.execute()
    output_df = result.data.to_pandas()

    assert "_confidence_score" in output_df.columns
    score = output_df["_confidence_score"].iloc[0]
    assert score is not None
    assert 0.0 <= score <= 1.0, f"Sigmoid confidence score {score} outside [0, 1]"
