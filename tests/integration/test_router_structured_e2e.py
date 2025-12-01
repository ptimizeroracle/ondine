"""
E2E integration tests for Router + Structured Output (CRITICAL COMBO).

This test validates that Router works correctly with Instructor-based
structured output, which is used by the bacon cleaner and similar pipelines.

THIS TEST SHOULD HAVE CAUGHT THE BUG!
"""

import os

import pandas as pd
import pytest
from pydantic import BaseModel, Field

from ondine import PipelineBuilder


class ExtractedData(BaseModel):
    """Test model for structured output (user fields only - no id!)."""

    answer: str = Field(description="The answer to the question")
    confidence: float = Field(description="Confidence score 0-1")


class BatchItem(BaseModel):
    """Single item in batch (internal structure)."""

    id: int = Field(description="Internal tracking ID")
    result: ExtractedData = Field(description="User's actual data")


class BatchResponse(BaseModel):
    """Batch of extracted data (like BaconBatch)."""

    items: list[BatchItem]


@pytest.mark.integration
def test_router_with_structured_output_groq_openai():
    """
    CRITICAL TEST: Router + Structured Output (Groq + OpenAI).

    This is the EXACT scenario used by bacon cleaner:
    - Router with Groq (primary) + OpenAI (fallback)
    - Structured output with Pydantic model
    - Batch processing

    If this test passes, bacon cleaner should work!
    """
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not groq_key or not openai_key:
        pytest.skip("GROQ_API_KEY and OPENAI_API_KEY both required")

    # Create test data (small batch)
    df = pd.DataFrame({"question": ["What is 2+2?", "What is 5+5?"]})

    # Build pipeline with Router + Structured Output
    # This is EXACTLY like bacon cleaner (with batch processing!)
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            df,
            input_columns=["question"],
            output_columns=["answer", "confidence"],  # User fields only (no id)
        )
        .with_prompt("Answer the question: {{ question }}")
        .with_batch_size(2)  # CRITICAL: Enable batch processing!
        .with_router(
            model_list=[
                {
                    "model_name": "test-llm",  # Shared model_name
                    "litellm_params": {
                        "model": "groq/llama-3.3-70b-versatile",
                        "api_key": groq_key,
                        "temperature": 0.1,
                        "max_tokens": 1000,
                    },
                },
                {
                    "model_name": "test-llm",  # Same name = automatic failover
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": openai_key,
                        "temperature": 0.1,
                        "max_tokens": 1000,
                    },
                },
            ],
            routing_strategy="simple-shuffle",
        )
        .with_structured_output(BatchResponse)  # CRITICAL: Structured output!
        .build()
    )

    # Execute
    result = pipeline.execute()

    # DEBUG: Print actual data to see what we got
    df = result.to_pandas()
    print("\n🔍 DEBUG - Result Data:")
    print(df)
    print("\n🔍 DEBUG - Columns:")
    print(df.columns.tolist())
    print("\n🔍 DEBUG - Data types:")
    print(df.dtypes)

    # Verify
    assert result.success, "Pipeline should succeed"
    assert len(df) == 2, "Should process all rows"

    # Check required fields (user fields only - id is internal)
    assert "answer" in df.columns, "Should have answer column"
    assert "confidence" in df.columns, "Should have confidence column"

    assert df["answer"].notnull().all(), "All answers should be non-null"
    assert df["confidence"].notnull().all(), "All confidence scores should be non-null"

    print("\n✅ Router + Structured Output E2E:")
    print(df)
    print(f"💰 Cost: ${result.costs.total_cost:.4f}")
    print("🎯 This validates the bacon cleaner scenario!")


@pytest.mark.integration
def test_router_structured_output_groq_only():
    """
    Test Router + Structured Output with Groq only.

    Simpler test with single provider.
    """
    groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        pytest.skip("GROQ_API_KEY required")

    df = pd.DataFrame({"q": ["What is AI?"]})

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            df, input_columns=["q"], output_columns=["answer", "confidence"]
        )
        .with_prompt("{q}")
        .with_router(
            model_list=[
                {
                    "model_name": "groq-test",
                    "litellm_params": {
                        "model": "groq/llama-3.3-70b-versatile",
                        "api_key": groq_key,
                        "temperature": 0.1,
                        "max_tokens": 1000,
                    },
                }
            ],
            routing_strategy="simple-shuffle",
        )
        .with_structured_output(BatchResponse)
        .build()
    )

    result = pipeline.execute()
    df = result.to_pandas()

    assert result.success
    assert len(df) == 1
    print("\n✅ Router + Structured (Groq only):")
    print(df)


@pytest.mark.integration
def test_router_structured_large_batch():
    """
    Test Router + Structured Output with larger batch (like bacon cleaner).

    This simulates the bacon cleaner's batch size of 50 rows.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not groq_key or not openai_key:
        pytest.skip("GROQ_API_KEY and OPENAI_API_KEY required")

    # Create 10-row batch (smaller than 50 for speed)
    questions = [f"What is {i}+{i}?" for i in range(1, 11)]
    df = pd.DataFrame({"question": questions})

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            df, input_columns=["question"], output_columns=["answer", "confidence"]
        )
        .with_prompt("{{ question }}")
        .with_router(
            model_list=[
                {
                    "model_name": "bacon-llm",
                    "litellm_params": {
                        "model": "groq/llama-3.3-70b-versatile",
                        "api_key": groq_key,
                        "temperature": 0.1,
                        "max_tokens": 2000,
                    },
                },
                {
                    "model_name": "bacon-llm",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": openai_key,
                        "temperature": 0.1,
                        "max_tokens": 2000,
                    },
                },
            ],
            routing_strategy="simple-shuffle",
        )
        .with_batch_size(10)  # Batch all 10 into single call
        .with_structured_output(BatchResponse)
        .build()
    )

    result = pipeline.execute()
    df = result.to_pandas()

    assert result.success
    assert len(df) == 10
    print(f"\n✅ Router + Structured Batch ({len(df)} rows):")
    print(f"💰 Cost: ${result.costs.total_cost:.4f}")
    print(f"📊 Rows/call: {len(df)}")
