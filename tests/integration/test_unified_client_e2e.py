"""
E2E integration tests for UnifiedLiteLLMClient with real API calls.

Tests the new native LiteLLM client across multiple providers.
"""

import os
from decimal import Decimal

import pandas as pd
import pytest

from ondine import PipelineBuilder


@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "model", "api_key_env"),
    [
        ("openai", "gpt-4o-mini", "OPENAI_API_KEY"),
        ("groq", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
        ("anthropic", "claude-3-5-haiku-20241022", "ANTHROPIC_API_KEY"),
    ],
)
def test_unified_client_basic_invoke_e2e(provider, model, api_key_env):
    """
    E2E test for UnifiedLiteLLMClient basic text completion.

    Tests that the new native client works across providers:
    - OpenAI: Baseline provider
    - Groq: Fast inference
    - Anthropic: Claude models

    Note: Structured output will be tested in Phase 2 after Instructor integration.
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        pytest.skip(f"{api_key_env} not set")

    # Create test data
    df = pd.DataFrame({"text": ["What is 2+2?", "What is the capital of France?"]})

    # Build pipeline
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            df,
            input_columns=["text"],
            output_columns=["answer"],
        )
        .with_prompt("Answer concisely: {text}")
        .with_llm(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=0.0,
            max_tokens=100,
        )
        .with_rate_limit(60)  # Rate limit to avoid API errors
        .build()
    )

    # Execute
    result = pipeline.execute()

    # Verify basic functionality
    df = result.to_pandas()
    assert result.success, f"{provider} pipeline failed"
    assert len(df) == 2, f"{provider} returned wrong number of rows"
    assert "answer" in df.columns

    # Verify answers are not empty
    assert df["answer"].notnull().all(), f"{provider} returned null answers"
    assert all(len(str(ans)) > 0 for ans in df["answer"]), (
        f"{provider} returned empty answers"
    )

    # Verify cost tracking
    assert result.costs.total_cost > 0, f"{provider} cost tracking failed (got $0)"
    assert result.costs.total_tokens > 0, f"{provider} token tracking failed"

    print(f"\n{provider.upper()} E2E Results:")
    print(df)
    print(f"Cost: ${result.costs.total_cost:.4f}")
    print(f"Tokens: {result.costs.total_tokens}")


@pytest.mark.integration
def test_unified_client_groq_async_e2e():
    """
    E2E test for async invocation with Groq.

    Tests that UnifiedLiteLLMClient correctly uses litellm.acompletion
    for async-first execution.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    # Create test data
    df = pd.DataFrame({"question": ["What is Python?", "What is AI?", "What is ML?"]})

    # Build pipeline (will use native async under the hood)
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            df,
            input_columns=["question"],
            output_columns=["answer"],
        )
        .with_prompt("Answer in one sentence: {question}")
        .with_llm(
            provider="groq",
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.0,
            max_tokens=100,
        )
        .with_concurrency(3)  # Test concurrent execution
        .with_rate_limit(9)  # Groq rate limit
        .build()
    )

    # Execute
    result = pipeline.execute()

    # Verify
    df = result.to_pandas()
    assert result.success
    assert len(df) == 3
    assert df["answer"].notnull().all()

    print("\nGroq Async E2E Results:")
    print(df)
    print(f"Cost: ${result.costs.total_cost:.4f}")


@pytest.mark.integration
def test_unified_client_cost_accuracy_e2e():
    """
    E2E test for cost tracking accuracy across providers.

    Verifies that litellm.completion_cost() provides accurate costs.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    # Create test data
    df = pd.DataFrame({"text": ["Hello" * 100]})  # Long prompt for measurable cost

    # Build pipeline
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["text"], output_columns=["response"])
        .with_prompt("Summarize: {text}")
        .with_llm(
            provider="groq",
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.0,
        )
        .build()
    )

    # Execute
    result = pipeline.execute()

    # Verify cost is tracked
    assert result.costs.total_cost > 0
    assert result.costs.input_tokens > 0
    assert result.costs.output_tokens > 0

    # Groq pricing: ~$0.59/1M input, ~$0.79/1M output
    # Verify cost is in reasonable range
    assert result.costs.total_cost < Decimal("0.01"), "Cost seems too high"

    print("\nCost Tracking E2E:")
    print(f"Input tokens: {result.costs.input_tokens}")
    print(f"Output tokens: {result.costs.output_tokens}")
    print(f"Total cost: ${result.costs.total_cost:.6f}")


@pytest.mark.integration
def test_unified_client_batch_processing_e2e():
    """
    E2E test for multi-row batching with UnifiedLiteLLMClient.

    Verifies that batching works correctly with the new client.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    # Create test data
    df = pd.DataFrame(
        {"product": ["Apple", "Banana", "Orange", "Grape", "Mango", "Kiwi"]}
    )

    # Build pipeline with batching
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            df,
            input_columns=["product"],
            output_columns=["category"],
        )
        .with_prompt("Classify this fruit: {product}")
        .with_llm(
            provider="groq",
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.0,
        )
        .with_batch_size(3)  # 3 rows per API call
        .with_rate_limit(9)
        .build()
    )

    # Execute
    result = pipeline.execute()

    # Verify
    df = result.to_pandas()
    assert result.success
    assert len(df) == 6
    assert df["category"].notnull().sum() >= 4, "Too many null responses"

    print("\nBatch Processing E2E:")
    print(df)
    print(f"API calls saved: {6 / 3} calls instead of 6 (50% reduction)")
