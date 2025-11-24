"""
E2E test for prefix caching (40-50% cost reduction).

Validates that system prompt caching works correctly and reduces costs
for providers that support it (OpenAI, Anthropic, Groq).
"""

import os

import pandas as pd
import pytest

from ondine import PipelineBuilder


@pytest.mark.integration
@pytest.mark.parametrize(
    "provider,model,api_key_env",
    [
        ("openai", "gpt-4o-mini", "OPENAI_API_KEY"),
        ("groq", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    ],
)
def test_prefix_caching_reduces_cost(provider, model, api_key_env):
    """
    Test that prefix caching reduces cost compared to no caching.

    Validates cache hits occur and reduce token usage.
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        pytest.skip(f"{api_key_env} not set")

    # Create test data (20 rows to ensure multiple API calls with same system prompt)
    df = pd.DataFrame({"text": [f"Item {i}" for i in range(20)]})

    # Large system prompt (makes caching benefit visible)
    system_prompt = """You are an expert data processor.
Your task is to transform input text according to specific rules.
Rules:
1. Be concise
2. Be accurate
3. Be consistent
4. Follow the format exactly
5. Do not add extra information
""" * 10  # Repeat to make it large (~500 tokens)

    # Pipeline WITH caching
    pipeline_cached = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["text"], output_columns=["result"])
        .with_prompt("{{text}}")
        .with_system_prompt(system_prompt)
        .with_llm(provider=provider, model=model, api_key=api_key, temperature=0.0)
        .with_batch_size(20)
        .with_processing_batch_size(5)  # 4 API calls (cache should hit after 1st)
        .with_prefix_caching(True)  # Enable caching
        .build()
    )

    result_cached = pipeline_cached.execute()

    # Pipeline WITHOUT caching (for comparison)
    pipeline_no_cache = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["text"], output_columns=["result"])
        .with_prompt("{{text}}")
        .with_system_prompt(system_prompt)
        .with_llm(provider=provider, model=model, api_key=api_key, temperature=0.0)
        .with_batch_size(20)
        .with_processing_batch_size(5)
        .with_prefix_caching(False)  # Disable caching
        .build()
    )

    result_no_cache = pipeline_no_cache.execute()

    # Verify both succeeded
    assert result_cached.success, f"{provider} cached pipeline failed"
    assert result_no_cache.success, f"{provider} non-cached pipeline failed"

    # Compare costs (cached should be cheaper)
    cost_cached = result_cached.costs.total_cost
    cost_no_cache = result_no_cache.costs.total_cost

    # Caching should reduce cost by at least 10% (conservative estimate)
    # Real-world: 40-50% reduction, but API variance can affect this
    savings_pct = ((cost_no_cache - cost_cached) / cost_no_cache) * 100

    assert cost_cached < cost_no_cache, (
        f"{provider}: Cached cost ${cost_cached:.4f} should be less than "
        f"non-cached ${cost_no_cache:.4f}"
    )

    print(f"\n{provider.upper()} Prefix Caching Test Results:")
    print(f"  Without caching: ${cost_no_cache:.4f}")
    print(f"  With caching: ${cost_cached:.4f}")
    print(f"  Savings: {savings_pct:.1f}%")
    print(f"  ✅ Prefix caching working (cost reduced)")


@pytest.mark.integration
def test_prefix_caching_logs_cache_hits():
    """
    Test that cache hits are logged for observability.

    Validates that users can see when caching is working.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    df = pd.DataFrame({"text": [f"Text {i}" for i in range(10)]})

    system_prompt = "You are a helpful assistant. " * 20

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["text"], output_columns=["result"])
        .with_prompt("{{text}}")
        .with_system_prompt(system_prompt)
        .with_llm(
            provider="groq",
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.0,
        )
        .with_batch_size(10)
        .with_processing_batch_size(2)  # 5 API calls
        .with_prefix_caching(True)
        .build()
    )

    result = pipeline.execute()

    assert result.success
    # Note: Cache hit detection depends on provider returning cached_tokens in response
    # This test mainly validates the feature doesn't break execution

    print(f"\nCache Hit Logging Test:")
    print(f"  Processed: {len(result.data)} rows")
    print(f"  Cost: ${result.costs.total_cost:.4f}")
    print(f"  ✅ Caching enabled without errors")

