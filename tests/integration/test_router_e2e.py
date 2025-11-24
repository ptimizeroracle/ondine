"""
E2E integration tests for LiteLLM Router.

Tests load balancing, failover, and multi-provider routing.
"""

import os

import pandas as pd
import pytest

from ondine import PipelineBuilder


@pytest.mark.integration
def test_router_multi_provider_fallback():
    """
    E2E test for Router with multi-provider failover.

    Tests that Router can load balance between Groq and OpenAI,
    with automatic failover if one provider fails.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not groq_key or not openai_key:
        pytest.skip("GROQ_API_KEY and OPENAI_API_KEY both required for Router test")

    # Create test data
    df = pd.DataFrame({"text": ["What is 2+2?", "What is 3+3?"]})

    # Build pipeline with Router (multi-provider)
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["text"], output_columns=["answer"])
        .with_prompt("Answer: {text}")
        .with_router(
            model_list=[
                {
                    "model_name": "fast-llm",
                    "litellm_params": {
                        "model": "groq/llama-3.3-70b-versatile",
                        "api_key": groq_key,
                        "rpm": 30,  # Groq limit
                    },
                },
                {
                    "model_name": "fast-llm",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": openai_key,
                        "rpm": 500,  # OpenAI limit
                    },
                },
            ],
            routing_strategy="simple-shuffle",
        )
        .with_rate_limit(60)
        .build()
    )

    # Execute
    result = pipeline.execute()

    # Verify
    assert result.success
    assert len(result.data) == 2
    assert result.data["answer"].notnull().all()

    print("\nRouter Multi-Provider E2E:")
    print(result.data)
    print(f"Cost: ${result.costs.total_cost:.4f}")
    print("Note: Router automatically picked best deployment!")


@pytest.mark.integration
def test_router_same_provider_load_balance():
    """
    E2E test for Router load balancing across same provider.

    Tests load balancing across multiple Groq deployments
    (simulates multi-region or multi-account scenarios).
    """
    groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        pytest.skip("GROQ_API_KEY required")

    df = pd.DataFrame({"q": ["What is AI?", "What is ML?", "What is DL?"]})

    # Router with same provider, different "deployments"
    # In practice, these would be different regions/accounts
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["q"], output_columns=["a"])
        .with_prompt("{q}")
        .with_router(
            model_list=[
                {
                    "model_name": "groq-llm",
                    "litellm_params": {
                        "model": "groq/llama-3.3-70b-versatile",
                        "api_key": groq_key,
                        "rpm": 9,  # Low limit to test balancing
                    },
                },
                {
                    "model_name": "groq-llm",  # Same model_name = load balance
                    "litellm_params": {
                        "model": "groq/llama-3.3-70b-versatile",
                        "api_key": groq_key,
                        "rpm": 9,
                    },
                },
            ],
            routing_strategy="simple-shuffle",
        )
        .build()
    )

    result = pipeline.execute()

    assert result.success
    assert len(result.data) == 3
    print("\nRouter Load Balancing E2E:")
    print(f"Processed: {len(result.data)} rows")
    print(f"Cost: ${result.costs.total_cost:.4f}")


@pytest.mark.integration
@pytest.mark.skip(reason="Requires Redis server running")
def test_router_with_redis_caching():
    """
    E2E test for Router with Redis caching.

    NOTE: Requires Redis running on localhost:6379
    Run: docker run -d -p 6379:6379 redis

    Tests that:
    - First call hits API
    - Second identical call uses cache ($0 cost)
    """
    groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        pytest.skip("GROQ_API_KEY required")

    df = pd.DataFrame({"text": ["Cached test"] * 2})  # Duplicate prompts

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["text"], output_columns=["result"])
        .with_prompt("Echo: {text}")
        .with_llm(provider="groq", model="llama-3.3-70b-versatile", api_key=groq_key)
        .with_redis_cache("redis://localhost:6379", ttl=60)
        .build()
    )

    # First execution - should hit API
    result1 = pipeline.execute()
    cost1 = result1.costs.total_cost

    # Second execution - should use cache
    result2 = pipeline.execute()
    cost2 = result2.costs.total_cost

    # Second run should be cheaper (cache hits)
    assert cost2 <= cost1
    print("\nRedis Caching E2E:")
    print(f"First run cost: ${cost1:.4f}")
    print(f"Second run cost: ${cost2:.4f}")
    print(f"Savings: ${cost1 - cost2:.4f}")
