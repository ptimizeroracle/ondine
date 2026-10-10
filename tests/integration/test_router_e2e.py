"""
E2E integration tests for LiteLLM Router.

Tests load balancing, failover, and multi-provider routing.
"""

import os

import pandas as pd
import pytest

from ondine import PipelineBuilder
from tests.integration.live_models import FREE_KEY_ENV, FREE_MODEL, FREE_PROVIDER


@pytest.mark.integration
def test_router_multi_provider_fallback():
    """
    E2E test for Router with multi-provider failover.

    Tests that Router can load balance between a free OpenRouter model and OpenAI,
    with automatic failover if one provider fails.
    """
    free_key = os.getenv(FREE_KEY_ENV)
    openai_key = os.getenv("OPENAI_API_KEY")

    if not free_key or not openai_key:
        pytest.skip(f"{FREE_KEY_ENV} and OPENAI_API_KEY both required for Router test")

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
                        "model": FREE_MODEL,
                        "api_key": free_key,
                        "rpm": 20,  # OpenRouter free-tier limit
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
    df = result.to_pandas()
    assert result.success
    assert len(df) == 2
    assert df["answer"].notnull().all()

    print("\nRouter Multi-Provider E2E:")
    print(df)
    print(f"Cost: ${result.costs.total_cost:.4f}")
    print("Note: Router automatically picked best deployment!")


@pytest.mark.integration
def test_router_same_provider_load_balance():
    """
    E2E test for Router load balancing across same provider.

    Tests load balancing across multiple deployments of one model
    (simulates multi-region or multi-account scenarios).
    """
    free_key = os.getenv(FREE_KEY_ENV)

    if not free_key:
        pytest.skip(f"{FREE_KEY_ENV} required")

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
                    "model_name": "free-llm",
                    "litellm_params": {
                        "model": FREE_MODEL,
                        "api_key": free_key,
                        "rpm": 9,  # Low limit to test balancing
                    },
                },
                {
                    "model_name": "free-llm",  # Same model_name = load balance
                    "litellm_params": {
                        "model": FREE_MODEL,
                        "api_key": free_key,
                        "rpm": 9,
                    },
                },
            ],
            routing_strategy="simple-shuffle",
        )
        .build()
    )

    result = pipeline.execute()
    df = result.to_pandas()

    assert result.success
    assert len(df) == 3
    print("\nRouter Load Balancing E2E:")
    print(f"Processed: {len(df)} rows")
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
    free_key = os.getenv(FREE_KEY_ENV)

    if not free_key:
        pytest.skip(f"{FREE_KEY_ENV} required")

    df = pd.DataFrame({"text": ["Cached test"] * 2})  # Duplicate prompts

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["text"], output_columns=["result"])
        .with_prompt("Echo: {text}")
        .with_llm(provider=FREE_PROVIDER, model=FREE_MODEL, api_key=free_key)
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
