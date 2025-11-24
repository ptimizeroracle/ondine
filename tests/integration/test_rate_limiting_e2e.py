"""
E2E test for rate limiting (Token Bucket algorithm).

Validates that the custom RateLimiter correctly throttles API requests
to stay within specified RPM limits under real load.
"""

import os
import time

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
def test_rate_limiting_enforces_rpm(provider, model, api_key_env):
    """
    Test that rate limiting correctly throttles requests to specified RPM.

    This validates the Token Bucket algorithm under real API load.
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        pytest.skip(f"{api_key_env} not set")

    # Create test data (20 rows)
    df = pd.DataFrame({"text": [f"Text {i}" for i in range(20)]})

    # Configure pipeline with strict rate limit: 10 RPM
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["text"], output_columns=["result"])
        .with_prompt("Summarize in 3 words: {{text}}")
        .with_llm(provider=provider, model=model, api_key=api_key, temperature=0.0)
        .with_batch_size(20)  # 1 API call total
        .with_processing_batch_size(1)  # 1 row per batch (20 API calls)
        .with_rate_limit(10)  # 10 requests per minute
        .with_concurrency(5)  # High concurrency to test throttling
        .build()
    )

    # Execute and measure actual throughput
    start_time = time.time()
    result = pipeline.execute()
    elapsed_time = time.time() - start_time

    # Verify success
    assert result.success, f"{provider} pipeline failed"
    assert len(result.data) == 20, f"Expected 20 rows, got {len(result.data)}"

    # Calculate actual RPM
    actual_rpm = (20 / elapsed_time) * 60

    # Verify rate limiting worked (allow 20% tolerance for API latency variance)
    # Expected: 10 RPM, so 20 requests should take ~2 minutes (120 seconds)
    # With tolerance: 1.6-2.4 minutes (96-144 seconds)
    assert elapsed_time >= 96, (
        f"{provider}: Too fast! {elapsed_time:.1f}s (expected ≥96s for 10 RPM). "
        f"Actual RPM: {actual_rpm:.1f}"
    )
    assert elapsed_time <= 144, (
        f"{provider}: Too slow! {elapsed_time:.1f}s (expected ≤144s). "
        f"Check if rate limiter is too conservative."
    )

    print(f"\n{provider.upper()} Rate Limiting Results:")
    print(f"  Configured: 10 RPM")
    print(f"  Actual: {actual_rpm:.1f} RPM")
    print(f"  Duration: {elapsed_time:.1f}s for 20 requests")
    print(f"  ✅ Rate limiting working correctly")


@pytest.mark.integration
def test_rate_limiting_with_burst_control():
    """
    Test that burst_size prevents initial request flooding.

    Validates that the first N requests are also throttled, not sent all at once.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    # Small dataset to measure initial burst
    df = pd.DataFrame({"text": [f"Text {i}" for i in range(5)]})

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["text"], output_columns=["result"])
        .with_prompt("Echo: {{text}}")
        .with_llm(provider="groq", model="llama-3.3-70b-versatile", api_key=api_key)
        .with_batch_size(5)
        .with_processing_batch_size(1)  # 5 API calls
        .with_rate_limit(30)  # 30 RPM = 1 request per 2 seconds
        .with_concurrency(5)  # All 5 could fire at once without rate limiting
        .build()
    )

    start_time = time.time()
    result = pipeline.execute()
    elapsed_time = time.time() - start_time

    # With burst control, 5 requests at 30 RPM should take ~10 seconds minimum
    # (2 seconds per request)
    assert elapsed_time >= 8, (
        f"Burst not controlled! {elapsed_time:.1f}s for 5 requests. "
        f"Expected ≥8s (burst_size should prevent instant flooding)"
    )

    print(f"\nBurst Control Results:")
    print(f"  5 requests at 30 RPM took {elapsed_time:.1f}s")
    print(f"  ✅ Burst control working (no instant flood)")

    assert result.success

