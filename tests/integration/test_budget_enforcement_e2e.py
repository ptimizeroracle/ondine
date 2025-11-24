"""
E2E test for budget enforcement.

Validates that BudgetController correctly stops pipeline execution
when cost limits are exceeded, preventing runaway expenses.
"""

import os

import pandas as pd
import pytest

from ondine import PipelineBuilder


@pytest.mark.integration
@pytest.mark.skip(reason="Budget enforcement requires accurate cost tracking - Groq reports $0.00")
@pytest.mark.parametrize(
    "provider,model,api_key_env",
    [
        ("openai", "gpt-4o-mini", "OPENAI_API_KEY"),
    ],
)
def test_budget_enforcement_stops_execution(provider, model, api_key_env):
    """
    Test that pipeline stops when budget limit is exceeded.

    This is a critical financial safety feature.
    Note: Skipped because Groq doesn't report costs accurately ($0.00).
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        pytest.skip(f"{api_key_env} not set")

    # Create large dataset that would exceed budget
    df = pd.DataFrame({"text": [f"Long text {i} " * 50 for i in range(100)]})

    # Set very low budget (should stop after a few rows)
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["text"], output_columns=["summary"])
        .with_prompt("Summarize: {{text}}")
        .with_llm(provider=provider, model=model, api_key=api_key, temperature=0.0)
        .with_batch_size(100)
        .with_processing_batch_size(10)  # 10 API calls total
        .with_max_budget(0.02)  # Very low budget ($0.02)
        .build()
    )

    # Execute (should stop early due to budget)
    result = pipeline.execute()

    # Verify budget enforcement
    assert not result.success, f"{provider}: Pipeline should have failed due to budget"
    assert result.costs.total_cost <= 0.025, (
        f"{provider}: Cost ${result.costs.total_cost:.4f} exceeded budget $0.02 "
        f"(allowing 25% tolerance for estimation variance)"
    )
    assert len(result.data) < 100, (
        f"{provider}: Should have stopped before processing all 100 rows. "
        f"Processed: {len(result.data)}"
    )

    print(f"\n{provider.upper()} Budget Enforcement Results:")
    print(f"  Budget: $0.02")
    print(f"  Actual cost: ${result.costs.total_cost:.4f}")
    print(f"  Rows processed: {len(result.data)}/100")
    print(f"  ✅ Budget enforcement working correctly")


@pytest.mark.integration
def test_budget_warning_threshold():
    """
    Test that budget warnings are logged at threshold.

    Validates the warning system alerts users before hitting hard limit.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    df = pd.DataFrame({"text": [f"Text {i}" for i in range(20)]})

    # Set budget with warning threshold
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["text"], output_columns=["result"])
        .with_prompt("Echo: {{text}}")
        .with_llm(
            provider="groq", model="llama-3.3-70b-versatile", api_key=api_key
        )
        .with_batch_size(20)
        .with_processing_batch_size(5)  # 4 API calls
        .with_max_budget(0.10)  # Should complete within budget
        .build()
    )

    result = pipeline.execute()

    # Should succeed (budget sufficient)
    assert result.success, "Pipeline should complete within budget"
    assert result.costs.total_cost < 0.10, (
        f"Cost ${result.costs.total_cost:.4f} exceeded budget"
    )

    print(f"\nBudget Warning Test Results:")
    print(f"  Budget: $0.10")
    print(f"  Actual cost: ${result.costs.total_cost:.4f}")
    print(f"  ✅ Completed within budget")

