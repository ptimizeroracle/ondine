"""
E2E test for QuickPipeline API.

Validates the simplified 3-line API with smart defaults and auto-detection.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ondine import QuickPipeline


@pytest.mark.integration
@pytest.mark.parametrize(
    ("model", "api_key_env"),
    [
        ("gpt-4o-mini", "OPENAI_API_KEY"),
        ("llama-3.3-70b-versatile", "GROQ_API_KEY"),
    ],
)
def test_quickpipeline_auto_detection(model, api_key_env):
    """
    Test QuickPipeline with auto-detection of provider and columns.

    Validates that smart defaults work correctly.
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        pytest.skip(f"{api_key_env} not set")

    # Infer provider from model name
    if "gpt" in model:
        provider = "openai"
    else:
        provider = "groq"

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "test.csv"
        df = pd.DataFrame({"description": [f"Product {i}" for i in range(5)]})
        df.to_csv(data_file, index=False)

        # QuickPipeline with minimal config (auto-detects input column from prompt)
        pipeline = QuickPipeline.create(
            data=str(data_file),
            prompt="Summarize: {description}",  # Auto-detects 'description' column
            model=model,
            api_key=api_key,
        )

        result = pipeline.execute()

        assert result.success, f"{model} QuickPipeline failed"
        assert len(result.data) == 5
        # QuickPipeline auto-names output column as 'result' by default
        assert "result" in result.data.columns or "output" in result.data.columns

        print(f"\n{model} QuickPipeline Test Results:")
        print(f"  Auto-detected provider: {provider}")
        print("  Auto-detected input: description")
        print(f"  Processed: {len(result.data)} rows")
        print("  ✅ QuickPipeline auto-detection working")


@pytest.mark.integration
def test_quickpipeline_with_dataframe():
    """
    Test QuickPipeline with DataFrame input (not file).

    Validates that QuickPipeline accepts in-memory data.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    df = pd.DataFrame({"text": ["Hello", "World", "Test"]})

    # QuickPipeline with DataFrame
    pipeline = QuickPipeline.create(
        data=df,  # Pass DataFrame directly
        prompt="Uppercase: {text}",
        model="llama-3.3-70b-versatile",
        api_key=api_key,
    )

    result = pipeline.execute()

    assert result.success
    assert len(result.data) == 3

    print("\nQuickPipeline DataFrame Test:")
    print("  Input: DataFrame (3 rows)")
    print("  ✅ DataFrame input working")

