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
from tests.integration.live_models import FREE_KEY_ENV, FREE_MODEL, FREE_PROVIDER


@pytest.mark.integration
@pytest.mark.parametrize(
    ("model", "api_key_env", "provider"),
    [
        # None lets QuickPipeline detect the provider from the model name.
        ("gpt-4o-mini", "OPENAI_API_KEY", None),
        # A prefixed LiteLLM model id is not auto-detected; name the provider.
        (FREE_MODEL, FREE_KEY_ENV, FREE_PROVIDER),
    ],
)
def test_quickpipeline_auto_detection(model, api_key_env, provider):
    """
    Test QuickPipeline with auto-detection of provider and columns.

    Validates that smart defaults work correctly.
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        pytest.skip(f"{api_key_env} not set")

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "test.csv"
        df = pd.DataFrame({"description": [f"Product {i}" for i in range(5)]})
        df.to_csv(data_file, index=False)

        # QuickPipeline with minimal config (auto-detects input column from prompt)
        pipeline = QuickPipeline.create(
            data=str(data_file),
            prompt="Summarize: {description}",  # Auto-detects 'description' column
            model=model,
            provider=provider,
            api_key=api_key,
        )

        result = pipeline.execute()
        df = result.to_pandas()

        assert result.success, f"{model} QuickPipeline failed"
        assert len(df) == 5
        # QuickPipeline auto-names output column as 'result' by default
        assert "result" in df.columns or "output" in df.columns

        print(f"\n{model} QuickPipeline Test Results:")
        print(f"  Auto-detected provider: {provider}")
        print("  Auto-detected input: description")
        print(f"  Processed: {len(df)} rows")
        print("  ✅ QuickPipeline auto-detection working")


@pytest.mark.integration
def test_quickpipeline_with_dataframe():
    """
    Test QuickPipeline with DataFrame input (not file).

    Validates that QuickPipeline accepts in-memory data.
    """
    api_key = os.getenv(FREE_KEY_ENV)
    if not api_key:
        pytest.skip(f"{FREE_KEY_ENV} not set")

    df = pd.DataFrame({"text": ["Hello", "World", "Test"]})

    # QuickPipeline with DataFrame
    pipeline = QuickPipeline.create(
        data=df,  # Pass DataFrame directly
        prompt="Uppercase: {text}",
        model=FREE_MODEL,
        provider=FREE_PROVIDER,
        api_key=api_key,
    )

    result = pipeline.execute()
    df = result.to_pandas()

    assert result.success
    assert len(df) == 3

    print("\nQuickPipeline DataFrame Test:")
    print("  Input: DataFrame (3 rows)")
    print("  ✅ DataFrame input working")
