"""
E2E test for YAML configuration loading and execution.

Validates that pipelines can be fully configured via YAML files
and execute correctly with all settings applied.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ondine.api.pipeline_composer import PipelineComposer


@pytest.mark.integration
@pytest.mark.parametrize(
    "provider,model,api_key_env",
    [
        ("openai", "gpt-4o-mini", "OPENAI_API_KEY"),
        ("groq", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    ],
)
def test_yaml_config_end_to_end(provider, model, api_key_env):
    """
    Test full pipeline execution from YAML configuration.

    Validates that all YAML settings are correctly applied.
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        pytest.skip(f"{api_key_env} not set")

    # Create test data file
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "test_data.csv"
        df = pd.DataFrame({"input_text": [f"Text {i}" for i in range(10)]})
        df.to_csv(data_file, index=False)

        # Create YAML config
        yaml_config = f"""
pipeline:
  name: yaml_test_pipeline
  description: E2E test for YAML configuration

data:
  source:
    type: csv
    path: {data_file}
  input_columns:
    - input_text
  output_columns:
    - output_text

prompt:
  template: "Summarize in 2 words: {{{{input_text}}}}"
  use_jinja2: true

llm:
  provider: {provider}
  model: {model}
  api_key: {api_key}
  temperature: 0.0

processing:
  batch_size: 10
  processing_batch_size: 5
  concurrency: 2
  rate_limit: 30
  error_policy: skip
  max_retries: 2

output:
  format: dataframe
"""

        yaml_file = Path(tmpdir) / "config.yaml"
        yaml_file.write_text(yaml_config)

        # Load and execute from YAML
        pipeline = PipelineComposer.from_yaml(str(yaml_file))
        result = pipeline.execute()

        # Verify execution
        assert result.success, f"{provider} YAML pipeline failed"
        assert len(result.data) == 10, f"Expected 10 rows, got {len(result.data)}"
        assert "output_text" in result.data.columns

        # Verify settings were applied (check via specs)
        specs = pipeline.specifications
        assert specs.processing.batch_size == 10
        assert specs.processing.processing_batch_size == 5
        assert specs.processing.concurrency == 2
        assert specs.processing.rate_limit == 30

        print(f"\n{provider.upper()} YAML Config Test Results:")
        print(f"  Loaded from: {yaml_file.name}")
        print(f"  Processed: {len(result.data)} rows")
        print(f"  Settings applied: batch_size=10, concurrency=2, rate_limit=30")
        print(f"  ✅ YAML configuration working correctly")


@pytest.mark.integration
def test_yaml_config_with_multi_column():
    """
    Test YAML configuration with multi-column JSON output.

    Validates complex YAML configurations work end-to-end.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "test_data.csv"
        df = pd.DataFrame({"product": ["Apple", "Banana", "Orange"]})
        df.to_csv(data_file, index=False)

        yaml_config = f"""
pipeline:
  name: multi_column_yaml_test

data:
  source:
    type: csv
    path: {data_file}
  input_columns:
    - product
  output_columns:
    - category
    - color

prompt:
  template: |
    Product: {{{{product}}}}
    Return JSON with category and color.
  use_jinja2: true

llm:
  provider: groq
  model: llama-3.3-70b-versatile
  api_key: {api_key}
  temperature: 0.0

processing:
  batch_size: 3
  processing_batch_size: 3

parser:
  type: json
  extract_fields:
    - category
    - color

output:
  format: dataframe
"""

        yaml_file = Path(tmpdir) / "config.yaml"
        yaml_file.write_text(yaml_config)

        pipeline = PipelineComposer.from_yaml(str(yaml_file))
        result = pipeline.execute()

        assert result.success
        assert len(result.data) == 3
        assert "category" in result.data.columns
        assert "color" in result.data.columns

        print(f"\nMulti-Column YAML Test Results:")
        print(f"  Processed: {len(result.data)} rows")
        print(f"  Output columns: {list(result.data.columns)}")
        print(f"  ✅ Multi-column YAML config working")

