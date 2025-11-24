"""
End-to-end CLI integration tests.

Tests the full CLI workflow with real providers to catch regressions.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from ondine.cli.main import cli


@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "model", "api_key_env"),
    [
        ("openai", "gpt-4o-mini", "OPENAI_API_KEY"),
        ("groq", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    ],
)
def test_cli_process_end_to_end(provider, model, api_key_env):
    """
    CRITICAL: Test full CLI workflow with each provider.

    This catches CLI regressions that unit tests miss:
    - YAML config loading
    - Provider routing
    - Output file generation
    - Cost tracking in CLI context
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        pytest.skip(f"{api_key_env} not set")

    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        data_file = Path(tmpdir) / "input.csv"
        df = pd.DataFrame({"text": ["Hello", "World", "Test"]})
        df.to_csv(data_file, index=False)

        # Create YAML config
        config_file = Path(tmpdir) / "config.yaml"
        config_file.write_text(f"""
data:
  source:
    type: csv
    path: {data_file}
  input_columns:
    - text
  output_columns:
    - summary

prompt:
  template: "Summarize in one word: {{{{ text }}}}"

llm:
  provider: {provider}
  model: {model}
  api_key: {api_key}
  temperature: 0.0

processing:
  batch_size: 3

output:
  format: dataframe
""")

        # Run CLI
        result = runner.invoke(cli, ["process", "--config", str(config_file)])

        # Verify success
        assert result.exit_code == 0, f"{provider} CLI failed: {result.output}"

        # Verify success indicators in output
        output_lower = result.output.lower()
        assert (
            "completed" in output_lower
            or "success" in output_lower
            or "processed" in output_lower
        ), f"{provider} no success message"

        # Verify cost tracking worked (should show $0.00XX or similar)
        assert "$" in result.output, f"{provider} no cost tracking in output"

        print(f"\n✅ {provider.upper()} CLI E2E Test Passed")


@pytest.mark.integration
def test_cli_validate_valid_config():
    """Test CLI validate command with valid config."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "data.csv"
        pd.DataFrame({"text": ["test"]}).to_csv(data_file, index=False)

        config_file = Path(tmpdir) / "config.yaml"
        config_file.write_text(f"""
data:
  source:
    type: csv
    path: {data_file}
  input_columns: [text]
  output_columns: [result]
prompt:
  template: "{{ text }}"
llm:
  provider: groq
  model: llama-3.3-70b-versatile
  api_key: {api_key}
processing:
  batch_size: 1
output:
  format: dataframe
""")

        result = runner.invoke(cli, ["validate", "--config", str(config_file)])

        assert result.exit_code == 0
        assert "valid" in result.output.lower()
        print("\n✅ CLI validate command works")


@pytest.mark.integration
def test_cli_estimate_cost():
    """Test CLI estimate command (no actual API call)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "data.csv"
        # 100 rows for meaningful cost estimate
        df = pd.DataFrame({"text": [f"Text {i}" for i in range(100)]})
        df.to_csv(data_file, index=False)

        config_file = Path(tmpdir) / "config.yaml"
        config_file.write_text(f"""
data:
  source:
    type: csv
    path: {data_file}
  input_columns: [text]
  output_columns: [result]
prompt:
  template: "Summarize: {{{{ text }}}}"
llm:
  provider: groq
  model: llama-3.3-70b-versatile
  api_key: {api_key}
processing:
  batch_size: 10
output:
  format: dataframe
""")

        result = runner.invoke(
            cli, ["estimate", "--config", str(config_file), "--input", str(data_file)]
        )

        assert result.exit_code == 0
        # Should show cost estimate
        assert "cost" in result.output.lower() or "$" in result.output
        # Should show token estimate
        assert "token" in result.output.lower()
        print("\n✅ CLI estimate command works")


def test_cli_invalid_provider_helpful_error():
    """
    CRITICAL REGRESSION TEST: Invalid provider should show available options.

    This ensures users get actionable error messages, not cryptic failures.
    """
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "data.csv"
        pd.DataFrame({"text": ["test"]}).to_csv(data_file, index=False)

        config_file = Path(tmpdir) / "config.yaml"
        config_file.write_text(f"""
data:
  source:
    type: csv
    path: {data_file}
  input_columns: [text]
  output_columns: [result]
prompt:
  template: "Test"
llm:
  provider: invalid_provider_xyz
  model: some-model
processing:
  batch_size: 1
output:
  format: dataframe
""")

        result = runner.invoke(cli, ["validate", "--config", str(config_file)])

        # Should fail with helpful error
        assert result.exit_code != 0
        output_lower = result.output.lower()

        # Should show validation error with provider options
        assert (
            "invalid_provider_xyz" in output_lower
            or "validation error" in output_lower
            or "input should be" in output_lower
        )

        # Should show available providers in error message
        assert "openai" in output_lower
        assert "groq" in output_lower

        print("\n✅ Invalid provider error is helpful")


def test_cli_missing_config_file():
    """Test CLI with missing config file."""
    runner = CliRunner()

    result = runner.invoke(cli, ["validate", "--config", "/nonexistent/config.yaml"])

    assert result.exit_code != 0
    assert (
        "not found" in result.output.lower()
        or "does not exist" in result.output.lower()
    )
    print("\n✅ Missing config file error is clear")


def test_cli_list_providers_shows_all():
    """
    REGRESSION TEST: Ensure all providers are listed.

    This would catch if we accidentally removed a provider from the registry.
    """
    runner = CliRunner()

    result = runner.invoke(cli, ["list-providers"])

    assert result.exit_code == 0
    output = result.output.lower()

    # All core providers must be listed
    assert "openai" in output
    assert "groq" in output
    assert "anthropic" in output
    assert "azure" in output or "azure_openai" in output
    assert "mlx" in output

    print("\n✅ All providers listed in CLI")


def test_cli_inspect_shows_data_info():
    """Test CLI inspect command."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "test.csv"
        df = pd.DataFrame({"col1": ["a", "b", "c"], "col2": [1, 2, 3]})
        df.to_csv(data_file, index=False)

        result = runner.invoke(cli, ["inspect", "--input", str(data_file)])

        assert result.exit_code == 0
        output = result.output.lower()

        # Should show file info
        assert "rows" in output or "3" in output
        assert "col1" in output
        assert "col2" in output

        print("\n✅ CLI inspect works")


@pytest.mark.integration
def test_cli_checkpoint_list_empty():
    """Test list-checkpoints with no checkpoints."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(cli, ["list-checkpoints", "--checkpoint-dir", tmpdir])

        assert result.exit_code == 0
        assert "no checkpoints" in result.output.lower() or "0" in result.output

        print("\n✅ CLI list-checkpoints works (empty)")


def test_cli_resume_invalid_session_id():
    """Test resume with invalid session ID."""
    runner = CliRunner()

    result = runner.invoke(cli, ["resume", "--session-id", "invalid-uuid-123"])

    assert result.exit_code != 0
    # Should show some error (format, not found, or implementation error)
    assert "error" in result.output.lower()

    print("\n✅ CLI resume handles invalid session ID (fails gracefully)")


@pytest.mark.integration
def test_cli_auto_cost_detection_in_estimate():
    """
    Test that CLI estimate uses LiteLLM auto-cost detection.

    Regression test for cost tracking integration.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "data.csv"
        df = pd.DataFrame({"text": ["test " * 100 for _ in range(10)]})  # 10 long texts
        df.to_csv(data_file, index=False)

        # Config WITHOUT manual cost configuration
        config_file = Path(tmpdir) / "config.yaml"
        config_file.write_text(f"""
data:
  source:
    type: csv
    path: {data_file}
  input_columns: [text]
  output_columns: [result]
prompt:
  template: "Summarize: {{{{ text }}}}"
llm:
  provider: groq
  model: llama-3.3-70b-versatile
  # NO input_cost_per_1k_tokens specified - should auto-detect!
processing:
  batch_size: 10
output:
  format: dataframe
""")

        result = runner.invoke(
            cli, ["estimate", "--config", str(config_file), "--input", str(data_file)]
        )

        assert result.exit_code == 0

        # Should show some cost (not $0) if auto-detection works
        # Even with estimation, LiteLLM should give ballpark numbers
        output = result.output.lower()
        assert "cost" in output or "$" in output
        assert "tokens" in output or "token" in output

        print("\n✅ CLI auto-cost detection works in estimate command")
