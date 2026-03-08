"""Unit tests for CLI commands."""

import tempfile
from pathlib import Path
from uuid import uuid4

import pandas as pd
from click.testing import CliRunner

from ondine.adapters import LocalFileCheckpointStorage
from ondine.cli.main import cli


class TestCLI:
    """Test suite for CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "LLM Dataset Engine" in result.output
        assert "process" in result.output
        assert "estimate" in result.output

    def test_cli_version(self):
        """Test CLI version command."""
        from ondine import __version__

        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_process_help(self):
        """Test process command help."""
        result = self.runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0
        assert "Process a dataset" in result.output
        assert "--config" in result.output
        assert "--input" in result.output

    def test_estimate_help(self):
        """Test estimate command help."""
        result = self.runner.invoke(cli, ["estimate", "--help"])
        assert result.exit_code == 0
        assert "Estimate processing cost" in result.output

    def test_validate_help(self):
        """Test validate command help."""
        result = self.runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "Validate pipeline configuration" in result.output

    def test_inspect_help(self):
        """Test inspect command help."""
        result = self.runner.invoke(cli, ["inspect", "--help"])
        assert result.exit_code == 0
        assert "Inspect input data file" in result.output

    def test_inspect_csv(self):
        """Test inspect command with CSV file."""
        # Create test CSV
        test_csv = Path(self.temp_dir) / "test.csv"
        df = pd.DataFrame(
            {
                "text": ["Hello", "World"],
                "value": [1, 2],
            }
        )
        df.to_csv(test_csv, index=False)

        # Run inspect
        result = self.runner.invoke(cli, ["inspect", "-i", str(test_csv)])

        assert result.exit_code == 0
        assert "File Information" in result.output
        assert "Total Rows" in result.output
        assert "2" in result.output
        assert "text" in result.output
        assert "value" in result.output

    def test_validate_missing_config(self):
        """Test validate with missing config file."""
        result = self.runner.invoke(cli, ["validate", "-c", "nonexistent.yaml"])
        assert result.exit_code != 0

    def test_process_missing_required_args(self):
        """Test process without required arguments."""
        result = self.runner.invoke(cli, ["process"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_estimate_missing_required_args(self):
        """Test estimate without required arguments."""
        result = self.runner.invoke(cli, ["estimate"])
        assert result.exit_code != 0

    def test_list_checkpoints_empty(self):
        """Test list-checkpoints with no checkpoints."""
        empty_dir = Path(self.temp_dir) / "empty_checkpoints"
        empty_dir.mkdir()

        result = self.runner.invoke(
            cli, ["list-checkpoints", "--checkpoint-dir", str(empty_dir)]
        )

        assert result.exit_code == 0
        assert "No checkpoints found" in result.output

    def test_resume_invalid_session_id(self):
        """Test resume with invalid session ID."""
        result = self.runner.invoke(cli, ["resume", "-s", "invalid-uuid"])

        assert result.exit_code != 0
        assert "Invalid session ID" in result.output or "Error" in result.output

    def test_resume_requires_config_for_existing_checkpoint(self):
        """Test resume reports missing config for real checkpoints."""
        checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        checkpoint_dir.mkdir()
        session_id = uuid4()
        storage = LocalFileCheckpointStorage(checkpoint_dir)
        storage.save(
            session_id,
            {
                "session_id": str(session_id),
                "pipeline_id": str(uuid4()),
                "start_time": "2026-03-08T00:00:00",
                "end_time": None,
                "current_stage_index": 2,
                "last_processed_row": 1,
                "total_rows": 3,
                "accumulated_cost": "0.20",
                "accumulated_tokens": 10,
                "intermediate_data": {},
                "failed_rows": 0,
                "skipped_rows": 0,
            },
        )

        result = self.runner.invoke(
            cli,
            ["resume", "-s", str(session_id), "--checkpoint-dir", str(checkpoint_dir)],
        )

        assert result.exit_code != 0
        assert "requires the original pipeline config" in result.output.lower()

    def test_list_checkpoints_shows_current_checkpoint_fields(self):
        """Test list-checkpoints renders current CheckpointInfo fields."""
        checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        checkpoint_dir.mkdir()

        session_id = uuid4()
        storage = LocalFileCheckpointStorage(checkpoint_dir)
        storage.save(
            session_id,
            {
                "session_id": str(session_id),
                "pipeline_id": str(uuid4()),
                "start_time": "2026-03-07T00:00:00",
                "end_time": None,
                "current_stage_index": 3,
                "last_processed_row": 4,
                "total_rows": 10,
                "accumulated_cost": "1.23",
                "accumulated_tokens": 100,
                "intermediate_data": {},
                "failed_rows": 0,
                "skipped_rows": 0,
            },
        )

        result = self.runner.invoke(
            cli, ["list-checkpoints", "--checkpoint-dir", str(checkpoint_dir)]
        )

        assert result.exit_code == 0
        assert str(session_id)[:8] in result.output
        assert "4" in result.output
        assert "10" in result.output
        assert "1.23" in result.output
        assert "3" in result.output
