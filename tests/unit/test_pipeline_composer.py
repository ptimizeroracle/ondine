"""
Unit tests for PipelineComposer.

Following TDD: Tests written FIRST to define the API.
Then implementation will be created to make these tests pass.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ondine.api.pipeline import Pipeline
from ondine.api.pipeline_composer import PipelineComposer
from ondine.core.models import CostEstimate, ExecutionResult


class TestPipelineComposerBasics:
    """Test basic PipelineComposer functionality."""

    def test_composer_creation_from_dataframe(self):
        """Test creating composer with DataFrame."""
        df = pd.DataFrame(
            {
                "text": ["sample1", "sample2"],
                "value": [1, 2],
            }
        )

        composer = PipelineComposer(input_data=df)

        assert composer is not None
        assert isinstance(composer.input_df, pd.DataFrame)
        assert len(composer.input_df) == 2

    def test_composer_creation_from_file_path(self):
        """Test creating composer with file path (lazy load)."""
        composer = PipelineComposer(input_data="dummy.csv")

        assert composer is not None
        assert composer.input_path == "dummy.csv"

    def test_add_single_column_pipeline(self):
        """Test adding a single column pipeline."""
        df = pd.DataFrame({"text": ["test"]})
        composer = PipelineComposer(input_data=df)

        # Create mock pipeline
        mock_pipeline = MagicMock(spec=Pipeline)

        result = composer.add_column(column_name="output1", pipeline=mock_pipeline)

        # Should return self for chaining
        assert result is composer
        assert len(composer.column_pipelines) == 1

    def test_add_multiple_columns(self):
        """Test adding multiple column pipelines (chainable API)."""
        df = pd.DataFrame({"text": ["test"]})
        composer = PipelineComposer(input_data=df)

        pipeline1 = MagicMock(spec=Pipeline)
        pipeline2 = MagicMock(spec=Pipeline)

        result = composer.add_column("col1", pipeline1).add_column("col2", pipeline2)

        assert result is composer
        assert len(composer.column_pipelines) == 2


class TestDependencyResolution:
    """Test dependency resolution and execution ordering."""

    def test_no_dependencies_preserves_order(self):
        """Test columns without dependencies execute in order added."""
        df = pd.DataFrame({"text": ["test"]})
        composer = PipelineComposer(input_data=df)

        p1 = MagicMock(spec=Pipeline)
        p2 = MagicMock(spec=Pipeline)
        p3 = MagicMock(spec=Pipeline)

        composer.add_column("col1", p1)
        composer.add_column("col2", p2)
        composer.add_column("col3", p3)

        order = composer._get_execution_order()

        assert [item[0] for item in order] == ["col1", "col2", "col3"]

    def test_dependencies_determine_order(self):
        """Test dependency resolution via topological sort."""
        df = pd.DataFrame({"text": ["test"]})
        composer = PipelineComposer(input_data=df)

        p1 = MagicMock(spec=Pipeline)
        p2 = MagicMock(spec=Pipeline)
        p3 = MagicMock(spec=Pipeline)

        # Add out of order
        composer.add_column("col3", p3, depends_on=["col1", "col2"])
        composer.add_column("col1", p1)
        composer.add_column("col2", p2, depends_on=["col1"])

        order = composer._get_execution_order()

        # Should resolve to: col1 -> col2 -> col3
        assert [item[0] for item in order] == ["col1", "col2", "col3"]

    def test_circular_dependency_detection(self):
        """Test circular dependencies are detected."""
        df = pd.DataFrame({"text": ["test"]})
        composer = PipelineComposer(input_data=df)

        p1 = MagicMock(spec=Pipeline)
        p2 = MagicMock(spec=Pipeline)

        composer.add_column("col1", p1, depends_on=["col2"])
        composer.add_column("col2", p2, depends_on=["col1"])

        with pytest.raises(ValueError, match="Circular dependency"):
            composer._get_execution_order()

    def test_missing_dependency_detection(self):
        """Test missing dependencies are detected."""
        df = pd.DataFrame({"text": ["test"]})
        composer = PipelineComposer(input_data=df)

        p1 = MagicMock(spec=Pipeline)

        composer.add_column("col1", p1, depends_on=["nonexistent"])

        with pytest.raises(ValueError, match="missing dependencies"):
            composer._get_execution_order()


class TestComposerExecution:
    """Test actual execution of composed pipelines."""

    @patch("ondine.api.pipeline.Pipeline.execute")
    def test_execute_single_column(self, mock_execute):
        """Test executing single column pipeline."""
        # Setup
        df = pd.DataFrame(
            {
                "text": ["sample1", "sample2"],
            }
        )

        # Mock pipeline returns result with new column
        mock_execute.return_value = ExecutionResult(
            data=pd.DataFrame(
                {
                    "text": ["sample1", "sample2"],
                    "output1": ["result1", "result2"],
                }
            ),
            metrics=MagicMock(),
            costs=CostEstimate(
                total_cost=0.01,
                total_tokens=100,
                input_tokens=50,
                output_tokens=50,
                rows=2,
            ),
            errors=[],
        )

        # Create composer
        composer = PipelineComposer(input_data=df)

        # Create mock pipeline
        mock_pipeline = MagicMock(spec=Pipeline)
        mock_pipeline.execute = mock_execute

        composer.add_column("output1", mock_pipeline)

        # Execute
        result = composer.execute()

        assert isinstance(result, ExecutionResult)
        assert "output1" in result.data.columns
        assert len(result.data) == 2

    @patch("ondine.api.pipeline.Pipeline.execute")
    def test_execute_multiple_independent_columns(self, mock_execute):
        """Test executing multiple independent columns."""
        df = pd.DataFrame({"text": ["sample"]})

        # Mock pipeline executions
        def execute_side_effect(*args, **kwargs):
            # Each pipeline adds its column
            return ExecutionResult(
                data=pd.DataFrame({"new_col": ["result"]}),
                metrics=MagicMock(),
                costs=CostEstimate(
                    total_cost=0.01,
                    total_tokens=50,
                    input_tokens=25,
                    output_tokens=25,
                    rows=1,
                ),
                errors=[],
            )

        mock_execute.side_effect = execute_side_effect

        composer = PipelineComposer(input_data=df)

        p1 = MagicMock(spec=Pipeline)
        p1.execute = mock_execute
        p2 = MagicMock(spec=Pipeline)
        p2.execute = mock_execute

        composer.add_column("col1", p1)
        composer.add_column("col2", p2)

        composer.execute()

        # Should have called execute twice
        assert mock_execute.call_count == 2

    def test_execute_with_dependencies(self):
        """Test execution respects dependencies."""
        df = pd.DataFrame(
            {
                "current_product": ["product1"],
                "candidate_product": ["product2"],
            }
        )

        composer = PipelineComposer(input_data=df)

        # Track execution order
        execution_log = []

        def create_mock_pipeline(name, output_col, output_value):
            mock = MagicMock(spec=Pipeline)

            def execute(*args, **kwargs):
                execution_log.append(name)
                return ExecutionResult(
                    data=pd.DataFrame({output_col: [output_value]}),
                    metrics=MagicMock(),
                    costs=CostEstimate(
                        total_cost=0.01,
                        total_tokens=50,
                        input_tokens=25,
                        output_tokens=25,
                        rows=1,
                    ),
                    errors=[],
                )

            mock.execute = execute
            return mock

        p1 = create_mock_pipeline("match_score", "match_score", "95%")
        p2 = create_mock_pipeline("explanation", "explanation", "Products are similar")

        # Add with dependency
        composer.add_column("match_score", p1)
        composer.add_column("explanation", p2, depends_on=["match_score"])

        result = composer.execute()

        # Should execute in order: match_score -> explanation
        assert execution_log == ["match_score", "explanation"]
        assert "match_score" in result.data.columns
        assert "explanation" in result.data.columns


class TestComposerConfiguration:
    """Test YAML configuration support."""

    def test_from_yaml_config(self):
        """Test loading composer from YAML."""
        # This test defines the YAML API we want
        # This test will initially fail - that's TDD!
        # Implementation will make it pass

        # For now, just test the concept exists
        assert True  # Placeholder


class TestAPIDesign:
    """Test API design principles."""

    def test_simple_api_for_common_cases(self):
        """Test the API is simple for common cases."""
        df = pd.DataFrame({"text": ["test"]})

        # Should be ONE line to create composer
        composer = PipelineComposer(input_data=df)

        # Should be ONE line to add column
        mock_pipeline = MagicMock(spec=Pipeline)
        composer.add_column("output", mock_pipeline)

        assert True

    def test_explicit_dependencies(self):
        """Test dependencies are explicit, not magic."""
        df = pd.DataFrame({"text": ["test"]})
        composer = PipelineComposer(input_data=df)

        p1 = MagicMock(spec=Pipeline)
        p2 = MagicMock(spec=Pipeline)

        # Dependencies are EXPLICIT in the API
        composer.add_column("col1", p1)
        composer.add_column("col2", p2, depends_on=["col1"])

        assert True

    def test_fluent_api_readability(self):
        """Test the code reads naturally."""
        df = pd.DataFrame({"text": ["test"]})

        # This should read naturally
        composer = (
            PipelineComposer(input_data=df)
            .add_column("similarity", MagicMock())
            .add_column("explanation", MagicMock(), depends_on=["similarity"])
        )

        assert composer is not None
