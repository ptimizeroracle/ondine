"""
Pipeline Composer for multi-column, multi-prompt processing.

This module enables composing multiple single-prompt pipelines to generate
multiple output columns, each with its own independent processing logic.

Example:
    composer = (
        PipelineComposer(input_data="data.xlsx")
        .add_column("similarity", similarity_pipeline)
        .add_column("explanation", explanation_pipeline, depends_on=["similarity"])
        .execute()
    )
"""

from pathlib import Path

import pandas as pd

from ondine.api.pipeline import Pipeline
from ondine.core.models import CostEstimate, ExecutionResult, ProcessingStats
from ondine.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PipelineComposer:
    """
    Composes multiple pipelines to process independent columns.

    Each pipeline processes one output column. Pipelines can depend on
    outputs from previous pipelines, enabling sequential processing.

    Design Philosophy:
    - Keep individual pipelines simple (single responsibility)
    - Compose complex workflows from simple building blocks
    - Make dependencies explicit (no hidden coupling)
    - Fail fast with clear error messages
    """

    def __init__(self, input_data: str | Path | pd.DataFrame):
        """
        Initialize pipeline composer.

        Args:
            input_data: Either a file path (CSV/Excel) or DataFrame

        Design Note:
            We accept both paths and DataFrames to support different workflows:
            - Path: Lazy loading, memory efficient
            - DataFrame: Already loaded, faster iteration
        """
        if isinstance(input_data, str | Path):
            self.input_path = str(input_data)
            self.input_df = None  # Lazy load
        elif isinstance(input_data, pd.DataFrame):
            self.input_path = None
            self.input_df = input_data.copy()
        else:
            raise TypeError(
                f"input_data must be str, Path, or DataFrame, got {type(input_data)}"
            )

        # Storage for column pipelines
        # Format: (column_name, pipeline, dependencies)
        self.column_pipelines: list[tuple[str, Pipeline, list[str]]] = []

        logger.info("PipelineComposer initialized")

    def add_column(
        self,
        column_name: str,
        pipeline: Pipeline,
        depends_on: list[str] | None = None,
    ) -> "PipelineComposer":
        """
        Add a pipeline for processing one output column.

        Args:
            column_name: Name of output column
            pipeline: Pipeline to generate this column
            depends_on: List of columns this depends on (optional)

        Returns:
            Self for method chaining (fluent API)

        Example:
            composer.add_column("score", score_pipeline)
            composer.add_column("explanation", explain_pipeline, depends_on=["score"])

        Design Note:
            Fluent API (returns self) enables readable chaining:
            composer.add_column("a", p1).add_column("b", p2).execute()
        """
        dependencies = depends_on or []

        # Validate column name unique
        existing_cols = [col for col, _, _ in self.column_pipelines]
        if column_name in existing_cols:
            raise ValueError(f"Column '{column_name}' already added")

        self.column_pipelines.append((column_name, pipeline, dependencies))

        logger.info(
            f"Added column '{column_name}' with dependencies: {dependencies or 'none'}"
        )

        return self  # Enable chaining

    def _get_execution_order(self) -> list[tuple[str, Pipeline, list[str]]]:
        """
        Resolve execution order using topological sort.

        Returns:
            List of (column_name, pipeline, dependencies) in execution order

        Raises:
            ValueError: If circular dependencies or missing dependencies detected

        Algorithm:
            Kahn's algorithm for topological sorting:
            1. Find nodes with no incoming edges
            2. Remove node and its edges
            3. Repeat until graph is empty
            4. If nodes remain, there's a cycle

        Design Note:
            We use Kahn's algorithm (not DFS) because:
            - Easier to understand
            - Detects cycles naturally
            - Stable ordering (preserves input order when possible)
        """
        # Build dependency graph
        nodes = {col: deps for col, _, deps in self.column_pipelines}
        all_columns = set(nodes.keys())

        # Validate: All dependencies exist
        for col, deps in nodes.items():
            missing = set(deps) - all_columns
            if missing:
                raise ValueError(f"Column '{col}' has missing dependencies: {missing}")

        # Kahn's algorithm
        in_degree = {col: len(deps) for col, deps in nodes.items()}
        queue = [col for col, deg in in_degree.items() if deg == 0]
        result = []
        processed = set()  # Track processed nodes

        while queue:
            # Process node with no dependencies
            col = queue.pop(0)
            result.append(col)
            processed.add(col)  # Mark as processed

            # Reduce in-degree for dependent nodes
            for other_col, deps in nodes.items():
                if other_col not in processed and col in deps:
                    in_degree[other_col] -= 1
                    if in_degree[other_col] == 0 and other_col not in queue:
                        queue.append(other_col)

        # Check for cycles
        if len(result) != len(nodes):
            remaining = set(nodes.keys()) - set(result)
            raise ValueError(f"Circular dependency detected among columns: {remaining}")

        # Map back to original tuples
        col_to_tuple = {
            col: item
            for col, item in zip(
                [c for c, _, _ in self.column_pipelines],
                self.column_pipelines,
                strict=False,
            )
        }

        return [col_to_tuple[col] for col in result]

    def execute(self) -> ExecutionResult:
        """
        Execute all column pipelines in dependency order.

        Returns:
            ExecutionResult with all columns merged

        Algorithm:
            1. Load input data (lazy if needed)
            2. Resolve execution order (topological sort)
            3. For each column:
               a. Inject current DataFrame into pipeline
               b. Execute pipeline
               c. Merge result column into DataFrame
            4. Aggregate metrics and return final result

        Design Note:
            Each pipeline operates on the accumulating DataFrame,
            so later pipelines can use earlier outputs as inputs.
        """
        # Lazy load if needed
        if self.input_df is None:
            logger.info(f"Loading input data from {self.input_path}")
            # Use DataReader adapter for consistent file loading
            from ondine.adapters.data_io import create_data_reader
            from ondine.core.specifications import DataSourceType

            # Detect source type from file extension
            if self.input_path.endswith(".xlsx") or self.input_path.endswith(".xls"):
                source_type = DataSourceType.EXCEL
            elif self.input_path.endswith(".csv"):
                source_type = DataSourceType.CSV
            elif self.input_path.endswith(".parquet"):
                source_type = DataSourceType.PARQUET
            else:
                raise ValueError(f"Unsupported file type: {self.input_path}")

            # Create reader and load data
            reader = create_data_reader(
                source_type=source_type, source_path=self.input_path
            )
            self.input_df = reader.read()

        # Start with input data
        df = self.input_df.copy()

        # Resolve execution order
        execution_order = self._get_execution_order()

        logger.info(
            f"Executing {len(execution_order)} column pipelines in order: "
            f"{[col for col, _, _ in execution_order]}"
        )

        # Track metrics (keep Decimal precision for costs)
        from decimal import Decimal

        total_cost = Decimal("0.0")
        total_errors = []

        # Execute each column pipeline
        for col_name, pipeline, _deps in execution_order:
            logger.info(f"Processing column '{col_name}'...")

            # Inject current DataFrame (includes previous outputs)
            pipeline.dataframe = df

            # Execute pipeline
            result = pipeline.execute()

            # Merge new column
            if col_name in result.data.columns:
                df[col_name] = result.data[col_name]
            else:
                logger.warning(
                    f"Pipeline for '{col_name}' didn't produce expected column"
                )

            # Accumulate metrics (preserve Decimal precision)
            cost_to_add = result.costs.total_cost
            if not isinstance(cost_to_add, Decimal):
                cost_to_add = Decimal(str(cost_to_add))
            total_cost += cost_to_add
            total_errors.extend(result.errors)

            logger.info(
                f"Column '{col_name}' complete: "
                f"{len(df)} rows, ${result.costs.total_cost:.4f}"
            )

        # Create final result
        final_result = ExecutionResult(
            data=df,
            metrics=ProcessingStats(
                total_rows=len(df),
                processed_rows=len(df),
                failed_rows=len(total_errors),
                skipped_rows=0,
                rows_per_second=0.0,  # Aggregate metric not meaningful
                total_duration_seconds=0.0,  # Would need timing
            ),
            costs=CostEstimate(
                total_cost=total_cost,
                total_tokens=0,  # Aggregate from individual pipelines
                input_tokens=0,
                output_tokens=0,
                rows=len(df),
            ),
            errors=total_errors,
        )

        logger.info(
            f"Composition complete: {len(execution_order)} columns, "
            f"${total_cost:.4f} total cost"
        )

        return final_result

    @classmethod
    def from_yaml(cls, config_path: str) -> "PipelineComposer":
        """
        Load composer configuration from YAML.

        Args:
            config_path: Path to composition config file

        Returns:
            Configured PipelineComposer

        Example YAML:
            composition:
              input: "data.xlsx"
              pipelines:
                - column: col1
                  config: pipeline1.yaml
                - column: col2
                  depends_on: [col1]
                  config: pipeline2.yaml

        Design Note:
            This enables pure YAML workflows without Python code.
        """
        import yaml

        from ondine.config import ConfigLoader

        with open(config_path) as f:
            config = yaml.safe_load(f)

        composition = config.get("composition", {})
        input_data = composition.get("input")

        if not input_data:
            raise ValueError("composition.input is required")

        composer = cls(input_data=input_data)

        # Load each pipeline config
        for pipeline_config in composition.get("pipelines", []):
            col_name = pipeline_config["column"]
            config_file = pipeline_config["config"]
            depends_on = pipeline_config.get("depends_on", [])

            # Load pipeline from its config
            pipeline_specs = ConfigLoader.from_yaml(config_file)
            pipeline = Pipeline(pipeline_specs)

            composer.add_column(col_name, pipeline, depends_on)

        return composer
