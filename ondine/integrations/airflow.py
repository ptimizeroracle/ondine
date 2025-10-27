"""
Airflow integration - Pre-built operators for Apache Airflow.

Provides LLMTransformOperator for easy integration into Airflow DAGs.
"""

from typing import Any

try:
    from airflow.models import BaseOperator
    from airflow.utils.decorators import apply_defaults

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    BaseOperator = object  # Placeholder

from ondine.api import Pipeline
from ondine.config import ConfigLoader

if AIRFLOW_AVAILABLE:

    class LLMTransformOperator(BaseOperator):
        """
        Airflow operator for LLM dataset transformations.

        Integrates LLM Dataset Engine into Airflow DAGs with minimal boilerplate.

        Example:
            from ondine.integrations.airflow import LLMTransformOperator

            llm_task = LLMTransformOperator(
                task_id='llm_enrichment',
                config_path='configs/llm_config.yaml',
                input_xcom_key='raw_data',
                output_xcom_key='enriched_data',
                max_budget=10.0,
                dag=dag,
            )
        """

        @apply_defaults
        def __init__(
            self,
            config_path: str,
            input_xcom_key: str | None = None,
            output_xcom_key: str = "llm_result",
            input_file: str | None = None,
            output_file: str | None = None,
            max_budget: float | None = None,
            provider_override: str | None = None,
            model_override: str | None = None,
            *args,
            **kwargs,
        ):
            """
            Initialize LLM transform operator.

            Args:
                config_path: Path to YAML/JSON configuration
                input_xcom_key: XCom key to pull DataFrame from previous task
                output_xcom_key: XCom key to push result DataFrame
                input_file: Path to input file (alternative to XCom)
                output_file: Path to output file (alternative to XCom)
                max_budget: Override maximum budget
                provider_override: Override LLM provider
                model_override: Override model name
                *args: Airflow BaseOperator args
                **kwargs: Airflow BaseOperator kwargs
            """
            super().__init__(*args, **kwargs)
            self.config_path = config_path
            self.input_xcom_key = input_xcom_key
            self.output_xcom_key = output_xcom_key
            self.input_file = input_file
            self.output_file = output_file
            self.max_budget = max_budget
            self.provider_override = provider_override
            self.model_override = model_override

        def execute(self, context: dict[str, Any]) -> Any:
            """
            Execute LLM transformation.

            Args:
                context: Airflow task context

            Returns:
                Result DataFrame (pushed to XCom)
            """
            # Load configuration
            specs = ConfigLoader.from_yaml(self.config_path)

            # Override settings
            if self.max_budget is not None:
                from decimal import Decimal

                specs.processing.max_budget = Decimal(str(self.max_budget))

            if self.provider_override:
                from ondine.core.specifications import LLMProvider

                specs.llm.provider = LLMProvider(self.provider_override)

            if self.model_override:
                specs.llm.model = self.model_override

            # Get input data
            if self.input_xcom_key:
                # Pull from XCom
                df = context["ti"].xcom_pull(key=self.input_xcom_key)
                if df is None:
                    raise ValueError(
                        f"No data found in XCom key: {self.input_xcom_key}"
                    )
                pipeline = Pipeline(specs, dataframe=df)
            elif self.input_file:
                # Read from file
                specs.dataset.source_path = self.input_file
                pipeline = Pipeline(specs)
            else:
                raise ValueError("Either input_xcom_key or input_file required")

            # Set output if specified
            if self.output_file:
                from pathlib import Path

                from ondine.core.specifications import (
                    DataSourceType,
                    MergeStrategy,
                    OutputSpec,
                )

                specs.output = OutputSpec(
                    destination_type=DataSourceType.CSV,
                    destination_path=Path(self.output_file),
                    merge_strategy=MergeStrategy.REPLACE,
                )

            # Execute pipeline
            result = pipeline.execute()

            # Log metrics
            self.log.info(f"Processed {result.metrics.total_rows} rows")
            self.log.info(f"Cost: ${result.costs.total_cost}")
            self.log.info(f"Duration: {result.duration:.2f}s")

            # Push result to XCom
            if self.output_xcom_key:
                context["ti"].xcom_push(key=self.output_xcom_key, value=result.data)

            return result.data

else:
    # Airflow not installed
    class LLMTransformOperator:
        """Placeholder when Airflow not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Apache Airflow is required to use LLMTransformOperator. "
                "Install with: pip install apache-airflow"
            )


__all__ = ["LLMTransformOperator"]
