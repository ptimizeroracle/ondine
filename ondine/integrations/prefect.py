"""
Prefect integration - Pre-built tasks for Prefect workflows.

Provides llm_transform_task for easy integration into Prefect flows.
"""

from pathlib import Path

import pandas as pd

try:
    from prefect import task

    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False

    # Placeholder decorator
    def task(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


from ondine.api import Pipeline
from ondine.config import ConfigLoader


@task(name="llm_transform")
def llm_transform_task(
    config_path: str,
    input_data: pd.DataFrame | None = None,
    input_file: str | None = None,
    output_file: str | None = None,
    max_budget: float | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> pd.DataFrame:
    """
    Prefect task for LLM dataset transformations.

    Integrates LLM Dataset Engine into Prefect flows.

    Example:
        from prefect import flow
        from ondine.integrations.prefect import llm_transform_task

        @flow
        def data_pipeline():
            raw_data = load_data()
            enriched = llm_transform_task(
                config_path='configs/llm_config.yaml',
                input_data=raw_data,
                max_budget=10.0,
            )
            save_data(enriched)

    Args:
        config_path: Path to YAML/JSON configuration
        input_data: Input DataFrame (from previous task)
        input_file: Path to input file (alternative to input_data)
        output_file: Path to output file (optional)
        max_budget: Override maximum budget
        provider_override: Override LLM provider
        model_override: Override model name

    Returns:
        Result DataFrame

    Raises:
        ValueError: If neither input_data nor input_file provided
    """
    if not PREFECT_AVAILABLE:
        raise ImportError(
            "Prefect is required to use llm_transform_task. "
            "Install with: pip install prefect"
        )

    # Load configuration
    specs = ConfigLoader.from_yaml(config_path)

    # Override settings
    if max_budget is not None:
        from decimal import Decimal

        specs.processing.max_budget = Decimal(str(max_budget))

    if provider_override:
        from ondine.core.specifications import LLMProvider

        specs.llm.provider = LLMProvider(provider_override)

    if model_override:
        specs.llm.model = model_override

    # Get input
    if input_data is not None:
        pipeline = Pipeline(specs, dataframe=input_data)
    elif input_file:
        specs.dataset.source_path = Path(input_file)
        pipeline = Pipeline(specs)
    else:
        raise ValueError("Either input_data or input_file required")

    # Set output if specified
    if output_file:
        from ondine.core.specifications import (
            DataSourceType,
            MergeStrategy,
            OutputSpec,
        )

        specs.output = OutputSpec(
            destination_type=DataSourceType.CSV,
            destination_path=Path(output_file),
            merge_strategy=MergeStrategy.REPLACE,
        )

    # Execute
    result = pipeline.execute()

    # Log metrics (Prefect will capture)
    print(f"✅ Processed {result.metrics.total_rows} rows")
    print(f"💰 Cost: ${result.costs.total_cost}")
    print(f"⏱️  Duration: {result.duration:.2f}s")

    return result.data


__all__ = ["llm_transform_task"]
