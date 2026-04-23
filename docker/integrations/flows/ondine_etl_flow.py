"""
Ondine + Prefect ETL demo flow.

Three-task pipeline:
  extract_validate → llm_enrich → load_report
"""

from __future__ import annotations

import pandas as pd
from prefect import flow, get_run_logger, task

from ondine.integrations.prefect import llm_transform_task

DATA_DIR = "/data"
CONFIG_PATH = "/config/ondine_config.yaml"
INPUT_FILE = f"{DATA_DIR}/products.csv"
OUTPUT_FILE = f"{DATA_DIR}/products_enriched.csv"


@task(name="extract-validate")
def extract_validate(input_file: str) -> dict:
    """Verify input data exists and is readable."""
    logger = get_run_logger()
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} rows — columns: {list(df.columns)}")
    if len(df) == 0:
        raise ValueError("Empty input dataset")
    missing = {"name", "description"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return {"rows": len(df), "columns": list(df.columns)}


@task(name="load-report")
def load_report(output_file: str) -> dict:
    """Read enriched output, print summary, return stats."""
    logger = get_run_logger()
    df = pd.read_csv(output_file)
    logger.info("\n── Enrichment Results ─────────────────────────────")
    logger.info(f"\n{df[['name', 'category', 'sentiment']].to_string(index=False)}")
    logger.info(f"\nTotal rows: {len(df)}")
    logger.info(f"Categories: {df['category'].value_counts().to_dict()}")
    logger.info(f"Sentiments: {df['sentiment'].value_counts().to_dict()}")
    return {
        "rows_enriched": len(df),
        "categories": df["category"].value_counts().to_dict(),
        "sentiments": df["sentiment"].value_counts().to_dict(),
    }


@flow(
    name="ondine-product-enrichment",
    description="Enrich product catalog with LLM-generated category + sentiment via Ondine",
    log_prints=True,
)
def product_enrichment_flow(
    config_path: str = CONFIG_PATH,
    input_file: str = INPUT_FILE,
    output_file: str = OUTPUT_FILE,
) -> dict:
    extract_validate(input_file)
    llm_transform_task(
        config_path=config_path,
        input_file=input_file,
        output_file=output_file,
    )
    return load_report(output_file)


if __name__ == "__main__":
    product_enrichment_flow()
