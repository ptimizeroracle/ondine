"""
Ondine + Airflow ETL demo DAG.

Three-task pipeline:
  extract → llm_enrich (LLMTransformOperator) → load_report
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from airflow.decorators import dag, task

from ondine.integrations.airflow import LLMTransformOperator

DATA_DIR = "/opt/airflow/data"
CONFIG_PATH = "/opt/airflow/config/ondine_config.yaml"
INPUT_FILE = f"{DATA_DIR}/products.csv"
OUTPUT_FILE = f"{DATA_DIR}/products_enriched.csv"


@dag(
    dag_id="ondine_product_enrichment",
    description="Enrich product catalog with LLM-generated category + sentiment via Ondine",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=1),
    },
    tags=["ondine", "llm", "etl", "demo"],
)
def ondine_product_enrichment_dag():
    @task(task_id="extract_validate")
    def extract_validate() -> dict:
        """Verify input data exists and is readable."""
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
        if len(df) == 0:
            raise ValueError("Empty input dataset")
        missing = {"name", "description"} - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return {"rows": len(df), "columns": list(df.columns)}

    enrich = LLMTransformOperator(
        task_id="llm_enrich",
        config_path=CONFIG_PATH,
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        # Uncomment to override provider/model at DAG level:
        # provider_override="groq",
        # model_override="llama-3.1-8b-instant",
    )

    @task(task_id="load_report")
    def load_report() -> dict:
        """Read enriched output, print summary, return stats."""
        df = pd.read_csv(OUTPUT_FILE)
        print("\n── Enrichment Results ─────────────────────────────")
        print(df[["name", "category", "sentiment"]].to_string(index=False))
        print(f"\nTotal rows: {len(df)}")
        print(f"Categories: {df['category'].value_counts().to_dict()}")
        print(f"Sentiments: {df['sentiment'].value_counts().to_dict()}")
        print("────────────────────────────────────────────────────\n")
        return {
            "rows_enriched": len(df),
            "categories": df["category"].value_counts().to_dict(),
        }

    extract_validate() >> enrich >> load_report()


ondine_product_enrichment_dag()
