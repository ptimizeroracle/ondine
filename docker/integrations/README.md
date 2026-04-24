# Ondine × Airflow / Prefect Docker Demo

Proves ondine plugs into both Apache Airflow and Prefect for LLM-powered ETL.

## Setup

```bash
cp .env.example .env
# Fill in GROQ_API_KEY (free tier at https://console.groq.com)
```

## Run Airflow

```bash
docker compose --profile airflow up --build
# UI: http://localhost:8080  (user: admin, password printed in logs)
```

Trigger the `ondine_product_enrichment` DAG from the UI.

## Run Prefect (one-shot)

```bash
docker compose --profile prefect up --build
# Runs once, logs results to stdout, writes data/products_enriched.csv
```

## Run Prefect with UI

```bash
docker compose --profile prefect-ui up --build
# UI: http://localhost:4200
```

## What it does

Reads `data/products.csv` (10 products), calls the LLM for each row to extract:
- `category` — Electronics | Fitness | Kitchen | Office | Nutrition
- `sentiment` — positive | neutral | negative

Writes the enriched CSV to `data/products_enriched.csv`.

## Known limitation

`ondine.config.ConfigLoader.from_yaml()` does **not** expand `${ENV_VAR}` in `api_key`. The demo works around this by leaving `api_key` out of `config/ondine_config.yaml` entirely — litellm auto-detects `GROQ_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` from the environment. See [#166](https://github.com/ptimizeroracle/ondine/issues/166).
