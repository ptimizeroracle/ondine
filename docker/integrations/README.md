# Ondine — Airflow & Prefect Integration Demo

Proves `ondine.integrations.airflow.LLMTransformOperator` and
`ondine.integrations.prefect.llm_transform_task` with a real 10-product ETL pipeline.

## Quick start

```bash
cd docker/integrations

# 1. Set your API key
cp .env.example .env
# edit .env → add GROQ_API_KEY (free at console.groq.com)

# 2. Run one of the three profiles below
```

---

## Prefect — local runner (fastest)

No server. Runs the flow, prints results, exits.

```bash
docker compose --profile prefect up --build
```

Expected output in logs:
```
── Enrichment Results ─────────────────────────────
           name       category sentiment
Wireless Headphones  Electronics  positive
   Trail Running Shoes   Fitness  positive
...
Total rows: 10
Categories: {'Electronics': 3, 'Fitness': 3, ...}
```

---

## Prefect — with UI

```bash
docker compose --profile prefect-ui up --build
# UI → http://localhost:4200
```

---

## Airflow — standalone

Starts webserver + scheduler in one container. Trigger DAG from UI.

```bash
docker compose --profile airflow up --build
```

Wait ~90s for startup, then:

1. Open http://localhost:8080 (admin / admin — printed in logs)
2. Enable DAG `ondine_product_enrichment`
3. Click **Trigger DAG ▶**
4. Watch tasks: `extract_validate` → `llm_enrich` → `load_report`
5. Click `load_report` → **Logs** to see enriched table

Enriched CSV written to `data/products_enriched.csv` (host-mounted).

---

## Switching provider

Edit `config/ondine_config.yaml`:

```yaml
llm:
  provider: "openai"          # or anthropic, groq
  model: "gpt-4o-mini"
```

Add the matching key in `.env` and restart.

---

## File layout

```
docker/integrations/
  Dockerfile.airflow        # extends apache/airflow:2.9.3, installs ondine
  Dockerfile.prefect        # python:3.11-slim + ondine + prefect
  docker-compose.yml        # profiles: airflow | prefect | prefect-ui
  dags/
    ondine_etl_dag.py       # 3-task Airflow DAG
  flows/
    ondine_etl_flow.py      # 3-task Prefect flow
  config/
    ondine_config.yaml      # shared pipeline config (Groq default)
  data/
    products.csv            # 10-row sample input
    products_enriched.csv   # written after run
  .env.example
```
