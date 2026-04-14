# Observability & Profiling

Profile your Ondine pipelines and trace execution with OpenTelemetry and self-hosted Langfuse. The entire stack runs locally; no data leaves your machine.

## Quick Start (Docker)

Start the stack (Jaeger + Langfuse + Python app):

```bash
docker compose -f docker/docker-compose.yml up -d
```

Then configure Langfuse (one-time). Open [http://localhost:3000](http://localhost:3000), create an account, create a project, and grab your API keys from **Settings > API Keys**. Drop them into `.env`:

```bash
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_HOST="http://localhost:3000"
```

Run profiling against any script:

```bash
./docker/profile.sh path/to/your_script.py
```

View results in three places:

- **Performance profile**: Open `profile_your_script.html` in your browser.
- **Infrastructure traces**: [http://localhost:16686](http://localhost:16686) (Jaeger).
- **LLM analytics**: [http://localhost:3000](http://localhost:3000) (Langfuse).

## Configuration

Wire observability into your pipeline with `.with_observer()`:

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    # ... steps ...
    # 1. OpenTelemetry (Infrastructure/Tracing)
    .with_observer("opentelemetry")
    
    # 2. Langfuse (LLM Analytics)
    .with_observer("langfuse", config={
        "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
        "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
        "host": os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    })
    .build()
)
```

### Environment Variables

| Variable | Description | Default (Local) |
| :--- | :--- | :--- |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Jaeger Endpoint | `http://localhost:4318` |
| `LANGFUSE_HOST` | Langfuse Server | `http://localhost:3000` |
| `LANGFUSE_PUBLIC_KEY` | Project Public Key | (Get from UI) |
| `LANGFUSE_SECRET_KEY` | Project Secret Key | (Get from UI) |

## Troubleshooting

- **Langfuse not loading**: First boot takes about 30 seconds. Check logs with `docker compose -f docker/docker-compose.yml logs -f langfuse-server`.
- **Database connection errors**: Connection refused from Clickhouse or Postgres usually clears after a restart: `docker compose -f docker/docker-compose.yml restart`.
- **Missing traces**: Make sure your `.env` file sits in the project root when running the profile script.
- **Docker version**: Langfuse is pinned to v2 (`langfuse/langfuse:2`) for stability with the local Clickhouse setup.
