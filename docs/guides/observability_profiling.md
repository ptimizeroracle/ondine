# Ondine Observability & Profiling

This guide explains how to profile your Ondine pipelines and trace execution using OpenTelemetry and Langfuse (Self-Hosted).

## Quick Start (Docker)

We provide a fully self-hosted observability stack using Docker. No data leaves your machine.

1.  **Start the Stack** (Jaeger + Langfuse + Python App):
    ```bash
    docker compose -f docker/docker-compose.yml up -d
    ```

2.  **Configure Langfuse (One-time Setup)**:
    *   Go to [http://localhost:3000](http://localhost:3000).
    *   Create an account (email/password).
    *   Create a new project.
    *   Go to **Settings -> API Keys** and create new keys.
    *   Copy `Public Key` and `Secret Key`.
    *   Add them to your `.env` file:
        ```bash
        LANGFUSE_PUBLIC_KEY="pk-lf-..."
        LANGFUSE_SECRET_KEY="sk-lf-..."
        LANGFUSE_HOST="http://localhost:3000"
        ```

3.  **Run Profiling**:
    ```bash
    ./docker/profile.sh path/to/your_script.py
    ```

4.  **View Results**:
    *   **Performance**: Open `profile_your_script.html` in your browser.
    *   **Infrastructure Traces**: [http://localhost:16686](http://localhost:16686) (Jaeger).
    *   **LLM Analytics**: [http://localhost:3000](http://localhost:3000) (Langfuse).

## Configuration

Enable observability in your pipeline with `.with_observer()`:

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

- **Langfuse not loading**: It takes ~30s to start the first time. Check logs: `docker compose -f docker/docker-compose.yml logs -f langfuse-server`.
- **Database Connection Errors**: If you see connection refused errors for Clickhouse or Postgres, try restarting the stack: `docker compose -f docker/docker-compose.yml restart`.
- **Missing Traces**: Ensure your `.env` file is present in the project root when running the profile script.
- **Docker Version**: We pin Langfuse to version 2 (`langfuse/langfuse:2`) to ensure stability with the local simplified Clickhouse setup.
