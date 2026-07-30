# Provider Batch API Mode

> **Not to be confused with [Async & Streaming](execution-modes.md).** That guide covers Ondine's own Standard/Async/Streaming *engines* -- how Ondine drives live HTTP calls against a provider. This page covers a different axis entirely: whether Ondine talks to the provider *live* at all, or hands the whole job to the provider's own asynchronous Batch API. You can combine either engine with either mode; they're orthogonal knobs.

Provider Batch mode compiles every prompt into a single JSONL/request payload, submits it to the provider's native Batch API, and collects results later -- instead of one live HTTP call per prompt. In exchange for a slower turnaround, OpenAI and Anthropic both advertise roughly 50% lower per-token pricing on batch jobs.

## At a Glance

| | Live mode (default) | Provider Batch mode |
|---|---|---|
| Providers | Any LiteLLM-supported provider | OpenAI, Anthropic only |
| Turnaround | Seconds to minutes | Up to 24h |
| Cost | Standard per-token pricing | ~50% discount (provider-advertised) |
| Blocking | `execute()` blocks until done | `submit()` returns immediately |
| Crash recovery | Checkpoint + resume | Job ID persisted in RunRegistry -- safe by construction |

## Only OpenAI and Anthropic

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Classify: {text}")
    .with_llm(provider="groq", model="llama-3.3-70b-versatile")
    .with_execution_mode("provider_batch")
    .build()
)
# ValueError: execution_mode='provider_batch' is only supported for
# ['anthropic', 'openai'] in v1, but the configured provider is 'groq'.
# Either switch to an OpenAI/Anthropic provider or use execution_mode='live'.
```

The guard fires at `.build()` time, not mid-run -- a misconfigured pipeline fails immediately with a clear message instead of crashing deep in the JSONL submit path hours later.

## Submit / Poll / Collect

Provider Batch is job-based, not call-based, so it has its own lifecycle instead of `execute()`:

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Classify: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_execution_mode("provider_batch")
    .build()
)

# 1. Submit -- non-blocking, returns as soon as the provider acknowledges the job.
handle = pipeline.submit()
print(handle.run_id, handle.provider_job_id, handle.status)
# <uuid> <provider-job-id> RunStatus.SUBMITTED_REMOTE

# 2. Poll from anywhere -- another process, another day.
from ondine.api.pipeline import Pipeline

handle = Pipeline.attach(handle.run_id)
print(handle.status)  # moves RUNNING -> SUBMITTED_REMOTE -> SUCCEEDED / FAILED / PARTIAL
```

The same lifecycle is available from the CLI:

```bash
ondine submit --config pipeline.yaml --input data.csv --output out.csv
# run_id: ...  provider_job_id: ...  status: submitted_remote
# Poll with: ondine status <run_id>

ondine status <run_id>
ondine collect <run_id> --output out.csv
```

## Crash Safety

A crash between `submit()` and collecting results is safe: the provider's `provider_job_id` is persisted on the `RunHandle` in the `RunRegistry` the moment the job is acknowledged, not after the fact. Restart the process, `Pipeline.attach(run_id)`, and the job is still there -- the provider is running it independently of whether your process is alive.

## The RunRegistry

`ondine.orchestration.run_registry.RunRegistry` is the durable, crash-safe index behind both Provider Batch jobs and the [MCP server](mcp-server.md)'s `ondine_run`. It is:

- **SQLite-backed**, in WAL mode, living inside your checkpoint directory (`runs.db`, alongside `checkpoint_*.json.gz` and `responses.db`).
- **The single source of truth for run identity and status** across process boundaries -- a second process (a later CLI invocation, an MCP `ondine_status` call) reads the same on-disk row, never a stale in-memory copy.
- **Forward-only.** Runs move `PENDING -> RUNNING -> SUBMITTED_REMOTE -> {SUCCEEDED, FAILED, PARTIAL}`; there's no "un-fail" transition, so the history on disk always reflects what actually happened.

You don't normally touch it directly -- `pipeline.submit()`, `Pipeline.attach()`, and the CLI's `submit`/`status`/`collect` commands all go through it -- but it's importable if you want to list or inspect runs programmatically:

```python
from pathlib import Path
from ondine.orchestration.run_registry import RunRegistry

registry = RunRegistry(Path(".checkpoints"))
for handle in registry.list():
    print(handle.run_id, handle.status.value, handle.provider_job_id)
```

## Related

- [Async & Streaming](execution-modes.md) -- Ondine's own live-call engines (Standard/Async/Streaming); orthogonal to this page
- [MCP Server](mcp-server.md) -- shares the same RunRegistry for `ondine_status` / `ondine_collect`
- [Checkpointing & Recovery](checkpointing.md) -- the checkpoint dir Provider Batch's `runs.db` lives alongside
- [CLI](cli.md) -- `ondine submit` / `status` / `collect` reference
