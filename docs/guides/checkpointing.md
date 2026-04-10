# Checkpointing

Checkpointing saves execution state to disk as the pipeline runs. If a job is interrupted — by a crash, a network failure, or a keyboard interrupt — it can be resumed from the last saved position instead of restarting from row zero.

## Overview

| Feature | Default | Notes |
|---------|---------|-------|
| Checkpoint directory | `.checkpoints/` | Configurable via `with_checkpoint_dir()` |
| Checkpoint interval | Every 500 rows | Configurable via `with_checkpoint_interval()` |
| Cleanup on success | Enabled | Configurable via `with_checkpoint_cleanup()` |
| Storage format | Gzip-compressed JSON | Human-readable, compact |

## How It Works

When a pipeline runs, the `StateManager` periodically serialises the execution context — including the last processed row index, accumulated cost, and completed responses — into a compressed JSON file under the checkpoint directory. Each file is named after the session UUID:

```
.checkpoints/checkpoint_<session-uuid>.json.gz
```

If execution fails, the pipeline logs the session UUID and a ready-to-paste resume call:

```
Pipeline failed. Checkpoint saved.
Resume with: pipeline.execute(resume_from=UUID('e650ee2a-0c71-4761-ac3f-bdab8ecd920b'))
```

On resume, the pipeline skips all rows already processed and continues from where it left off, at zero additional LLM cost for completed rows.

## Builder Methods

### `with_checkpoint_dir(directory: str)`

Set the directory where checkpoint files are written. The directory is created automatically if it does not exist.

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Summarise: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_checkpoint_dir("/var/checkpoints/my-job")
    .build()
)
```

**Default:** `.checkpoints` (relative to the working directory).

### `with_checkpoint_interval(rows: int)`

Control how often a checkpoint is written. Lower values mean less re-work on failure, at the cost of slightly more disk I/O.

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Analyse: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_checkpoint_interval(100)   # Checkpoint every 100 rows
    .build()
)
```

**Default:** `500` rows.

**Guidelines:**

| Dataset Size | Recommended Interval |
|-------------|---------------------|
| < 5K rows | 100 |
| 5K–50K rows | 500 |
| 50K–500K rows | 1,000–2,000 |
| 500K+ rows | 5,000 |

### `with_checkpoint_cleanup(enabled: bool = True)`

Control whether checkpoint files are deleted after a successful run.

- `True` (default) — checkpoints are deleted once the pipeline returns successfully, keeping the directory clean.
- `False` — checkpoints are retained even after success. Use this when downstream code (e.g. writing to a database) might fail after the pipeline completes, so you can resume without re-running LLM calls.

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Classify: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_checkpoint_cleanup(False)   # Keep checkpoint as safety net
    .build()
)
```

## Resuming a Failed Pipeline

When execution fails or is interrupted, the pipeline prints a resume instruction to the log. Copy the UUID from that message and pass it to `execute()`:

```python
from uuid import UUID
from ondine import PipelineBuilder

# Original pipeline definition — must be identical to the failed run
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Summarise: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_checkpoint_dir(".checkpoints")
    .build()
)

# Resume from the checkpoint saved by the interrupted run
result = pipeline.execute(resume_from=UUID("e650ee2a-0c71-4761-ac3f-bdab8ecd920b"))
print(f"Resumed. Processed {result.metrics.processed_rows} rows total.")
```

The `resume_from` parameter is also available on the async entry point:

```python
result = await pipeline.execute_async(
    resume_from=UUID("e650ee2a-0c71-4761-ac3f-bdab8ecd920b")
)
```

## Practical Patterns

### Overnight Batch Job

A long-running job should checkpoint frequently and keep checkpoints after success in case the downstream write fails:

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv(
        "500k_records.csv",
        input_columns=["description"],
        output_columns=["category", "tags"],
    )
    .with_prompt("Classify this product description: {description}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_checkpoint_dir("/data/checkpoints/product-classification")
    .with_checkpoint_interval(500)
    .with_checkpoint_cleanup(False)   # Keep until DB write confirmed
    .with_max_budget(50.0)
    .build()
)

result = pipeline.execute()
write_to_database(result.to_pandas())   # If this fails, you can resume
```

### Robust Resume Script

Wrap the execute call to capture the session ID on failure and re-run automatically:

```python
import logging
from uuid import UUID
from ondine import PipelineBuilder

log = logging.getLogger(__name__)

def build_pipeline():
    return (
        PipelineBuilder.create()
        .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
        .with_prompt("Process: {text}")
        .with_llm(provider="openai", model="gpt-4o-mini")
        .with_checkpoint_dir(".checkpoints")
        .with_checkpoint_interval(250)
        .build()
    )

def run(resume_from: UUID | None = None):
    pipeline = build_pipeline()
    try:
        return pipeline.execute(resume_from=resume_from)
    except Exception as e:
        # The pipeline logs the session UUID automatically.
        # Extract it from result.execution_id if you need it programmatically.
        log.error(f"Pipeline failed: {e}")
        raise

# First attempt
result = run()

# On a subsequent attempt after a failure, pass the UUID from the log:
# result = run(resume_from=UUID("..."))
```

### Inspecting Available Checkpoints

Use `LocalFileCheckpointStorage` directly to list checkpoints and inspect their state before deciding which to resume:

```python
from pathlib import Path
from ondine.adapters.checkpoint_storage import LocalFileCheckpointStorage

storage = LocalFileCheckpointStorage(checkpoint_dir=Path(".checkpoints"))

for info in storage.list_checkpoints():
    print(
        f"Session: {info.session_id} | "
        f"Rows: {info.rows_processed}/{info.total_rows} | "
        f"Cost so far: ${info.cost_so_far:.4f} | "
        f"Saved: {info.timestamp:%Y-%m-%d %H:%M}"
    )
```

### Cleaning Up Old Checkpoints

To delete checkpoints older than a given number of days:

```python
from pathlib import Path
from ondine.adapters.checkpoint_storage import LocalFileCheckpointStorage

storage = LocalFileCheckpointStorage(checkpoint_dir=Path(".checkpoints"))
deleted = storage.cleanup_old_checkpoints(days=7)
print(f"Deleted {deleted} old checkpoint files.")
```

## When NOT to Use Checkpointing

- **Small datasets (< 1K rows):** The pipeline completes quickly; checkpointing adds overhead with little benefit.
- **Idempotent pipelines that are cheap to rerun:** If re-processing from scratch costs less than managing checkpoint state, skip it.
- **Streaming mode:** When using `execute_stream()`, consider checkpointing at the chunk level rather than the row level.

## Related

- [Execution Modes](execution-modes.md) — choosing between standard, async, and streaming
- [Cost Control](cost-control.md) — budget limits to pair with long-running jobs
- [Error Handling](error-handling.md) — retry policies for transient failures
