# Error Handling

Two layers. A **policy** decides what happens when a row fails — skip it, retry it, substitute a default, or abort. A **retry handler** deals with transient API errors using exponential backoff. They work together, but they're separate things.

## Quick Reference

| Layer | What it handles | Where configured |
|-------|----------------|-----------------|
| `ErrorPolicy` | Row-level failures: skip, retry, fail, or substitute | `with_error_policy()` |
| `max_retries` | Maximum attempts under the `retry` policy | `with_max_retries()` |
| `retry_delay` | Initial backoff delay in seconds | `ProcessingSpec.retry_delay` |
| `RetryHandler` | Transient API errors (rate limits, network timeouts) | Applied automatically |

## ErrorPolicy

`ErrorPolicy` is a string enum in `ondine.core.specifications`. It tells the pipeline what to do when a stage fails for a given row.

```python
from ondine.core.specifications import ErrorPolicy

# Available values:
ErrorPolicy.SKIP        # "skip"
ErrorPolicy.FAIL        # "fail"
ErrorPolicy.RETRY       # "retry"
ErrorPolicy.USE_DEFAULT # "use_default"
```

### SKIP (default)

Logs the error, leaves the output column empty for that row, moves on.

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Classify: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_error_policy("skip")
    .build()
)

result = pipeline.execute()
# Failed rows will have None/NaN in the "result" column
failed = result.to_pandas()[result.to_pandas()["result"].isna()]
print(f"{len(failed)} rows failed and were skipped")
```

Good when partial results are fine and you want throughput.

### FAIL

Raises immediately. Stops the pipeline. A checkpoint gets saved before the raise, so you can fix the problem and resume.

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Classify: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_error_policy("fail")
    .build()
)
```

The right choice for CI pipelines or data validation where any failure means the output is useless.

### RETRY

Retries the failed row up to `max_retries` times. If all attempts are exhausted, falls back to skip behavior.

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Extract entities: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_error_policy("retry")
    .with_max_retries(3)
    .build()
)
```

The retry policy you'll want for most production use. Rate limits and network hiccups are the norm, not the exception.

### USE_DEFAULT

Returns a fixed default value for any failed row instead of leaving it empty or raising.

```python
from ondine.core.specifications import ProcessingSpec, ErrorPolicy

# Configure via ProcessingSpec directly for use_default
# (pass the spec to PipelineBuilder.with_processing_spec if supported,
#  or combine with a custom stage for the default value)
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["sentiment"])
    .with_prompt("Rate sentiment of: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_error_policy("use_default")
    .build()
)
```

For when downstream consumers can't handle nulls — dashboards, reports, that kind of thing.

## Configuring Retries

### `with_max_retries(retries: int)`

Maximum retry attempts when the policy is `retry`.

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Summarise: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_error_policy("retry")
    .with_max_retries(5)
    .build()
)
```

**Default:** `3`. Set to `0` to disable retries entirely.

### `retry_delay` (via `ProcessingSpec`)

Initial delay in seconds between retries. The `RetryHandler` applies exponential backoff: `retry_delay * 2^(attempt - 1)`, capped at 60 seconds.

```python
from ondine.core.specifications import ProcessingSpec, ErrorPolicy

spec = ProcessingSpec(
    max_retries=5,
    retry_delay=2.0,     # First retry waits 2s, second 4s, third 8s, ...
    error_policy=ErrorPolicy.RETRY,
)
```

**Default:** `1.0` seconds.

### Backoff Schedule

With `retry_delay=1.0` and `max_retries=5`:

| Attempt | Delay before attempt |
|---------|---------------------|
| 1 (initial) | — |
| 2 | 1s |
| 3 | 2s |
| 4 | 4s |
| 5 | 8s |

Cap is 60 seconds no matter the multiplier.

## Two Levels of Retry

One thing to watch: Ondine has two retry mechanisms at different scopes.

| Mechanism | Scope | Trigger |
|-----------|-------|---------|
| `RetryHandler` | Single LLM API call | Transient errors: rate limits (429), network timeouts, 502/503 |
| `ErrorPolicy.RETRY` | Pipeline row | Any stage failure, after `RetryHandler` is exhausted |

`RetryHandler` fires automatically for transient errors regardless of your `ErrorPolicy`. It only retries `RetryableError`, `RateLimitError`, and `NetworkError` subtypes. Config errors (bad API key, 401, 403) fail immediately — no retries, no wasted time.

## Handling Partial Failures

With `skip` or `retry`, you get successful rows alongside failed ones. Here's how to inspect what happened:

```python
result = pipeline.execute()
df = result.to_pandas()

# Check overall failure rate
total = result.metrics.total_rows
skipped = result.metrics.total_rows - result.metrics.success_count
print(f"Success rate: {result.metrics.success_count}/{total} ({skipped} skipped)")

# Inspect errors
for err in result.errors:
    print(f"Row {err}: failed")

# Rows with empty outputs (skipped due to error)
failed_rows = df[df["result"].isna()]
print(f"Rows with missing output: {len(failed_rows)}")
```

## Production Patterns

### Skip and Monitor

Max throughput, but alert if too many rows fail:

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv(
        "data.csv",
        input_columns=["description"],
        output_columns=["category"],
    )
    .with_prompt("Categorise this product: {description}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_error_policy("skip")
    .with_checkpoint_interval(500)
    .with_max_budget(20.0)
    .build()
)

result = pipeline.execute()
df = result.to_pandas()

failure_rate = 1 - (result.metrics.success_count / result.metrics.total_rows)
if failure_rate > 0.05:
    raise RuntimeError(f"Failure rate {failure_rate:.1%} exceeds 5% threshold")

df.to_csv("output.csv", index=False)
```

### Retry with Backoff

The setup you'll want for high-volume jobs against rate-limited APIs:

```python
from ondine import PipelineBuilder
from ondine.core.specifications import ProcessingSpec, ErrorPolicy

processing = ProcessingSpec(
    batch_size=50,
    concurrency=5,
    error_policy=ErrorPolicy.RETRY,
    max_retries=5,
    retry_delay=2.0,          # 2s → 4s → 8s → 16s → 32s
    checkpoint_interval=250,
    rate_limit_rpm=60,
)

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Analyse sentiment: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .build()
)
# Attach the custom processing spec
pipeline.specifications.processing = processing

result = pipeline.execute()
```

### Fail Fast

For data-quality pipelines where partial results are worse than no results:

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("validated_input.csv", input_columns=["text"], output_columns=["label"])
    .with_prompt("Label this text: {text}")
    .with_llm(provider="anthropic", model="claude-sonnet-4-20250514")
    .with_error_policy("fail")
    .with_checkpoint_dir(".checkpoints/labelling-job")
    .build()
)

try:
    result = pipeline.execute()
except Exception as e:
    print(f"Pipeline aborted: {e}")
    print("Fix the issue and resume from the checkpoint printed above.")
```

### Retry + Checkpointing Together

Multi-hour jobs need both. A failure mid-run shouldn't mean losing hours of completed work.

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv(
        "large_dataset.csv",
        input_columns=["text"],
        output_columns=["summary"],
    )
    .with_prompt("Summarise the following: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_error_policy("retry")
    .with_max_retries(3)
    .with_checkpoint_dir(".checkpoints/summary-job")
    .with_checkpoint_interval(500)
    .with_checkpoint_cleanup(False)  # Retain in case downstream write fails
    .with_max_budget(100.0)
    .build()
)

result = pipeline.execute()
```

## Non-Retryable Errors

Some errors bypass all retry and skip logic. Pipeline halts immediately regardless of policy:

- Invalid API key / auth failures (`401`, `403`)
- `NonRetryableError` and its subclasses

No point burning retries on a bad API key. Fix the config, re-run.

## Related

- [Checkpointing](checkpointing.md) — saving state so failed pipelines can resume
- [Cost Control](cost-control.md) — budget limits that complement fault-tolerance settings
- [Execution Modes](execution-modes.md) — async and streaming modes for large datasets
