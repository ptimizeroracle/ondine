# Error Handling

Ondine provides two complementary layers of error handling: a configurable **policy** that determines what happens when a row fails (skip it, retry it, use a default, or abort), and a **retry handler** that executes transient-error retries with exponential backoff.

## Overview

| Layer | What it handles | Where configured |
|-------|----------------|-----------------|
| `ErrorPolicy` | Row-level failures: skip, retry, fail, or substitute | `with_error_policy()` |
| `max_retries` | Maximum attempts under the `retry` policy | `with_max_retries()` |
| `retry_delay` | Initial backoff delay in seconds | `ProcessingSpec.retry_delay` |
| `RetryHandler` | Transient API errors (rate limits, network timeouts) | Applied automatically |

## ErrorPolicy

`ErrorPolicy` is a string enum defined in `ondine.core.specifications`. It controls what the pipeline does when a stage fails for a given row.

```python
from ondine.core.specifications import ErrorPolicy

# Available values:
ErrorPolicy.SKIP        # "skip"
ErrorPolicy.FAIL        # "fail"
ErrorPolicy.RETRY       # "retry"
ErrorPolicy.USE_DEFAULT # "use_default"
```

### SKIP (default)

Log the error and leave the output column empty for that row. Processing continues with the next row.

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

Use `skip` when partial results are acceptable and you want maximum throughput.

### FAIL

Raise the error immediately and stop the pipeline. A checkpoint is saved before raising, so you can resume after fixing the underlying problem.

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

Use `fail` in CI pipelines or data validation workflows where any failure is unacceptable.

### RETRY

Retry the failed row up to `max_retries` times. If all attempts are exhausted, the row is skipped (same behaviour as `skip`).

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

Use `retry` when failures are likely to be transient (rate limits, network hiccups).

### USE_DEFAULT

Return a fixed default value for any failed row instead of leaving it empty or raising.

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

Use `use_default` when downstream consumers require a non-null value in every row (e.g. a report that cannot tolerate missing values).

## Configuring Retries

### `with_max_retries(retries: int)`

Set the maximum number of retry attempts when the policy is `retry`.

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

The initial delay in seconds between retry attempts. The `RetryHandler` applies exponential backoff, so delays grow as: `retry_delay * 2^(attempt - 1)`, capped at 60 seconds.

```python
from ondine.core.specifications import ProcessingSpec, ErrorPolicy

spec = ProcessingSpec(
    max_retries=5,
    retry_delay=2.0,     # First retry waits 2s, second 4s, third 8s, ...
    error_policy=ErrorPolicy.RETRY,
)
```

**Default:** `1.0` seconds.

### Exponential Backoff Schedule

Given `retry_delay=1.0` and `max_retries=5`:

| Attempt | Delay before attempt |
|---------|---------------------|
| 1 (initial) | — |
| 2 | 1s |
| 3 | 2s |
| 4 | 4s |
| 5 | 8s |

The maximum delay is capped at 60 seconds regardless of the multiplier.

## Retry Levels

Ondine has two distinct retry mechanisms that operate at different scopes:

| Mechanism | Scope | Trigger |
|-----------|-------|---------|
| `RetryHandler` | Single LLM API call | Transient errors: rate limits (429), network timeouts, 502/503 |
| `ErrorPolicy.RETRY` | Pipeline row | Any stage failure, after `RetryHandler` exhausted |

`RetryHandler` runs automatically for transient errors regardless of your `ErrorPolicy`. It only retries `RetryableError`, `RateLimitError`, and `NetworkError` subtypes — configuration errors (invalid API key, 401, 403) are never retried and always fail immediately.

## Handling Partial Failures

When using `skip` or `retry`, successful rows are returned alongside failed ones. Inspect `result.errors` and the output DataFrame together to understand partial results:

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

## Production Error Handling Patterns

### Conservative: Skip and Monitor

Maximise throughput while capturing failure statistics for alerting:

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

### Resilient: Retry with Backoff

For high-volume jobs hitting rate-limited APIs:

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

### Strict: Fail Fast on Any Error

Use in data-quality pipelines where partial results are worse than no results:

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

### Combining Retry and Checkpointing

For multi-hour jobs, combine both features so a failure mid-run can be resumed without losing completed work:

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

Certain errors bypass all retry and skip logic and immediately halt the pipeline regardless of policy:

- Invalid API key / authentication failures (`401`, `403`)
- `NonRetryableError` and its subclasses

This prevents burning retries on configuration mistakes. Fix the underlying issue (e.g. set the correct API key) and re-run.

## Related

- [Checkpointing](checkpointing.md) — saving state so failed pipelines can resume
- [Cost Control](cost-control.md) — budget limits that complement fault-tolerance settings
- [Execution Modes](execution-modes.md) — async and streaming modes for large datasets
