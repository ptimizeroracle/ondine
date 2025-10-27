# Execution Modes

Ondine supports three execution modes optimized for different use cases. Choosing the right mode impacts performance, memory usage, and throughput.

## Overview

| Mode | Best For | Memory Usage | Throughput | Complexity |
|------|----------|--------------|------------|------------|
| **Standard** | Small datasets (< 50K rows) | High | Low-Medium | Simple |
| **Async** | High throughput needs | High | High | Medium |
| **Streaming** | Large datasets (100K+ rows) | Constant | Medium | Medium |

## Standard Execution (Default)

Synchronous, single-threaded processing. The simplest mode.

### Usage

```python
from ondine import PipelineBuilder

pipeline = PipelineBuilder.create().from_csv(...).build()
result = pipeline.execute()
```

### Characteristics

- **Execution:** Sequential row-by-row or batch-by-batch
- **Memory:** Loads entire dataset into memory
- **Concurrency:** None (single-threaded)
- **Return:** Complete `ExecutionResult` with full DataFrame

### When to Use

- Dataset fits comfortably in memory (< 50K rows typical)
- Straightforward processing without complex coordination
- Debugging or testing pipelines
- Simple scripts and notebooks

### When NOT to Use

- Dataset is large (> 100K rows)
- Need maximum throughput
- Memory is constrained

### Example

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Process: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_batch_size(100)
    .build()
)

# Simple synchronous execution
result = pipeline.execute()
print(f"Processed {result.metrics.processed_rows} rows")
```

## Async Execution (Concurrent)

Asynchronous processing with configurable concurrency. Maximizes throughput.

### Usage

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv(...)
    .with_async_execution(max_concurrency=10)
    .build()
)

result = await pipeline.execute_async()
```

### Characteristics

- **Execution:** Concurrent async/await with controlled parallelism
- **Memory:** Loads entire dataset into memory
- **Concurrency:** Configurable (e.g., 10-50 concurrent requests)
- **Return:** Complete `ExecutionResult` with full DataFrame

### When to Use

- Need high throughput (processing many rows quickly)
- LLM API supports async (most modern APIs do)
- Running in async context (FastAPI, aiohttp, async scripts)
- Dataset fits in memory but need speed
- Provider has high rate limits

### When NOT to Use

- Running in synchronous context (use standard mode instead)
- Dataset is very large (> 100K rows) - consider streaming
- Provider has strict rate limits (async may hit limits faster)
- Memory is constrained

### Example

```python
import asyncio
from ondine import PipelineBuilder

async def process_data():
    pipeline = (
        PipelineBuilder.create()
        .from_csv("large_data.csv", input_columns=["text"], output_columns=["result"])
        .with_prompt("Analyze: {text}")
        .with_llm(provider="openai", model="gpt-4o-mini")
        .with_async_execution(max_concurrency=20)  # 20 concurrent requests
        .with_rate_limit(100)  # Respect API limits
        .build()
    )
    
    result = await pipeline.execute_async()
    print(f"Processed {result.metrics.processed_rows} rows")
    print(f"Time: {result.metrics.elapsed_time:.2f}s")
    return result

# Run async pipeline
result = asyncio.run(process_data())
```

### Concurrency Guidelines

| Provider | Recommended Max Concurrency |
|----------|----------------------------|
| OpenAI (Tier 1) | 10-20 |
| OpenAI (Tier 4+) | 50-100 |
| Anthropic | 10-20 |
| Groq | 30-50 |
| Azure OpenAI | 10-30 (varies by deployment) |
| Local MLX | 1 (no concurrency benefit) |

## Streaming Execution (Memory-Efficient)

Process data in chunks with constant memory footprint. Best for very large datasets.

### Usage

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv(...)
    .with_streaming(chunk_size=1000)
    .build()
)

for chunk_result in pipeline.execute_stream():
    # Process each chunk as it completes
    print(f"Processed chunk: {len(chunk_result.data)} rows")
    chunk_result.data.to_csv("output.csv", mode="a", header=False)
```

### Characteristics

- **Execution:** Processes data in fixed-size chunks
- **Memory:** Constant footprint (1-2 chunks in memory max)
- **Concurrency:** Can combine with async for concurrent chunks
- **Return:** Iterator yielding `ExecutionResult` per chunk

### When to Use

- Large datasets (100K+ rows)
- Limited memory (processing datasets larger than available RAM)
- Need constant memory footprint
- Want early/incremental results
- Processing takes hours/days

### When NOT to Use

- Dataset under 50K rows (overhead not justified)
- Need entire dataset in memory for post-processing
- Pipeline has dependencies between rows
- Checkpointing is sufficient for your use case

### Example

```python
from ondine import PipelineBuilder
import pandas as pd

pipeline = (
    PipelineBuilder.create()
    .from_csv(
        "huge_dataset.csv",  # 500K rows
        input_columns=["text"],
        output_columns=["summary"]
    )
    .with_prompt("Summarize: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_streaming(chunk_size=5000)  # Process 5K rows at a time
    .build()
)

# Process in streaming fashion
all_results = []
for i, chunk_result in enumerate(pipeline.execute_stream()):
    print(f"Chunk {i+1}: {len(chunk_result.data)} rows, "
          f"Cost: ${chunk_result.costs.total_cost:.4f}")
    
    # Write incrementally
    mode = "w" if i == 0 else "a"
    header = i == 0
    chunk_result.data.to_csv("output.csv", mode=mode, header=header, index=False)
    
    all_results.append(chunk_result)

# Aggregate metrics
total_rows = sum(r.metrics.processed_rows for r in all_results)
total_cost = sum(r.costs.total_cost for r in all_results)
print(f"Total: {total_rows} rows, ${total_cost:.2f}")
```

### Chunk Size Guidelines

| Dataset Size | Recommended Chunk Size |
|-------------|------------------------|
| 10K-50K | 1,000 |
| 50K-100K | 2,500 |
| 100K-500K | 5,000 |
| 500K-1M | 10,000 |
| 1M+ | 25,000 |

Larger chunks = fewer overhead, smaller chunks = finer progress tracking.

## Streaming + Async (Maximum Efficiency)

Combine streaming with async for both memory efficiency and high throughput:

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("huge_dataset.csv", ...)
    .with_streaming(chunk_size=5000)
    .with_async_execution(max_concurrency=20)
    .build()
)

async for chunk_result in pipeline.execute_stream_async():
    # Each chunk processed with 20 concurrent requests
    print(f"Chunk done: {len(chunk_result.data)} rows")
```

## Comparison Example

Processing the same 10K row dataset with different modes:

### Standard Mode
```python
# Loads all 10K rows into memory, processes sequentially
result = pipeline.execute()
# Time: ~120s, Memory: ~500MB peak
```

### Async Mode
```python
# Loads all 10K rows into memory, 20 concurrent requests
result = await pipeline.execute_async()
# Time: ~15s, Memory: ~500MB peak
```

### Streaming Mode
```python
# Processes 1K rows at a time
for chunk in pipeline.execute_stream():
    pass
# Time: ~110s, Memory: ~50MB constant
```

### Streaming + Async
```python
# Processes 1K rows at a time with 20 concurrent requests per chunk
async for chunk in pipeline.execute_stream_async():
    pass
# Time: ~18s, Memory: ~50MB constant
```

## Choosing the Right Mode

Use this decision tree:

```
Is dataset > 100K rows?
├─ YES → Use Streaming
│         └─ Need speed? → Add Async (streaming + async)
│
└─ NO → Dataset < 100K rows
         ├─ Need maximum speed?
         │  └─ YES → Use Async
         │
         └─ NO → Use Standard (simplest)
```

## Memory Considerations

### Standard/Async Memory Usage

```
Memory = Base + (Dataset Size × Row Size)
```

Example: 50K rows × 10KB/row = ~500MB

### Streaming Memory Usage

```
Memory = Base + (Chunk Size × Row Size)
```

Example: 1K chunk × 10KB/row = ~10MB (constant)

## Performance Tips

### For All Modes

1. **Use appropriate batch size**: Larger batches = fewer API calls
2. **Enable checkpointing**: Resume on failures
3. **Set rate limits**: Respect provider limits
4. **Monitor costs**: Use budget controls

### For Async Mode

1. **Tune concurrency**: Start low, increase gradually
2. **Respect rate limits**: Too much concurrency can trigger rate limiting
3. **Monitor memory**: Each concurrent request consumes memory

### For Streaming Mode

1. **Choose appropriate chunk size**: Balance memory vs. overhead
2. **Write incrementally**: Don't accumulate all results in memory
3. **Enable checkpointing per chunk**: More frequent checkpoints

## Related Examples

- [`examples/07_async_execution.py`](../../examples/07_async_execution.py) - Async processing
- [`examples/08_streaming_large_files.py`](../../examples/08_streaming_large_files.py) - Streaming
- [Cost Control Guide](cost-control.md) - Budget management
- [API Reference](../api/pipeline.md) - Complete API docs

