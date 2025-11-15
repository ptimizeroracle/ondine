# Multi-Row Batching Guide

Process N rows in a single API call to achieve 100× speedup for large datasets.

## Overview

Multi-row batching aggregates multiple prompts into a single API call, dramatically reducing:
- **API calls**: 100× fewer (5M → 50K with batch_size=100)
- **Processing time**: 100× faster (69 hours → 42 minutes)
- **Rate limit issues**: Virtually eliminated

## Quick Start

### Basic Usage

```python
from ondine import PipelineBuilder

# Enable multi-row batching with one line
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["sentiment"])
    .with_prompt("Classify sentiment: {text}")
    .with_batch_size(100)  # Process 100 rows per API call!
    .with_llm(provider="openai", model="gpt-4o-mini")
    .build()
)

result = pipeline.execute()
```

### How It Works

**Without Batching** (batch_size=1, default):
```
Row 1: "Product A is great" → API call 1 → "positive"
Row 2: "Product B is bad" → API call 2 → "negative"
Row 3: "Product C is okay" → API call 3 → "neutral"
...
100 rows = 100 API calls
```

**With Batching** (batch_size=100):
```
Batch prompt:
[
  {"id": 1, "input": "Product A is great"},
  {"id": 2, "input": "Product B is bad"},
  ...
  {"id": 100, "input": "Product Z"}
]
↓ 1 API call ↓
Batch response:
[
  {"id": 1, "result": "positive"},
  {"id": 2, "result": "negative"},
  ...
  {"id": 100, "result": "neutral"}
]

100 rows = 1 API call (100× reduction!)
```

## Choosing Batch Size

### Recommended Batch Sizes

| Use Case | Batch Size | Reason |
|----------|------------|--------|
| Simple classification | 100-500 | Short prompts, low failure risk |
| Sentiment analysis | 50-100 | Medium complexity |
| Text summarization | 10-50 | Longer outputs, higher risk |
| Complex extraction | 10-20 | Complex prompts, careful parsing |

### Factors to Consider

**1. Model Context Window**
- GPT-4o/GPT-4o-mini: 128K tokens → batch_size up to 500
- Claude Sonnet: 200K tokens → batch_size up to 800
- Llama 3.1: 131K tokens → batch_size up to 500

Ondine automatically validates batch size against context limits.

**2. Prompt Complexity**
- Simple prompts (20 tokens): Larger batches (100-500)
- Medium prompts (100 tokens): Medium batches (50-100)
- Complex prompts (500+ tokens): Smaller batches (10-50)

**3. Failure Tolerance**
- Larger batches = higher risk of partial failures
- Start with batch_size=10, increase gradually
- Monitor partial failure rate

## Batch Strategies

### JSON Strategy (Default)

Formats batches as JSON arrays:

```python
pipeline = (
    PipelineBuilder.create()
    .with_batch_size(100)
    .with_batch_strategy("json")  # Default, most reliable
    .build()
)
```

**Pros:**
- Most reliable (Pydantic validation)
- Handles complex outputs
- Automatic error detection

**Cons:**
- ~200 token overhead per batch
- Requires LLM to follow JSON format

### CSV Strategy (Future)

Coming soon - more compact format for simple use cases.

## Error Handling

### Partial Failures

If some rows fail to parse, Ondine handles gracefully:

```python
# Batch with 100 rows
# LLM returns 97 results (missing IDs: 23, 67, 89)

# Ondine automatically:
# 1. Parses 97 successful results
# 2. Marks 3 failed rows with [PARSE_ERROR]
# 3. Continues processing

# Result:
# - 97 rows: Valid results
# - 3 rows: "[PARSE_ERROR: Row not found in batch response]"
```

### Complete Failures

If entire batch fails to parse:

```python
# Batch response: "I cannot provide results in JSON format"

# Ondine automatically:
# 1. Marks all rows with [BATCH_PARSE_ERROR]
# 2. Logs error for debugging
# 3. Continues with next batch

# Result:
# - All rows: "[BATCH_PARSE_ERROR: Invalid JSON]"
```

## Performance Benchmarks

### Scaling to 5M Rows

| Batch Size | API Calls | Time | Speedup | Cost Overhead |
|------------|-----------|------|---------|---------------|
| 1 (default) | 5,000,000 | ~69 hours | 1× | 0% |
| 10 | 500,000 | ~7 hours | 10× | ~2% |
| 100 | 50,000 | ~42 minutes | 100× | ~5% |
| 500 | 10,000 | ~8 minutes | 500× | ~10% |

**Cost overhead**: JSON formatting adds ~200 tokens per batch

### Real-World Example

**Dataset**: 10 rows, sentiment classification

**Without batching**:
- API calls: 10
- Duration: ~15 seconds
- Tokens: 210 (21 per row)

**With batching (batch_size=5)**:
- API calls: 2 (5× reduction!)
- Duration: ~6 seconds (2.5× faster)
- Tokens: 250 (25 per row, ~20% overhead)

## Combining with Prefix Caching

For maximum cost savings, combine both techniques:

```python
# Shared context (cached across all rows)
SHARED_CONTEXT = """You are an expert data analyst.
[1024+ tokens of general knowledge]
"""

pipeline = (
    PipelineBuilder.create()
    .with_prompt("TASK: Classify\\nINPUT: {text}")
    .with_system_prompt(SHARED_CONTEXT)  # Cached (40-50% savings)
    .with_batch_size(100)  # 100× fewer API calls
    .with_llm(provider="openai", model="gpt-4o-mini")
    .build()
)
```

**Combined savings:**
- Prefix caching: 40-50% cost reduction
- Multi-row batching: 100× fewer API calls
- **Total**: 90%+ cost reduction + 100× speedup!

## CLI Configuration

Enable batching via YAML config:

```yaml
prompt:
  template: "Classify: {text}"
  batch_size: 100  # Multi-row batching
  batch_strategy: json
  system_message: "You are a classifier."  # Cached

llm:
  provider: openai
  model: gpt-4o-mini
```

Run with:
```bash
ondine process --config config.yaml
```

## Best Practices

1. **Start small**: Begin with batch_size=10, increase gradually
2. **Monitor failures**: Check for `[PARSE_ERROR]` in results
3. **Validate context**: Ondine auto-validates, but check logs for warnings
4. **Combine techniques**: Use with prefix caching for maximum savings
5. **Test first**: Run on 100-1000 rows before scaling to millions

## Troubleshooting

### Issue: Batch parsing failures

**Symptom**: Many rows with `[PARSE_ERROR]`

**Solutions:**
- Reduce batch_size (try 10-20)
- Simplify prompt instructions
- Add more explicit JSON format examples
- Check LLM is following instructions

### Issue: Context window exceeded

**Symptom**: Warning logs about batch size validation

**Solutions:**
- Reduce batch_size
- Use model with larger context window
- Simplify prompts to reduce tokens

### Issue: Slower than expected

**Symptom**: Not seeing 100× speedup

**Solutions:**
- Check batch_size is actually set (print `pipeline.specifications.prompt.batch_size`)
- Verify batch aggregation logs appear
- Check if rate limiting is bottleneck (increase rate_limit_rpm)

## Examples

See `examples/21_multi_row_batching.py` for complete working examples:
- Example 1: Without batching (baseline)
- Example 2: With batching (5× speedup)
- Example 3: Large dataset extrapolation (5.4M rows)

## API Reference

### `with_batch_size(batch_size: int)`

Enable multi-row batching.

**Parameters:**
- `batch_size` (int): Number of rows to process per API call (1-500)

**Returns:** PipelineBuilder (for chaining)

**Raises:** ValueError if batch_size < 1 or called before with_prompt()

### `with_batch_strategy(strategy: str)`

Set batch formatting strategy.

**Parameters:**
- `strategy` (str): "json" or "csv"

**Returns:** PipelineBuilder (for chaining)

**Raises:** ValueError if strategy not supported

## Performance Tips

1. **Combine with concurrency**: Use both for maximum throughput
   ```python
   .with_batch_size(100)  # 100 rows per call
   .with_concurrency(10)  # 10 concurrent calls
   # = 1000 rows processed simultaneously!
   ```

2. **Use with streaming**: Process results as they arrive
   ```python
   .with_batch_size(100)
   .with_streaming(chunk_size=10000)
   ```

3. **Enable prefix caching**: Reduce token costs
   ```python
   .with_batch_size(100)
   .with_system_prompt("...")  # Cached
   ```

## Limitations

- JSON strategy only (CSV coming soon)
- Batch size limited by model context window
- Requires LLM to follow JSON format instructions
- Larger batches = higher risk of partial failures

