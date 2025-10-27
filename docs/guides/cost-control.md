# Cost Control

Ondine provides comprehensive cost management features to prevent budget overruns and optimize spending on LLM APIs.

## Pre-Execution Cost Estimation

Always estimate costs before processing large datasets:

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["summary"])
    .with_prompt("Summarize: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .build()
)

# Get cost estimate
estimate = pipeline.estimate_cost()

print(f"Total rows: {estimate.total_rows}")
print(f"Estimated tokens: {estimate.estimated_tokens}")
print(f"Estimated cost: ${estimate.total_cost:.4f}")
print(f"Cost per row: ${estimate.cost_per_row:.6f}")
```

## Budget Limits

Set maximum budget to prevent overspending:

```python
from decimal import Decimal

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", ...)
    .with_prompt("...")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_max_budget(Decimal("10.0"))  # Max $10 USD
    .build()
)

# Execution stops if budget exceeded
result = pipeline.execute()
```

## Real-Time Cost Tracking

Monitor costs during execution:

```python
result = pipeline.execute()

# View detailed costs
print(f"Total cost: ${result.costs.total_cost:.4f}")
print(f"Input tokens: {result.costs.input_tokens:,}")
print(f"Output tokens: {result.costs.output_tokens:,}")
print(f"Total tokens: {result.costs.total_tokens:,}")
print(f"Cost per row: ${result.costs.total_cost / result.metrics.processed_rows:.6f}")
```

## Cost Optimization Strategies

### 1. Choose Cost-Effective Models

```python
# Expensive: GPT-4
.with_llm(provider="openai", model="gpt-4")  # ~$0.03/1K tokens

# Cost-effective: GPT-4o-mini
.with_llm(provider="openai", model="gpt-4o-mini")  # ~$0.0001/1K tokens

# Free: Local MLX (Apple Silicon)
.with_llm(provider="mlx", model="mlx-community/Qwen2.5-7B-Instruct-4bit")  # $0
```

### 2. Optimize Prompts

Shorter prompts = lower costs:

```python
# Expensive: Verbose prompt
prompt = """
You are a helpful assistant specialized in text summarization.
Please carefully read the following text and provide a comprehensive
summary that captures the main points while being concise.

Text: {text}

Please provide your summary below:
"""

# Cost-effective: Concise prompt
prompt = "Summarize in 1 sentence: {text}"
```

### 3. Use Temperature=0 for Deterministic Tasks

Lower temperature often produces shorter, more focused responses:

```python
.with_llm(provider="openai", model="gpt-4o-mini", temperature=0.0)
```

### 4. Set Max Tokens

Limit response length:

```python
.with_llm(
    provider="openai",
    model="gpt-4o-mini",
    max_tokens=100  # Limit response to 100 tokens
)
```

### 5. Batch Processing

Process multiple items per request when possible:

```python
.with_batch_size(100)  # Process 100 rows per batch
```

### 6. Use Cheaper Providers

Consider alternative providers for cost savings:

| Provider | Cost (per 1M tokens) | Speed |
|----------|---------------------|-------|
| OpenAI GPT-4o-mini | $0.15 | Fast |
| Groq (Llama) | $0.05-0.10 | Very Fast |
| Together.AI | $0.20-0.60 | Fast |
| Local MLX | $0 | Medium |

## Cost Reporting

### Summary Report

```python
result = pipeline.execute()

print("\n=== Cost Summary ===")
print(f"Rows processed: {result.metrics.processed_rows}")
print(f"Total cost: ${result.costs.total_cost:.4f}")
print(f"Average cost/row: ${result.costs.total_cost / result.metrics.processed_rows:.6f}")
print(f"Input tokens: {result.costs.input_tokens:,}")
print(f"Output tokens: {result.costs.output_tokens:,}")
```

### Export to CSV

```python
import pandas as pd

# Create cost report
cost_report = pd.DataFrame([{
    "date": pd.Timestamp.now(),
    "rows": result.metrics.processed_rows,
    "total_cost": result.costs.total_cost,
    "input_tokens": result.costs.input_tokens,
    "output_tokens": result.costs.output_tokens,
    "provider": "openai",
    "model": "gpt-4o-mini"
}])

# Append to running cost log
cost_report.to_csv("cost_log.csv", mode="a", header=False, index=False)
```

## Budget-Aware Workflows

### Estimate Before Execution

```python
def safe_execute(pipeline, max_cost=10.0):
    estimate = pipeline.estimate_cost()
    
    if estimate.total_cost > max_cost:
        print(f"Estimated cost ${estimate.total_cost:.2f} exceeds budget ${max_cost:.2f}")
        return None
    
    print(f"Proceeding with estimated cost: ${estimate.total_cost:.2f}")
    return pipeline.execute()

result = safe_execute(pipeline, max_cost=5.0)
```

### Incremental Processing with Cost Checks

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("large_data.csv", ...)
    .with_streaming(chunk_size=1000)
    .with_max_budget(Decimal("20.0"))
    .build()
)

total_cost = 0.0
for chunk_result in pipeline.execute_stream():
    total_cost += chunk_result.costs.total_cost
    print(f"Chunk cost: ${chunk_result.costs.total_cost:.4f}, "
          f"Total so far: ${total_cost:.4f}")
    
    if total_cost > 15.0:
        print("Approaching budget limit, stopping")
        break
```

## Cost Tracking Across Runs

Track costs across multiple pipeline runs:

```python
from ondine.utils import CostTracker

tracker = CostTracker()

# Run multiple pipelines
for config in pipeline_configs:
    pipeline = build_pipeline(config)
    result = pipeline.execute()
    
    tracker.add(
        provider=config["provider"],
        model=config["model"],
        cost=result.costs.total_cost,
        tokens=result.costs.total_tokens
    )

# View summary
summary = tracker.summary()
print(f"Total spend: ${summary['total_cost']:.2f}")
print(f"Total tokens: {summary['total_tokens']:,}")
```

## Related

- [Execution Modes](execution-modes.md) - Choose efficient execution strategy
- [API Reference: CostTracker](../api/utils.md#costtracker)
- [Structured Output](structured-output.md) - Optimize response parsing

