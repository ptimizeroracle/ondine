# Quickstart

Build your first Ondine pipeline in under five minutes.

## Prerequisites

- Python 3.10+
- Ondine installed (`pip install ondine`)
- An API key for OpenAI or another LLM provider

## Your First Pipeline

### 1. Setup

Create a new Python file and set your API key:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."  # Or use .env file
```

### 2. Prepare Sample Data

Create a simple CSV file or use a pandas DataFrame:

```python
import pandas as pd

# Sample data
data = pd.DataFrame({
    "product": [
        "iPhone 15 Pro Max 256GB",
        "Samsung Galaxy S24 Ultra",
        "Google Pixel 8 Pro"
    ]
})

# Save to CSV
data.to_csv("products.csv", index=False)
```

### 3. Quick API (Simplest)

Fewest lines of code to get results:

```python
from ondine import QuickPipeline

# Create and run pipeline
pipeline = QuickPipeline.create(
    data="products.csv",
    prompt="Extract the brand name from: {product}",
    model="gpt-4o-mini"
)

result = pipeline.execute()

# View results
print(result.data)
print(f"Cost: ${result.costs.total_cost:.4f}")
```

**Output:**
```
   product                      response
0  iPhone 15 Pro Max 256GB       Apple
1  Samsung Galaxy S24 Ultra      Samsung
2  Google Pixel 8 Pro            Google

Cost: $0.0012
```

### 4. Builder API (More Control)

When you need to pin temperature, set concurrency, or name your output columns explicitly:

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv(
        "products.csv",
        input_columns=["product"],
        output_columns=["brand"]
    )
    .with_prompt("Extract the brand name from: {product}")
    .with_llm(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0
    )
    .with_batch_size(100)
    .with_concurrency(5)
    .build()
)

# Estimate cost before running
estimate = pipeline.estimate_cost()
print(f"Estimated cost: ${estimate.total_cost:.4f}")

# Execute
result = pipeline.execute()
print(result.data)
```

## Common Patterns

### Pattern 1: Data Cleaning

```python
from ondine import QuickPipeline

pipeline = QuickPipeline.create(
    data="messy_data.csv",
    prompt="""
    Clean and standardize this text:
    {text}
    
    Remove special characters, fix capitalization, trim whitespace.
    """,
    model="gpt-4o-mini"
)

result = pipeline.execute()
```

### Pattern 2: Classification

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv(
        "reviews.csv",
        input_columns=["review_text"],
        output_columns=["sentiment"]
    )
    .with_prompt("""
    Classify the sentiment of this review as: positive, negative, or neutral
    
    Review: {review_text}
    """)
    .with_llm(provider="openai", model="gpt-4o-mini", temperature=0.0)
    .build()
)

result = pipeline.execute()
```

### Pattern 3: Structured Extraction

```python
from ondine import PipelineBuilder
from ondine.stages.parser_factory import JSONParser

pipeline = (
    PipelineBuilder.create()
    .from_csv(
        "descriptions.csv",
        input_columns=["description"],
        output_columns=["brand", "model", "price"]
    )
    .with_prompt("""
    Extract product information as JSON:
    {{
      "brand": "...",
      "model": "...",
      "price": "..."
    }}
    
    Description: {description}
    """)
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_parser(JSONParser())
    .build()
)

result = pipeline.execute()
```

## Understanding the Results

`execute()` returns an `ExecutionResult` with the processed DataFrame, metrics, and cost breakdown:

```python
result = pipeline.execute()

# Access the processed data
print(result.data)              # pandas DataFrame with results

# View metrics
print(result.metrics.processed_rows)    # Number of rows processed
print(result.metrics.successful_rows)   # Successfully processed
print(result.metrics.failed_rows)       # Failed rows
print(result.metrics.elapsed_time)      # Total time in seconds

# Check costs
print(result.costs.total_cost)          # Total cost in USD
print(result.costs.input_tokens)        # Input tokens used
print(result.costs.output_tokens)       # Output tokens generated
```

## Cost Estimation

Estimate costs before processing large datasets. A 100k-row job on GPT-4o can run into real money; five seconds of estimation saves you from a surprise invoice:

```python
pipeline = PipelineBuilder.create()...build()

# Get cost estimate
estimate = pipeline.estimate_cost()

print(f"Estimated rows: {estimate.total_rows}")
print(f"Estimated tokens: {estimate.estimated_tokens}")
print(f"Estimated cost: ${estimate.total_cost:.4f}")

# Proceed only if acceptable
if estimate.total_cost < 10.0:
    result = pipeline.execute()
else:
    print("Cost too high, aborting")
```

## Error Handling

Failed requests are retried automatically with exponential backoff:

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", ...)
    .with_prompt("...")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_retry_policy(max_retries=3, backoff_factor=2.0)
    .build()
)

result = pipeline.execute()

# Check for failures
if result.metrics.failed_rows > 0:
    print(f"Failed rows: {result.metrics.failed_rows}")
    print(result.data[result.data['response'].isna()])
```

## Checkpointing

Long-running jobs should checkpoint so a crash at row 9,500 of 10,000 doesn't force a full restart:

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("large_dataset.csv", ...)
    .with_prompt("...")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_checkpoint("./checkpoints", interval=100)  # Save every 100 rows
    .build()
)

# If interrupted, re-run same command - it will resume from checkpoint
result = pipeline.execute()
```

## Next Steps

- [Core Concepts](core-concepts.md) - How the pipeline architecture fits together
- [Execution Modes](../guides/execution-modes.md) - Async and streaming execution
- [Structured Output](../guides/structured-output.md) - Type-safe Pydantic models
- [Cost Control](../guides/cost-control.md) - Budget limits and optimization
- [API Reference](../api/index.md) - Full API documentation

