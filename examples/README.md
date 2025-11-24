# Ondine - Examples

This directory contains example scripts demonstrating various features of Ondine LLM Dataset Engine.

## What's New in v1.4.0

**LiteLLM Native Integration:**
- ✅ 100+ providers supported (vs 5 before)
- ✅ Router for load balancing and failover
- ✅ Instructor for type-safe structured output
- ✅ Redis caching for response deduplication
- ✅ Native async throughout

All examples updated to work with new architecture!

## Prerequisites

1. Install the package:
   ```bash
   uv add ondine
   ```

2. Set up your API keys:
   ```bash
   # LiteLLM uses standard environment variables
   export OPENAI_API_KEY="your-key-here"
   export GROQ_API_KEY="your-key-here"
   export ANTHROPIC_API_KEY="your-key-here"
   export AZURE_API_KEY="your-key-here"  # For Azure
   # ... +100 more providers supported!
   ```

## Examples

### 01_quickstart.py
**Basic pipeline usage with fluent API**

Demonstrates:
- Creating a pipeline with PipelineBuilder
- Processing a DataFrame
- Multi-column inputs
- Cost estimation
- Viewing metrics

Run:
```bash
python examples/01_quickstart.py
```

### 02_simple_processor.py
**Minimal configuration with DatasetProcessor**

Demonstrates:
- Simplified API for single-column processing
- Processing CSV files
- Running on sample data first
- Sentiment analysis use case

Run:
```bash
python examples/02_simple_processor.py
```

### 03_structured_output.py
**Type-safe structured output with Pydantic (NEW in v1.4.0)**

Demonstrates:
- **Instructor integration** for type-safe output
- Pydantic model validation with auto-retry
- Multi-column output extraction
- Automatic mode selection (JSON vs function calling)
- Works with all providers (Groq, OpenAI, Anthropic)

Run:
```bash
python examples/03_structured_output.py
```

**New Features**:
- Auto-retry on validation errors (max_retries=3)
- Groq works with JSON mode (no XML issues!)
- OpenAI/Anthropic use function calling (native)

### 04_with_cost_control.py
**Budget limits and cost tracking**

Demonstrates:
- Setting maximum budget
- Cost estimation before execution
- Batch processing configuration
- Rate limiting
- Checkpointing
- Cost variance analysis

Run:
```bash
python examples/04_with_cost_control.py
```

## New Features in v1.4.0

### Router for Load Balancing (NEW!)

Multi-provider failover and load balancing:

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", ...)
    .with_router(
        model_list=[
            {
                "model_name": "fast-llm",
                "litellm_params": {
                    "model": "groq/llama-3.3-70b-versatile",
                    "api_key": os.getenv("GROQ_API_KEY")
                }
            },
            {
                "model_name": "fast-llm",  # Automatic failover
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": os.getenv("OPENAI_API_KEY")
                }
            }
        ],
        routing_strategy="simple-shuffle"
    )
    .build()
)
```

**Features**:
- Automatic failover if one provider fails
- Load balancing across deployments
- Built-in retries and cooldowns

### Redis Caching (NEW!)

Avoid duplicate API calls with response caching:

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", ...)
    .with_redis_cache(
        redis_url="redis://localhost:6379",
        ttl=3600  # 1 hour
    )
    .build()
)
```

**Benefits**:
- Zero cost for cache hits
- Instant response (no API call)
- Distributed across processes

### Structured Output with Instructor (NEW!)

Type-safe Pydantic models with auto-retry:

```python
from pydantic import BaseModel

class ProductInfo(BaseModel):
    brand: str
    price: float

pipeline = (
    PipelineBuilder.create()
    .from_csv("products.csv", ...)
    .with_structured_output(ProductInfo)  # Type-safe!
    .build()
)
```

**Features**:
- Automatic validation retry (max_retries=3)
- Auto-detection (JSON mode vs function calling)
- Works with all 100+ providers

## Common Patterns

### Pattern 1: Quick Test on Sample Data
```python
# Always test on a small sample first
pipeline = builder.build()
sample_data = df.head(10)
# ... process sample ...
# Then process full dataset
```

### Pattern 2: Cost Estimation Before Execution
```python
estimate = pipeline.estimate_cost()
if estimate.total_cost > MAX_BUDGET:
    print("Cost too high, aborting")
    exit()
```

### Pattern 3: Multi-Provider Resilience
```python
# Use Router for automatic failover
pipeline = (
    builder
    .with_router(model_list=[...])  # Groq + OpenAI fallback
    .with_redis_cache(...)  # Avoid duplicate calls
    .build()
)
```

## Tips

1. **Start Small**: Always test on a sample (10-100 rows) before processing large datasets
2. **Estimate First**: Use `estimate_cost()` to avoid surprise bills
3. **Use Batching**: Configure appropriate batch sizes (default: 100)
4. **Enable Checkpointing**: Process can resume from last checkpoint on failure
5. **Monitor Costs**: Watch the cost tracking during execution

## Need Help?

- Check the main README.md for full documentation
- See LLM_DATASET_ENGINE.md for architecture details
- Review code comments in each example
