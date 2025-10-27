# LLM Dataset Engine - Examples

This directory contains example scripts demonstrating various features of the LLM Dataset Engine.

## Prerequisites

1. Install the package:
   ```bash
   uv add llm-dataset-engine
   ```

2. Set up your API keys:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   # or
   export AZURE_OPENAI_API_KEY="your-key-here"
   # or
   export ANTHROPIC_API_KEY="your-key-here"
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
**Extracting structured data with JSON**

Demonstrates:
- JSON response parsing
- Multi-column output
- Information extraction use case
- Exporting results

Run:
```bash
python examples/03_structured_output.py
```

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

### Pattern 3: Progress Monitoring

```python
# Progress bar and logging are enabled by default
# Add custom observers if needed
from ondine.orchestration import CostTrackingObserver

pipeline.add_observer(CostTrackingObserver(warning_threshold=0.8))
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
