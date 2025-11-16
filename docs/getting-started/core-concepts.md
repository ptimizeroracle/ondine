# Core Concepts

Understanding Ondine's architecture will help you build more sophisticated pipelines and debug issues effectively.

## Architecture Overview

Ondine is built on a layered architecture:

```
┌─────────────────────────────────────────┐
│   High-Level APIs (QuickPipeline)      │  User-friendly interfaces
├─────────────────────────────────────────┤
│   Pipeline Builder & Configuration     │  Fluent API, YAML config
├─────────────────────────────────────────┤
│   Pipeline Orchestration               │  Execution strategies
├─────────────────────────────────────────┤
│   Pipeline Stages                      │  Composable processing units
├─────────────────────────────────────────┤
│   Adapters (LLM, Storage, IO)         │  External integrations
└─────────────────────────────────────────┘
```

## Key Components

### 1. Pipeline

The `Pipeline` is the central execution unit. It orchestrates the flow of data through stages.

```python
from ondine import Pipeline

# Pipelines are built via PipelineBuilder
pipeline = PipelineBuilder.create()...build()

# Execute synchronously
result = pipeline.execute()

# Execute asynchronously
result = await pipeline.execute_async()
```

**Key characteristics:**
- Immutable once built (thread-safe)
- Encapsulates all configuration
- Handles checkpointing and recovery
- Tracks costs and metrics

### 2. Pipeline Stages

Ondine processes data through a series of composable stages:

1. **DataLoaderStage** - Load data from CSV, Excel, Parquet, or DataFrame
2. **PromptFormatterStage** - Format prompts with row data
3. **BatchAggregatorStage** - (Optional) Aggregate N prompts into 1 for multi-row batching
4. **LLMInvocationStage** - Call LLM API with retry and rate limiting
5. **BatchDisaggregatorStage** - (Optional) Split batch response into N results
6. **ResponseParserStage** - Parse LLM responses (text, JSON, regex)
7. **ResultWriterStage** - Write results to output

**Multi-Row Batching** (NEW):
- Stages 3 and 5 are only inserted when `batch_size > 1`
- Enables 100× speedup by processing N rows per API call
- Automatic context window validation
- Partial failure handling

### 3. Pipeline Builder

The `PipelineBuilder` provides a fluent API for constructing pipelines:

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    # Data source
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    
    # Prompt configuration
    .with_prompt("Process: {text}")
    
    # LLM configuration
    .with_llm(provider="openai", model="gpt-4o-mini")
    
    # Processing configuration
    .with_batch_size(100)
    .with_concurrency(5)
    .with_retry_policy(max_retries=3)
    
    # Build immutable pipeline
    .build()
)
```

**Builder methods:**
- Data: `from_csv()`, `from_dataframe()`, `from_parquet()`, `from_excel()`
- Prompt: `with_prompt()`, `with_system_prompt()`
- LLM: `with_llm()`, `with_llm_spec()`
- Processing: `with_batch_size()`, `with_concurrency()`, `with_rate_limit()`
- Reliability: `with_retry_policy()`, `with_checkpoint()`
- Cost: `with_max_budget()`
- Execution: `with_async_execution()`, `with_streaming()`

### 3. Pipeline Stages

Stages are composable processing units that form a pipeline:

```
┌────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│   Data     │───▶│   Prompt    │───▶│     LLM      │───▶│   Response   │
│   Loader   │    │  Formatter  │    │  Invocation  │    │    Parser    │
└────────────┘    └─────────────┘    └──────────────┘    └──────────────┘
```

**Built-in stages:**
- `DataLoaderStage`: Load data from files/dataframes
- `PromptFormatterStage`: Format prompts with variables
- `LLMInvocationStage`: Call LLM APIs
- `ResponseParserStage`: Parse and validate responses
- `ResultWriterStage`: Write results to storage

**Custom stages:**
You can create custom stages by extending `PipelineStage`:

```python
from ondine.stages import PipelineStage

class MyCustomStage(PipelineStage):
    def process(self, input_data, context):
        # Your processing logic
        return processed_data
    
    def validate_input(self, input_data):
        # Validation logic
        return ValidationResult(valid=True)
```

### 4. Specifications

Specifications are Pydantic models that define configuration:

#### DatasetSpec

Defines input data configuration:

```python
from ondine.core.specifications import DatasetSpec

spec = DatasetSpec(
    source="data.csv",
    input_columns=["text"],
    output_columns=["result"],
    format="csv"
)
```

#### PromptSpec

Defines prompt templates:

```python
from ondine.core.specifications import PromptSpec

spec = PromptSpec(
    template="Summarize: {text}",
    system_prompt="You are a helpful assistant."
)
```

#### LLMSpec

Defines LLM provider configuration:

```python
from ondine.core.specifications import LLMSpec

spec = LLMSpec(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000,
    api_key="sk-..."  # Or use environment variable
)
```

#### ProcessingSpec

Defines execution configuration:

```python
from ondine.core.specifications import ProcessingSpec

spec = ProcessingSpec(
    batch_size=100,
    concurrency=5,
    max_retries=3,
    checkpoint_interval=500,
    rate_limit=60  # requests per minute
)
```

### 5. Execution Strategies

Ondine supports multiple execution modes:

#### Synchronous (Default)

Single-threaded, sequential processing:

```python
result = pipeline.execute()
```

**Use when:** Dataset fits in memory, simplicity is priority.

#### Asynchronous

Concurrent processing with async/await:

```python
pipeline = (
    PipelineBuilder.create()
    ...
    .with_async_execution(max_concurrency=10)
    .build()
)

result = await pipeline.execute_async()
```

**Use when:** Need high throughput, LLM API supports async.

#### Streaming

Memory-efficient processing for large datasets:

```python
pipeline = (
    PipelineBuilder.create()
    ...
    .with_streaming(chunk_size=1000)
    .build()
)

result = pipeline.execute()
```

**Use when:** Dataset is large (100K+ rows), memory is limited.

See [Execution Modes Guide](../guides/execution-modes.md) for detailed comparison.

### 6. Adapters

Adapters abstract external dependencies:

#### LLM Client

Adapts different LLM providers to a common interface:

```python
from ondine.adapters import LLMClient

# Automatically selected based on provider
client = create_llm_client(llm_spec)
response = client.complete(prompt, temperature=0.7)
```

**Supported providers:**
- OpenAI
- Azure OpenAI
- Anthropic Claude
- Groq
- MLX (local Apple Silicon)
- Custom OpenAI-compatible APIs

#### Storage

Handles checkpoint persistence:

```python
from ondine.adapters import CheckpointStorage

storage = CheckpointStorage(path="./checkpoints")
storage.save(state)
state = storage.load()
```

#### Data IO

Handles various data formats:

```python
from ondine.adapters import DataIO

# Supports CSV, Parquet, Excel, JSON
data = DataIO.read("data.csv")
DataIO.write(data, "output.parquet")
```

## Execution Flow

Here's what happens when you call `pipeline.execute()`:

1. **Validation**: Validate configuration and input data
2. **Cost Estimation**: Calculate expected cost and token usage
3. **Checkpoint Check**: Look for existing checkpoint to resume
4. **Data Loading**: Load input data (streaming or in-memory)
5. **Prompt Formatting**: Format prompts with input variables
6. **LLM Invocation**: Call LLM API with rate limiting and retries
7. **Response Parsing**: Parse and validate LLM responses
8. **Result Writing**: Write results to output (file or DataFrame)
9. **Metrics Collection**: Aggregate costs, tokens, timing
10. **Checkpoint Cleanup**: Remove checkpoint on successful completion

## Error Handling

Ondine provides robust error handling:

### Automatic Retries

Failed requests are automatically retried with exponential backoff:

```python
.with_retry_policy(
    max_retries=3,
    backoff_factor=2.0,
    retry_on=[RateLimitError, NetworkError]
)
```

### Checkpointing

Long-running jobs can be resumed on failure:

```python
.with_checkpoint("./checkpoints", interval=100)
```

### Error Policies

Control how errors are handled:

```python
.with_error_policy("continue")  # Continue on errors
.with_error_policy("stop")      # Stop on first error
```

## Cost Tracking

Ondine tracks costs in real-time:

```python
result = pipeline.execute()

print(f"Total cost: ${result.costs.total_cost:.4f}")
print(f"Input tokens: {result.costs.input_tokens}")
print(f"Output tokens: {result.costs.output_tokens}")
print(f"Cost per row: ${result.costs.total_cost / result.metrics.processed_rows:.6f}")
```

### Budget Control

Set maximum budget limits:

```python
from decimal import Decimal

pipeline = (
    PipelineBuilder.create()
    ...
    .with_max_budget(Decimal("10.0"))  # Max $10 USD
    .build()
)

# Execution stops if budget exceeded
result = pipeline.execute()
```

## Observability

Monitor pipeline execution:

### Progress Bars

Automatic progress tracking with tqdm:

```
Processing: 100%|████████| 1000/1000 [00:45<00:00, 22.1rows/s]
```

### Structured Logging

JSON-formatted logs with structlog:

```python
from ondine.utils import configure_logging

configure_logging(level="INFO", json_format=True)
```

### Metrics Export

Export metrics to Prometheus:

```python
from ondine.utils import MetricsExporter

exporter = MetricsExporter(port=9090)
exporter.start()
```

## Next Steps

- [Execution Modes](../guides/execution-modes.md) - Choose the right execution strategy
- [Structured Output](../guides/structured-output.md) - Type-safe response parsing
- [Cost Control](../guides/cost-control.md) - Optimize costs and set budgets
- [API Reference](../api/index.md) - Detailed API documentation

