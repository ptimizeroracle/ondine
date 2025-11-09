<div align="center">
  <img src="https://raw.githubusercontent.com/ptimizeroracle/ondine/main/assets/images/ondine-logo.png" alt="Ondine Logo" width="600"/>

  # LLM Dataset Engine
</div>

[![Documentation](https://img.shields.io/badge/docs-MkDocs%20Material-blue.svg)](https://ptimizeroracle.github.io/ondine)
[![Tests](https://github.com/ptimizeroracle/Ondine/actions/workflows/ci.yml/badge.svg)](https://github.com/ptimizeroracle/Ondine/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ptimizeroracle/Ondine/branch/main/graph/badge.svg)](https://codecov.io/gh/ptimizeroracle/Ondine)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

SDK for batch processing tabular datasets with LLMs. Built on LlamaIndex for provider abstraction, adds batch orchestration, automatic cost tracking, checkpointing, and YAML configuration for dataset transformation at scale.

## Features

- **Quick API**: 3-line hello world with smart defaults and auto-detection
- **Simple API**: Fluent builder pattern for full control when needed
- **Reliability**: Automatic retries, checkpointing, error policies (99.9% completion rate)
- **Cost Control**: Pre-execution estimation, budget limits, real-time tracking
- **Observability**: Progress bars, structured logging, metrics, cost reports
- **Extensibility**: Plugin architecture, custom stages, multiple LLM providers
- **Fault Tolerant**: Zero data loss on crashes, resume from checkpoint
- **Multiple Providers**: OpenAI, Azure OpenAI, Anthropic Claude, Groq, MLX (Apple Silicon), and custom APIs
- **Local Inference**: Run models locally with MLX (Apple Silicon) or Ollama - 100% free, private, offline-capable
- **Multi-Column Processing**: Generate multiple output columns with composition or JSON parsing
- **Custom Providers**: Integrate any OpenAI-compatible API (Together.AI, vLLM, Ollama, custom endpoints)

## Quick Start

### Option 1: Quick API (Recommended)

The simplest way to get started - just provide your data, prompt, and model:

```python
from ondine import QuickPipeline

# Process data with smart defaults
pipeline = QuickPipeline.create(
    data="data.csv",
    prompt="Clean this text: {description}",
    model="gpt-4o-mini"
)

# Execute pipeline
result = pipeline.execute()
print(f"Processed {result.metrics.processed_rows} rows")
print(f"Total cost: ${result.costs.total_cost:.4f}")
```

**What's auto-detected:**
- Input columns from `{placeholders}` in prompt
- Provider from model name (gpt-4 → openai, claude → anthropic)
- Parser type (JSON for multi-column, text for single column)
- Sensible batch size and concurrency for the provider

### Option 2: Builder API (Full Control)

For advanced use cases requiring explicit configuration:

```python
from ondine import PipelineBuilder

# Build with explicit settings
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["description"],
              output_columns=["cleaned"])
    .with_prompt("Clean this text: {description}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_batch_size(100)
    .with_concurrency(5)
    .build()
)

# Estimate cost before running
estimate = pipeline.estimate_cost()
print(f"Estimated cost: ${estimate.total_cost:.4f}")

# Execute pipeline
result = pipeline.execute()
print(f"Total cost: ${result.costs.total_cost:.4f}")
```

## Installation

### Using uv (recommended)

```bash
# Basic installation
uv add ondine

# With MLX support (Apple Silicon only)
uv add "ondine[mlx]"
```

### Using pip

```bash
# Basic installation
pip install ondine

# With MLX support (Apple Silicon only)
pip install "ondine[mlx]"
```

### Set up API keys

```bash
# For cloud providers
export OPENAI_API_KEY="your-key-here"  # pragma: allowlist secret
# or
export AZURE_OPENAI_API_KEY="your-key-here"  # pragma: allowlist secret
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
# or
export ANTHROPIC_API_KEY="your-key-here"
# or
export GROQ_API_KEY="your-key-here"
# or
export TOGETHER_API_KEY="your-key-here"

# For MLX (Apple Silicon)
export HUGGING_FACE_HUB_TOKEN="your-token-here"  # For model downloads

# Local providers (Ollama, vLLM) don't need API keys
```

## Documentation

**Complete documentation is available at: https://ptimizeroracle.github.io/ondine**

The documentation includes:
- Installation and setup guides
- Quickstart tutorial (build your first pipeline in 5 minutes)
- Core concepts and architecture
- Execution modes (sync, async, streaming)
- Structured output with Pydantic
- Cost control and optimization
- Provider-specific guides
- Complete API reference (auto-generated from source)

## Usage Examples

### 1. Simple Data Processing

```python
from ondine import DatasetProcessor

# Minimal configuration for simple use cases
processor = DatasetProcessor(
    data="reviews.csv",
    input_column="customer_review",
    output_column="sentiment",
    prompt="Classify sentiment as: Positive, Negative, or Neutral\nReview: {customer_review}\nSentiment:",
    llm_config={"provider": "openai", "model": "gpt-4o-mini"}
)

# Test on sample first
sample = processor.run_sample(n=10)
print(sample)

# Process full dataset
result = processor.run()
```

### 2. Structured Data Extraction (JSON)

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_dataframe(
        df,
        input_columns=["product_description"],
        output_columns=["brand", "model", "price", "condition"]
    )
    .with_prompt("""
        Extract structured information and return JSON:
        {
          "brand": "...",
          "model": "...",
          "price": "...",
          "condition": "new|used|refurbished"
        }

        Description: {product_description}
    """)
    .with_llm(provider="openai", model="gpt-4o-mini", temperature=0.0)
    .build()
)

result = pipeline.execute()
```

### 3. Type-Safe Structured Output (Pydantic)

```python
from pydantic import BaseModel
from ondine import PipelineBuilder
from ondine.stages.response_parser_stage import PydanticParser

# Define your Pydantic model for type-safe validation
class ProductInfo(BaseModel):
    brand: str
    model: str
    price: float
    condition: str

pipeline = (
    PipelineBuilder.create()
    .from_dataframe(
        df,
        input_columns=["product_description"],
        output_columns=["brand", "model", "price", "condition"]
    )
    .with_prompt("""
        Extract product information and return JSON:
        {
          "brand": "manufacturer name",
          "model": "product model",
          "price": 999.99,
          "condition": "new|used|refurbished"
        }

        Description: {product_description}
    """)
    .with_llm(provider="openai", model="gpt-4o-mini", temperature=0.0)
    .with_parser(PydanticParser(ProductInfo, strict=True))  # Type-safe validation!
    .build()
)

result = pipeline.execute()
# Results are validated against ProductInfo model
```

### 4. With Cost Control

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("large_dataset.csv",
              input_columns=["text"],
              output_columns=["summary"])
    .with_prompt("Summarize in 10 words: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    # Cost control settings
    .with_max_budget(10.0)  # Maximum $10
    .with_batch_size(100)
    .with_concurrency(5)
    .with_rate_limit(60)  # 60 requests/min
    .with_checkpoint_interval(500)  # Checkpoint every 500 rows
    .build()
)

# Estimate first
estimate = pipeline.estimate_cost()
if estimate.total_cost > 10.0:
    print("Cost too high!")
    exit()

result = pipeline.execute()
```

### 5. Multiple Input Columns

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("products.csv",
              input_columns=["title", "description", "category"],
              output_columns=["optimized_title"])
    .with_prompt("""
        Optimize this product title for SEO.

        Current Title: {title}
        Description: {description}
        Category: {category}

        Optimized Title:
    """)
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_output("optimized_products.csv", format="csv")
    .build()
)

result = pipeline.execute()
```

### 6. Azure OpenAI

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Process: {text}")
    .with_llm(
        provider="azure_openai",
        model="gpt-4",
        azure_endpoint="https://your-endpoint.openai.azure.com/",
        azure_deployment="your-deployment-name",
        api_version="2024-02-15-preview"
    )
    .build()
)
```

### 7. Anthropic Claude

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["analysis"])
    .with_prompt("Analyze: {text}")
    .with_llm(
        provider="anthropic",
        model="claude-3-opus-20240229",
        temperature=0.0,
        max_tokens=1024
    )
    .build()
)
```

### 8. Local Inference with MLX (Apple Silicon)

```python
# 100% free, private, offline-capable inference on M1/M2/M3/M4 Macs
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["summary"])
    .with_prompt("Summarize: {text}")
    .with_llm(
        provider="mlx",
        model="mlx-community/Qwen3-1.7B-4bit",  # Fast, small model
        max_tokens=100,
        input_cost_per_1k_tokens=0.0,  # Free!
        output_cost_per_1k_tokens=0.0
    )
    .with_concurrency(1)  # MLX works best with concurrency=1
    .build()
)
```

**Requirements**:
- macOS with Apple Silicon (M1/M2/M3/M4)
- Install with: `pip install ondine[mlx]`

### 9. Provider Presets (Simplified Configuration)

```python
from ondine import PipelineBuilder
from ondine.core.specifications import LLMProviderPresets

# Use pre-configured providers (80% less boilerplate!)
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Process: {text}")
    .with_llm_spec(LLMProviderPresets.TOGETHER_AI_LLAMA_70B)  # One line!
    .build()
)

# Available presets:
# - LLMProviderPresets.GPT4O_MINI
# - LLMProviderPresets.GPT4O
# - LLMProviderPresets.TOGETHER_AI_LLAMA_70B
# - LLMProviderPresets.TOGETHER_AI_LLAMA_8B
# - LLMProviderPresets.OLLAMA_LLAMA_70B (free, local)
# - LLMProviderPresets.OLLAMA_LLAMA_8B (free, local)
# - LLMProviderPresets.GROQ_LLAMA_70B
# - LLMProviderPresets.CLAUDE_SONNET_4

# Override preset settings:
custom = LLMProviderPresets.GPT4O_MINI.model_copy(
    update={"temperature": 0.9, "max_tokens": 500}
)
pipeline.with_llm_spec(custom)

# Custom provider via factory:
custom_vllm = LLMProviderPresets.create_custom_openai_compatible(
    provider_name="My vLLM Server",
    model="mistral-7b-instruct",
    base_url="http://my-server:8000/v1"
)
pipeline.with_llm_spec(custom_vllm)
```

**Benefits**:
- Zero configuration errors (pre-validated)
- Correct pricing and URLs built-in
- IDE autocomplete for discovery
- 80% code reduction vs parameter-based config

### 10. Custom OpenAI-Compatible APIs (Parameter-Based)

```python
# Alternative: Configure providers with individual parameters
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Process: {text}")
    .with_llm(
        provider="openai_compatible",
        provider_name="Together.AI",  # Or "Ollama", "vLLM", etc.
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        base_url="https://api.together.xyz/v1",  # Custom endpoint
        api_key="${TOGETHER_API_KEY}",
        input_cost_per_1k_tokens=0.0006,
        output_cost_per_1k_tokens=0.0006
    )
    .build()
)
```

**Supported APIs**:
- **Ollama** (local): `http://localhost:11434/v1`
- **Together.AI** (cloud): `https://api.together.xyz/v1`
- **vLLM** (self-hosted): Your custom endpoint
- **Any OpenAI-compatible API**

### 11. Multi-Column Output with JSON Parsing

```python
# Single LLM call generates multiple output columns
pipeline = (
    PipelineBuilder.create()
    .from_csv("products.csv",
              input_columns=["description"],
              output_columns=["brand", "category", "price"])  # Multiple outputs!
    .with_prompt("""
        Extract structured data from this product description.
        Return JSON format:
        {
          "brand": "...",
          "category": "...",
          "price": "..."
        }

        Description: {description}
    """)
    .with_llm(provider="openai", model="gpt-4o-mini", temperature=0.0)
    .build()
)

result = pipeline.execute()
# Result has 3 new columns: brand, category, price
```

### 12. Pipeline Composition (Multi-Column with Dependencies)

```python
from ondine import PipelineComposer

# Create multiple pipelines with dependencies
composer = PipelineComposer(input_data=df)

# Pipeline 1: Generate sentiment score
sentiment_pipeline = (
    PipelineBuilder.create()
    .from_dataframe(df, input_columns=["review"], output_columns=["sentiment"])
    .with_prompt("Rate sentiment (0-100): {review}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .build()
)

# Pipeline 2: Generate explanation (depends on sentiment)
explanation_pipeline = (
    PipelineBuilder.create()
    .from_dataframe(df,
                    input_columns=["review", "sentiment"],
                    output_columns=["explanation"])
    .with_prompt("Explain why this review has {sentiment}% sentiment: {review}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .build()
)

# Compose and execute
result = (
    composer
    .add_column("sentiment", sentiment_pipeline)
    .add_column("explanation", explanation_pipeline, depends_on=["sentiment"])
    .execute()
)
```

## CLI Usage

Ondine includes a powerful command-line interface for processing datasets without writing code.

### List Available Providers

```bash
# See all supported LLM providers
ondine list-providers
```

This shows:
- Provider IDs (openai, azure_openai, anthropic, groq, mlx, openai_compatible)
- Platform requirements
- Cost estimates
- Use cases
- Required environment variables

### Process Datasets

```bash
# Basic usage
ondine process --config config.yaml

# Override input/output
ondine process --config config.yaml --input data.csv --output results.csv

# Override provider and model
ondine process --config config.yaml --provider groq --model llama-3.3-70b-versatile

# Set budget limit
ondine process --config config.yaml --max-budget 10.0

# Dry run (estimate only, don't execute)
ondine process --config config.yaml --dry-run

# Estimate cost
ondine estimate --config config.yaml --input data.csv

# Inspect data
ondine inspect --input data.csv --head 10
```

### Example Config File

```yaml
# config.yaml
dataset:
  source_type: csv
  source_path: data.csv
  input_columns: [text]
  output_columns: [sentiment]

prompt:
  template: "Classify sentiment: {text}"

llm:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.0

processing:
  batch_size: 100
  concurrency: 5
  max_budget: 10.0

output:
  destination_type: csv
  destination_path: output.csv
```

## Architecture

The SDK follows a **layered architecture**:

```
┌─────────────────────────────────────────┐
│  Layer 4: High-Level API                │
│  (Pipeline, PipelineBuilder)            │
├─────────────────────────────────────────┤
│  Layer 3: Orchestration Engine          │
│  (PipelineExecutor, StateManager)       │
├─────────────────────────────────────────┤
│  Layer 2: Processing Stages             │
│  (DataLoader, LLMInvocation, Parser)    │
├─────────────────────────────────────────┤
│  Layer 1: Infrastructure Adapters       │
│  (LLMClient, DataReader, Checkpoint)    │
├─────────────────────────────────────────┤
│  Layer 0: Core Utilities                │
│  (RetryHandler, RateLimiter, Logging)   │
└─────────────────────────────────────────┘
```

### Key Design Principles

- **Simple**: Straightforward solutions
- **DRY**: No code duplication
- **Type Safe**: Type hints throughout
- **Separation of Concerns**: Configuration vs. execution

## Supported LLM Providers

| Provider | Platform | Cost | Use Case | Setup |
|----------|----------|------|----------|-------|
| **OpenAI** | Cloud (All) | $$ | Production, high quality | `OPENAI_API_KEY` |
| **Azure OpenAI** | Cloud (All) | $$ | Enterprise, compliance, **Managed Identity support** | `AZURE_OPENAI_API_KEY` or Managed Identity |
| **Anthropic** | Cloud (All) | $$$ | Long context, Claude models | `ANTHROPIC_API_KEY` |
| **Groq** | Cloud (All) | Free tier | Fast inference, development | `GROQ_API_KEY` |
| **MLX** | macOS (M1/M2/M3/M4) | Free | Local, private, offline | `pip install ondine[mlx]` |
| **OpenAI-Compatible** | Custom/Local/Cloud | Varies | Ollama, vLLM, Together.AI | `base_url` + optional API key |

Run `ondine list-providers` to see detailed information about each provider.

## Use Cases

- **Data Cleaning**: Clean, normalize, standardize text data
- **Sentiment Analysis**: Classify sentiment at scale
- **Information Extraction**: Extract structured data from unstructured text
- **Categorization**: Auto-categorize products, documents, emails
- **Content Generation**: Generate descriptions, summaries, titles
- **Translation**: Translate content to multiple languages
- **Data Enrichment**: Enhance datasets with LLM-generated insights
- **Product Matching**: Compare and score product similarity
- **Content Moderation**: Flag inappropriate content at scale

## Performance

- **Throughput**: Process 1,000 rows in < 5 minutes (GPT-4o-mini, concurrency=5)
- **Reliability**: 99.9% completion rate with automatic retries
- **Cost Efficiency**: Pre-execution estimation within 10% accuracy
- **Memory**: < 500MB for datasets up to 50K rows

## Observability & Debugging

Ondine leverages **LlamaIndex's built-in observability** for automatic instrumentation of all LLM calls. Add observability with a single line:

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Process: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    # Add observability - automatically tracks ALL LLM calls!
    .with_observer("langfuse", config={
        "public_key": "pk-lf-...",
        "secret_key": "sk-lf-..."  # pragma: allowlist secret
    })
    .build()
)

result = pipeline.execute()
```

### Supported Observers

**Langfuse** - LLM-specific observability (recommended):
```python
.with_observer("langfuse", config={
    "public_key": "pk-lf-...",
    "secret_key": "sk-lf-...",  # pragma: allowlist secret
    "host": "https://cloud.langfuse.com"  # optional
})
```
Tracks: prompts, completions, tokens, costs, latency, model info

**OpenTelemetry** - Infrastructure monitoring:
```python
.with_observer("opentelemetry", config={})
```
Tracks: spans, traces, metrics - works with Jaeger, Datadog, Grafana

**Logging** - Simple console output:
```python
.with_observer("logging", config={})
```
Tracks: basic LLM call logs to console

**Multiple observers simultaneously**:
```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", ...)
    .with_prompt("...")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_observer("langfuse", config={...})
    .with_observer("opentelemetry", config={...})
    .with_observer("logging", config={})
    .build()
)
```

### What's Tracked Automatically

**Powered by LlamaIndex instrumentation:**
- Full prompt and completion text
- Token usage (input, output, total)
- Cost per call
- Latency metrics
- Model and provider information
- Future: RAG retrieval when we add it

### Examples

See complete examples:
- `examples/15_observability_logging.py` - Simple console logging
- `examples/16_observability_opentelemetry.py` - OpenTelemetry + Jaeger
- `examples/17_observability_langfuse.py` - Langfuse integration
- `examples/18_observability_multi.py` - Multiple observers

### Setup Langfuse (Recommended for LLM Observability)

1. Sign up at https://cloud.langfuse.com (free tier available)
2. Get your API keys
3. Add to your pipeline:
```python
.with_observer("langfuse", config={
    "public_key": "pk-lf-...",
    "secret_key": "sk-lf-..."  # pragma: allowlist secret
})
```
4. View detailed traces in Langfuse dashboard

## Configuration Options

### Execution Modes

**Standard Execution** (default)
```python
pipeline = PipelineBuilder.create().from_csv(...).build()
result = pipeline.execute()
```
Use when: Dataset fits in memory (< 50K rows typical), straightforward processing.

**Async Execution** (concurrent processing)
```python
pipeline = (
    PipelineBuilder.create()
    .from_csv(...)
    .with_async_execution(max_concurrency=10)
    .build()
)
result = await pipeline.execute_async()
```
Use when: Need high throughput, LLM API supports async, running in async context (FastAPI, aiohttp).

**Streaming Execution** (memory-efficient)
```python
pipeline = (
    PipelineBuilder.create()
    .from_csv(...)
    .with_streaming(chunk_size=1000)
    .build()
)
for chunk_result in pipeline.execute_stream():
    # Process each chunk as it completes
    chunk_result.data.to_csv("output_chunk.csv", mode="a")
```
Use when: Large datasets (100K+ rows), limited memory, need constant memory footprint, early results desired.

**When NOT to use streaming:**
- Dataset under 50K rows (overhead not justified)
- Need entire dataset in memory for post-processing
- Pipeline has dependencies between rows

See `examples/08_streaming_large_files.py` for detailed streaming example.

### Processing Configuration

```python
.with_batch_size(100)          # Rows per batch
.with_concurrency(5)            # Parallel requests
.with_checkpoint_interval(500)  # Checkpoint frequency
.with_rate_limit(60)            # Requests per minute
.with_max_budget(10.0)          # Maximum USD budget
```

### LLM Configuration

```python
.with_llm(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.0,        # 0.0-2.0
    max_tokens=1024,        # Max output tokens
    api_key="..."           # Or from env
)
```

### Output Configuration

```python
.with_output(
    path="output.csv",
    format="csv",              # csv, excel, parquet
    merge_strategy="replace"   # replace, append, update
)
```

## Testing

```bash
# Run tests
uv run pytest

# With coverage
uv run pytest --cov=src --cov-report=html

# Run specific test
uv run pytest tests/test_pipeline.py
```

## Documentation

**Full documentation**: https://ptimizeroracle.github.io/ondine

Additional resources:
- **README.md** (this file): Quick start and usage guide
- **examples/**: Example scripts demonstrating various features
- **Code docstrings**: Inline documentation for all public APIs

## Contributing

Contributions welcome! Please follow:

1. Fork the repository at https://github.com/ptimizeroracle/Ondine
2. Create a feature branch
3. Follow the existing code style (Black, Ruff)
4. Add tests for new features
5. Update documentation
6. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [LlamaIndex](https://www.llamaindex.ai/) for LLM provider abstraction and observability
- Ondine leverages:
  - **LlamaIndex LLM clients** for OpenAI, Anthropic, Groq, Azure
  - **LlamaIndex observability** for automatic instrumentation of LLM calls
  - Ondine adds: batch processing, cost tracking, checkpointing, YAML configuration, and dataset orchestration
- Thanks to the open-source community

## Support

- **Repository**: https://github.com/ptimizeroracle/Ondine
- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Email**: git@binblok.com

## Recent Updates

### Version 1.0.0 (October 2025)

**New Features:**
- **Provider Presets**: Pre-configured LLMSpec objects for common providers (80% code reduction)
- **Simplified Configuration**: New `with_llm_spec()` method accepting LLMSpec objects
- **MLX Integration**: Local inference on Apple Silicon (M1/M2/M3/M4) - 100% free, private, offline
- **OpenAI-Compatible Provider**: Support for Ollama, vLLM, Together.AI, and custom APIs
- **Multi-Column Processing**: Generate multiple output columns with JSON parsing
- **Pipeline Composition**: Chain pipelines with dependencies between columns
- **CLI Provider Discovery**: `ondine list-providers` command to explore all providers
- **Auto-Retry for Multi-Column**: Automatic retry now checks all output columns for failures
- **Custom LLM Clients**: Extend `LLMClient` base class for exotic APIs

**Improvements:**
- Zero configuration errors with validated presets
- Enhanced error handling for multi-column outputs
- Better streaming implementation
- Improved documentation with provider comparison guide
- More examples (14+ example files including provider presets demo)

## Roadmap

### Upcoming Features

**RAG Integration (Next Release)**
- Retrieval-Augmented Generation for context-aware dataset processing
- Custom retrieval stage via plugin architecture
- Vector store integration (Pinecone, Weaviate, ChromaDB)
- Dynamic context injection per row
- See `docs/DESIGN_IMPROVEMENT.md` for detailed design exploration

**Other Planned Features**
- Support for true streaming execution (in progress)
- Multi-modal support (images, PDFs)
- Distributed processing (Spark integration)
- Web UI for pipeline management
- Additional LLM providers (Cohere, AI21, Mistral)

---

Built with Python and LlamaIndex
