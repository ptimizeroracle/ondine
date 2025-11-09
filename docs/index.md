<div align="center">
  <img src="assets/images/ondine-logo.png" alt="Ondine Logo" width="600"/>
</div>

[![Tests](https://github.com/ptimizeroracle/Ondine/actions/workflows/ci.yml/badge.svg)](https://github.com/ptimizeroracle/Ondine/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ptimizeroracle/Ondine/branch/main/graph/badge.svg)](https://codecov.io/gh/ptimizeroracle/Ondine)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SDK for batch processing tabular datasets with LLMs. Built on LlamaIndex for provider abstraction, adds batch orchestration, automatic cost tracking, checkpointing, and YAML configuration for dataset transformation at scale.

## Features

- **Quick API**: 3-line hello world with smart defaults and auto-detection
- **Simple API**: Fluent builder pattern for full control when needed
- **Reliability**: Automatic retries, checkpointing, error policies (99.9% completion rate)
- **Cost Control**: Pre-execution estimation, budget limits, real-time tracking
- **Observability**: LlamaIndex-powered automatic LLM tracking (Langfuse, OpenTelemetry), progress bars, cost reports
- **Extensibility**: Plugin architecture, custom stages, multiple LLM providers
- **Production Ready**: Zero data loss on crashes, resume from checkpoint
- **Multiple Providers**: OpenAI, Azure OpenAI (with Managed Identity), Anthropic Claude, Groq, MLX (Apple Silicon), and custom APIs
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

Install with pip or uv:

```bash
pip install ondine
```

Or with optional dependencies:

```bash
# For Apple Silicon local inference
pip install ondine[mlx]

# For Azure Managed Identity (keyless auth)
pip install ondine[azure]

# Observability is now built-in (OpenTelemetry + Langfuse)
# No separate install needed!

# For development
pip install ondine[dev]
```

## Next Steps

- [Installation Guide](getting-started/installation.md) - Detailed installation instructions
- [Quickstart](getting-started/quickstart.md) - Your first pipeline in 5 minutes
- [Core Concepts](getting-started/core-concepts.md) - Understanding pipelines, stages, and specifications
- [Execution Modes](guides/execution-modes.md) - When to use sync, async, or streaming
- [API Reference](api/index.md) - Complete API documentation

## Use Cases

Ondine excels at:

- Data cleaning and normalization (PII detection, standardization)
- Content enrichment (classification, tagging, summarization)
- Extraction tasks (structured data from unstructured text)
- Translation and localization at scale
- Synthetic data generation with cost controls
- Quality assurance (validation, scoring, feedback)

## Why Ondine?

- **Reliable**: Checkpointing, auto-retry, budget controls, observability
- **Developer-Friendly**: Fluent API, YAML config, CLI tools, extensive examples
- **Cost-Aware**: Pre-run estimation, real-time tracking, budget limits
- **Flexible**: Multiple providers, custom stages, extensible architecture
- **Well-Tested**: 95%+ code coverage, integration tests with real APIs

## License

MIT License - see [LICENSE](https://github.com/ptimizeroracle/ondine/blob/main/LICENSE) for details.
