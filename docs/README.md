# Ondine

**SDK for batch processing tabular datasets with LLMs.**

For data engineers and ML practitioners who need to process millions of tabular rows with LLMs, Ondine delivers **100x fewer API calls** via multi-row batching and **40-50% cost reduction** via prefix caching -- with cost estimation, budget limits, checkpointing, and 100+ provider support built in.

## Quick Start

```python
from ondine import QuickPipeline

result = QuickPipeline.create(
    data="products.csv",
    prompt="Classify this product into a category: {name} - {description}",
    model="gpt-4o-mini"
)
```

## Key Features

- **Quick API** -- 3-line hello world with smart defaults
- **Builder API** -- Full control over pipeline configuration
- **100+ LLM providers** via LiteLLM (OpenAI, Anthropic, Groq, Azure, local MLX)
- **Multi-row batching** -- Process 100 rows per API call
- **Prefix caching** -- 40-50% cost reduction on repeated prompts
- **Cost estimation** -- Know the cost before you run
- **Budget limits** -- Hard caps on spending
- **Checkpointing** -- Resume failed pipelines from where they stopped
- **Structured output** -- Pydantic models, JSON, regex parsing
- **Observability** -- OpenTelemetry, Langfuse, structured logging

## Installation

```bash
pip install ondine
```

## Links

- [GitHub Repository](https://github.com/ptimizeroracle/ondine)
- [PyPI Package](https://pypi.org/project/ondine/)
- [Contributing Guide](contributing.md)
