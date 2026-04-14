# Ondine

**Run LLMs on tabular data — 100x fewer API calls, 40-50% lower cost.**

Ondine is an open-source Python SDK for data engineers and ML practitioners who need to process large CSV/DataFrame datasets with LLMs. Multi-row batching, prefix caching, and budget controls are built in from day one — not bolted on later.

## Quick Start

```python
from ondine import QuickPipeline

result = QuickPipeline.create(
    data="products.csv",
    prompt="Classify this product into a category: {name} - {description}",
    model="gpt-4o-mini"
)
```

## Why Ondine?

Most teams processing tabular data with LLMs hit the same three walls: **API cost spirals**, **brittle pipelines that crash halfway through**, and **hallucinated outputs that are hard to catch at scale**. Ondine is built to solve exactly those three problems.

| Problem | Ondine's answer |
|---|---|
| API bills too high | Multi-row batching (100 rows/call) + prefix caching → 40-50% savings |
| Pipeline crashes lose progress | Checkpointing — resume from last saved row, not row 0 |
| LLM makes things up | Context Store — ground each response against your source data |
| Too many providers to juggle | 100+ providers via LiteLLM, single unified API |

## Key Features

- **Quick API** — 3-line hello world with smart defaults
- **Builder API** — Full control over pipeline configuration
- **100+ LLM providers** via LiteLLM (OpenAI, Anthropic, Groq, Azure, local MLX)
- **Multi-row batching** — Process 100 rows per API call
- **Prefix caching** — 40-50% cost reduction on repeated prompts
- **Cost estimation** — Know the cost before you run
- **Budget limits** — Hard caps on spending
- **Checkpointing** — Resume failed pipelines from where they stopped
- **Structured output** — Pydantic models, JSON, regex parsing
- **Observability** — OpenTelemetry, Langfuse, structured logging

## Installation

```bash
pip install ondine
```

## Links

- [GitHub Repository](https://github.com/ptimizeroracle/ondine)
- [PyPI Package](https://pypi.org/project/ondine/)
- [Contributing Guide](contributing.md)
