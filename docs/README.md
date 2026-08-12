# Ondine

[![PyPI version](https://img.shields.io/pypi/v/ondine.svg)](https://pypi.org/project/ondine/)
[![Downloads](https://static.pepy.tech/badge/ondine/month)](https://pepy.tech/project/ondine)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/ptimizeroracle/ondine.svg?style=social)](https://github.com/ptimizeroracle/ondine)
[![Tests](https://github.com/ptimizeroracle/ondine/actions/workflows/ci.yml/badge.svg)](https://github.com/ptimizeroracle/ondine/actions/workflows/ci.yml)

**Batch-process your DataFrames with LLMs, without the boilerplate.**

Agents reason row-by-row. Ondine computes columns.

In a [measured benchmark](https://github.com/ptimizeroracle/ondine/blob/main/benchmarks/RESULTS.md) processing 100,000 rows (DeepSeek Chat, batch size 15, projected from a 30-row real-API sample), Ondine made ~15x fewer API calls than a naive per-row loop (6,666 vs 100,000) and cost ~35% less ($0.4815 vs $0.7411).

Ondine is an open-source Python SDK for data engineers and ML practitioners who need to process large CSV/DataFrame datasets with LLMs. Multi-row batching, prefix caching, and budget controls are built in from day one, not bolted on later.

## Quick Start

```python
from ondine import enrich

df = enrich(
    "products.csv",
    "Classify this product into a category: {name} - {description}",
    output_columns=["category"],
    model="gpt-4o-mini",
)
```

## Why Ondine?

Most teams processing tabular data with LLMs hit the same three walls: **API cost spirals**, **brittle pipelines that crash halfway through**, and **hallucinated outputs that are hard to catch at scale**. Ondine is built to solve exactly those three problems.

| Problem | Ondine's answer |
|---|---|
| API bills too high | Multi-row batching (100 rows/call) + prefix caching -- ~35% lower cost in our [benchmark](https://github.com/ptimizeroracle/ondine/blob/main/benchmarks/RESULTS.md) |
| Pipeline crashes lose progress | Checkpointing -- resume from last saved row, not row 0 |
| LLM makes things up | Context Store -- ground each response against your source data |
| Too many providers to juggle | 100+ providers via LiteLLM, single unified API |

## Key Features

- **`enrich()`** -- one-call front door; input type in, same type out
- **`plan()`** -- draft a pipeline from a plain-language goal, inspect and approve before execution
- **Builder API** -- Full control over pipeline configuration
- **MCP server** (`ondine-mcp`) -- expose pipeline runs as MCP tools for agent clients, budget-capped by design
- **Provider Batch API mode** -- OpenAI/Anthropic native Batch jobs for ~50% provider-advertised savings
- **100+ LLM providers** via LiteLLM (OpenAI, Anthropic, Groq, Azure, local MLX)
- **Multi-row batching** -- Process 100 rows per API call
- **Prefix caching** -- reduces token cost on repeated prompts
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
