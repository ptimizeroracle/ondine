# enrich() -- One-Call Enrichment

`ondine.enrich()` is the front door: give it data, a prompt, and the columns you want back, get an enriched DataFrame. No builder chain, no explicit `.execute()`.

```python
from ondine import enrich

df = enrich(
    "reviews.csv",
    "Classify the tone of: {review}",
    output_columns=["sentiment"],
    model="gpt-4o-mini",
    budget=5.0,
)
```

It's a thin facade over `QuickPipeline` -- no logic of its own. Every default (batching, retries, adaptive concurrency, checkpointing) comes from `QuickPipeline` / `PipelineBuilder` underneath, so `enrich()` and the builder chain never drift apart in behavior.

## At a Glance

| | `enrich()` | Builder chain |
|---|---|---|
| Lines to a result | 1 call | 4-8 chained calls |
| Column naming, temperature, parser choice | Auto | Explicit |
| Structured output | `schema=` kwarg | `.with_structured_output()` |
| Best for | Notebooks, scripts, first pass | Production configs, fine-grained control |

Reach for the [Builder API](../getting-started/quickstart.md#4-builder-api-more-control) when you need to name output columns per-column, tune retry policy, or wire checkpointing/observability explicitly.

## Signature

```python
def enrich(
    data: str | Path | pd.DataFrame | pl.DataFrame,
    prompt: str,
    output_columns: list[str] | str | None = None,
    *,
    model: str = "gpt-4o-mini",
    budget: float | Decimal | str | None = None,
    schema: Any | None = None,
    **options: Any,
) -> pd.DataFrame | pl.DataFrame
```

| Arg | Meaning |
|---|---|
| `data` | CSV/Excel/Parquet/JSON path, or an in-memory pandas or Polars DataFrame. |
| `prompt` | Template with `{column}` placeholders for each input column. |
| `output_columns` | Column(s) to populate. Defaults to `["output"]`. |
| `model` | LiteLLM model string (`"gpt-4o-mini"`, `"claude-3-5-sonnet"`, `"ollama/qwen3"`, ...). Provider is auto-detected from the name. |
| `budget` | Hard USD cap for the run; the pipeline halts at the limit. |
| `schema` | Optional Pydantic model for structured output. |
| `**options` | Forwarded to `QuickPipeline` -- see below. |

## Input Type Is Preserved

The return type mirrors the input:

```python
import pandas as pd
import polars as pl

# pandas in -> pandas out
enrich(pd.DataFrame({"review": ["great product"]}), "Classify: {review}")

# polars in -> polars out
enrich(pl.DataFrame({"review": ["great product"]}), "Classify: {review}")

# CSV path in -> pandas out (QuickPipeline's native format)
enrich("reviews.csv", "Classify: {review}")
```

## Structured Output

Pass a Pydantic model via `schema=` and `enrich()` configures structured output on the pipeline **and** auto-injects a `JSONParser` -- you don't wire either piece by hand:

```python
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str
    score: float

df = enrich(
    "reviews.csv",
    "Classify: {review}",
    output_columns=["label", "score"],
    schema=Sentiment,
)
```

Without `schema=`, no structured-output metadata or parser is added -- the raw text response lands in `output_columns` as-is.

## Allowed Options

`**options` is an explicit allowlist, not a `**kwargs` passthrough: `provider`, `temperature`, `max_tokens`, `batch_size`, `concurrency`. Anything else -- including a typo, or an internal name like `max_budget` -- raises `TypeError` instead of being silently swallowed:

```python
enrich(df, "Classify: {review}", bogus=True)
# TypeError: enrich() got unexpected option(s): ['bogus']. Allowed: [...]
```

```python
df = enrich(
    "reviews.csv",
    "Classify: {review}",
    provider="groq",
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    batch_size=25,
    concurrency=10,
)
```

## Related

- [5-Minute Quickstart](../getting-started/quickstart.md) -- `enrich()` as the opening example
- [Structured Output (Pydantic)](structured-output.md) -- the parser mechanics behind `schema=`
- [Cost Estimation & Budgets](cost-control.md) -- what `budget=` enforces under the hood
