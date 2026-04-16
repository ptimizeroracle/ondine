# Anthropic Claude Provider

Configure and use Anthropic Claude models with Ondine.

## Setup

```bash
export ANTHROPIC_API_KEY="sk-ant-..."  # pragma: allowlist secret
```

## Basic Usage

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["analysis"])
    .with_prompt("Analyze: {text}")
    .with_llm(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        temperature=0.0,
        max_tokens=1024
    )
    .build()
)

result = pipeline.execute()
```

## Available Models

- `claude-sonnet-4-20250514` - Most capable (recommended)
- `claude-opus-4.6` - Most capable, legacy

## Configuration Options

```python
.with_llm(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    temperature=0.7,
    max_tokens=4096,
    top_p=1.0
)
```

## Rate Limits

Recommended concurrency: 10-20

## Related

- [OpenAI](openai.md)
- [Cost Control](../cost-control.md)
