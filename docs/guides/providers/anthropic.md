# Anthropic Claude Provider

Configure and use Anthropic Claude models with Ondine.

## Setup

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
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
        model="claude-3-5-sonnet-20241022",
        temperature=0.0,
        max_tokens=1024
    )
    .build()
)

result = pipeline.execute()
```

## Available Models

- `claude-3-5-sonnet-20241022` - Most capable (recommended)
- `claude-3-5-haiku-20241022` - Fast and cost-effective
- `claude-3-opus-20240229` - Most capable, legacy

## Configuration Options

```python
.with_llm(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
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

