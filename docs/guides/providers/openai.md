# OpenAI Provider

Configure and use OpenAI models with Ondine.

## Setup

```bash
export OPENAI_API_KEY="sk-..."
```

## Basic Usage

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Process: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .build()
)

result = pipeline.execute()
```

## Available Models

- `gpt-4o` - Most capable, balanced performance
- `gpt-4o-mini` - Fast and cost-effective (recommended)
- `gpt-4-turbo` - Advanced reasoning
- `gpt-3.5-turbo` - Legacy, cost-effective

## Configuration Options

```python
.with_llm(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.7,      # 0.0-2.0
    max_tokens=1000,      # Max response length
    top_p=1.0,            # Nucleus sampling
    frequency_penalty=0.0,  # -2.0 to 2.0
    presence_penalty=0.0    # -2.0 to 2.0
)
```

## Rate Limits

See [OpenAI Rate Limits](https://platform.openai.com/docs/guides/rate-limits) for your tier.

Recommended concurrency:
- Tier 1: 10-20
- Tier 4+: 50-100

## Related

- [Azure OpenAI](azure.md)
- [Cost Control](../cost-control.md)

