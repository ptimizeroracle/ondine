# Groq Provider

Configure and use Groq for ultra-fast inference with Ondine.

## Setup

```bash
export GROQ_API_KEY="gsk_..."  # pragma: allowlist secret
```

## Basic Usage

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Process: {text}")
    .with_llm(provider="groq", model="openai/gpt-oss-120b")
    .build()
)

result = pipeline.execute()
```

## Available Models

- `openai/gpt-oss-120b` - Best quality
- `openai/gpt-oss-20b` - Fastest and cheapest

Groq names models by the vendor that made them, so the `openai/` prefix is part
of the model name. Keep `provider="groq"`: it is what routes the call to Groq.
Groq has retired its Llama and Mixtral models (`llama-3.3-70b-versatile`,
`llama-3.1-70b-versatile`, `mixtral-8x7b-32768`); requests for them fail with
"model does not exist". See Groq's model list for what is currently served.

## Configuration Options

```python
.with_llm(
    provider="groq",
    model="openai/gpt-oss-120b",
    temperature=0.7,
    max_tokens=1000
)
```

## Performance

Groq is optimized for speed. Recommended concurrency: 30-50

## Related

- [OpenAI](openai.md)
- [Execution Modes](../execution-modes.md)
