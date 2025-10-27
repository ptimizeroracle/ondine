# Groq Provider

Configure and use Groq for ultra-fast inference with Ondine.

## Setup

```bash
export GROQ_API_KEY="gsk_..."
```

## Basic Usage

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Process: {text}")
    .with_llm(provider="groq", model="llama-3.3-70b-versatile")
    .build()
)

result = pipeline.execute()
```

## Available Models

- `llama-3.3-70b-versatile` - Best performance
- `llama-3.1-70b-versatile` - Fast and capable
- `mixtral-8x7b-32768` - Long context window

## Configuration Options

```python
.with_llm(
    provider="groq",
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=1000
)
```

## Performance

Groq is optimized for speed. Recommended concurrency: 30-50

## Related

- [OpenAI](openai.md)
- [Execution Modes](../execution-modes.md)

