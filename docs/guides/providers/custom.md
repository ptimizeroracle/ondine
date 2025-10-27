# Custom OpenAI-Compatible APIs

Integrate any OpenAI-compatible API with Ondine, including Together.AI, vLLM, Ollama, and custom endpoints.

## Basic Usage

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Process: {text}")
    .with_llm(
        provider="openai",  # Use openai provider
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        api_base="https://api.together.xyz/v1",  # Custom endpoint
        api_key=os.getenv("TOGETHER_API_KEY")
    )
    .build()
)

result = pipeline.execute()
```

## Provider Examples

### Together.AI

```python
.with_llm(
    provider="openai",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    api_base="https://api.together.xyz/v1",
    api_key=os.getenv("TOGETHER_API_KEY")
)
```

### vLLM (Self-Hosted)

```python
.with_llm(
    provider="openai",
    model="meta-llama/Llama-2-7b-chat-hf",
    api_base="http://localhost:8000/v1",
    api_key="dummy"  # vLLM doesn't require auth
)
```

### Ollama (Local)

```python
.with_llm(
    provider="openai",
    model="llama2",
    api_base="http://localhost:11434/v1",
    api_key="ollama"  # Any non-empty string
)
```

### Custom Endpoint

```python
.with_llm(
    provider="openai",
    model="your-model-name",
    api_base="https://your-api.example.com/v1",
    api_key=os.getenv("YOUR_API_KEY")
)
```

## Using LLMSpec (Advanced)

For more control, use `LLMSpec`:

```python
from ondine.core.specifications import LLMSpec

custom_spec = LLMSpec(
    provider="openai",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    api_base="https://api.together.xyz/v1",
    api_key=os.getenv("TOGETHER_API_KEY"),
    temperature=0.7,
    max_tokens=1000,
    # Pricing (optional, for cost tracking)
    input_cost_per_million=0.20,
    output_cost_per_million=0.20
)

pipeline = (
    PipelineBuilder.create()
    ...
    .with_llm_spec(custom_spec)
    .build()
)
```

## Requirements

The custom API must be OpenAI-compatible, supporting:
- `/v1/chat/completions` endpoint
- Standard OpenAI request/response format
- Bearer token authentication

## Troubleshooting

### Connection Errors

Check that the API endpoint is accessible:
```bash
curl -X POST https://api.example.com/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"test"}]}'
```

### Authentication Errors

Verify your API key is correct and has proper permissions.

### Model Not Found

Ensure the model name matches what the API expects.

## Related

- [OpenAI](openai.md)
- [API Reference: LLMSpec](../../api/specifications.md)

