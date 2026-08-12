# Local Models (Ollama)

Run models locally with [Ollama](https://ollama.com) — 100% free, private, no
API key. Ollama exposes an OpenAI-compatible endpoint, so Ondine talks to it
through the `openai_compatible` provider pointed at `http://localhost:11434/v1`.

Good for: development and testing without spending, sensitive data that must not
leave the machine, and offline runs.

## Requirements

- [Ollama installed](https://ollama.com/download) and running
- A pulled model
- Python 3.10+

## Setup

```bash
# 1. Install Ollama (see https://ollama.com/download), then pull a model
ollama pull llama3.1:8b

# 2. Ollama serves an OpenAI-compatible API on http://localhost:11434/v1
#    (it starts automatically after install; `ollama serve` if not)
```

## Basic Usage

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["sentiment"])
    .with_prompt(
        "Classify the sentiment as positive, negative, or neutral: {text}"
    )
    .with_llm(
        provider="openai_compatible",
        model="llama3.1:8b",           # any model you've pulled
        base_url="http://localhost:11434/v1",
        # No api_key needed — it's local.
        temperature=0.0,
        max_tokens=10,
        # Local inference is free; pin costs to zero so reports read $0.
        input_cost_per_1k_tokens=0.0,
        output_cost_per_1k_tokens=0.0,
    )
    # Local models usually run best with modest concurrency.
    .with_concurrency(1)
    .build()
)

result = pipeline.execute()
print(result.to_pandas())
print(f"Cost: ${result.costs.total_cost}")   # $0.00 — it ran on your machine
```

## From a YAML config

The same job as a config file (see `examples/10_ollama_local.yaml` in the repo):

```yaml
llm:
  provider: openai_compatible
  provider_name: "Ollama-Local"
  model: llama3.1:8b
  base_url: http://localhost:11434/v1
  temperature: 0.0
  max_tokens: 10
  input_cost_per_1k_tokens: 0.0
  output_cost_per_1k_tokens: 0.0
```

```bash
ondine process --config examples/10_ollama_local.yaml
```

## Notes

- **Cost is $0.** Nothing is billed; the cost fields above just keep reports tidy.
- **Throughput is bound by your hardware,** not a rate limit. Start with
  `concurrency=1` and raise it only if your GPU has headroom.
- **Model names must match what you pulled** (`ollama list` shows them).
- The endpoint is OpenAI-compatible, so anything that works against a custom
  OpenAI-style endpoint works here — see [Custom / Any API](custom.md).

## Related

- [Custom / Any API](custom.md) — the general `openai_compatible` provider
- [Local Models (MLX)](local-mlx.md) — the Apple-Silicon-native alternative
- [Cost Estimation & Budgets](../cost-control.md)
