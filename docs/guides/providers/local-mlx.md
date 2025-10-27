# Local MLX Provider (Apple Silicon)

Run models locally on Apple Silicon (M1/M2/M3/M4) with MLX - 100% free, private, offline-capable.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB+ RAM recommended
- Python 3.10+

## Installation

```bash
pip install ondine[mlx]
```

## Basic Usage

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["summary"])
    .with_prompt("Summarize: {text}")
    .with_llm(
        provider="mlx",
        model="mlx-community/Qwen2.5-7B-Instruct-4bit",
        temperature=0.7,
        max_tokens=500
    )
    .build()
)

result = pipeline.execute()
print(f"Cost: ${result.costs.total_cost:.2f}")  # Always $0.00
```

## Available Models

Any MLX-compatible model from Hugging Face:

- `mlx-community/Qwen2.5-7B-Instruct-4bit` - Recommended, fast
- `mlx-community/Llama-3.2-3B-Instruct-4bit` - Lightweight
- `mlx-community/Mistral-7B-Instruct-v0.3-4bit` - Good quality

## Configuration

```python
.with_llm(
    provider="mlx",
    model="mlx-community/Qwen2.5-7B-Instruct-4bit",
    temperature=0.7,
    max_tokens=1000
)
```

## Performance Considerations

- **No concurrency benefit**: MLX runs on single GPU, use concurrency=1
- **First run slower**: Model downloads and caches on first use
- **Memory usage**: 4-bit models use ~4-8GB RAM

## Benefits

- **Zero cost**: No API fees
- **Privacy**: Data never leaves your machine
- **Offline**: Works without internet
- **No rate limits**: Process as much as you want

## Limitations

- Only works on Apple Silicon Macs
- Slower than cloud APIs (but free!)
- No concurrency benefit

## Related

- [Cost Control](../cost-control.md)
- [Custom Providers](custom.md)

