# Examples

Runnable scripts covering every major feature. Clone the repo, set your API keys, pick one.

```bash
git clone https://github.com/ptimizeroracle/ondine.git
cd ondine
pip install -e ".[all]"
export OPENAI_API_KEY="your-key"
```

## Getting Started

| # | Script | What it does |
|---|---|---|
| 01 | [`01_quickstart.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/01_quickstart.py) | PipelineBuilder basics, DataFrame input, cost estimation |
| 02 | [`02_simple_processor.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/02_simple_processor.py) | Minimal CSV processing with DatasetProcessor |
| 03 | [`03_structured_output.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/03_structured_output.py) | Pydantic models, auto-retry on validation errors |

## Cost & Performance

| # | Script | What it does |
|---|---|---|
| 04 | [`04_with_cost_control.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/04_with_cost_control.py) | Budget limits, cost estimation, rate limiting, checkpointing |
| 20 | [`20_prefix_caching.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/20_prefix_caching.py) | System prompt caching for 40-50% cost reduction |
| 21 | [`21_multi_row_batching.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/21_multi_row_batching.py) | 100 rows per API call, 100x fewer requests |

## Execution Modes

| # | Script | What it does |
|---|---|---|
| 07 | [`07_async_execution.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/07_async_execution.py) | Concurrent async processing with configurable parallelism |
| 08 | [`08_streaming_large_files.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/08_streaming_large_files.py) | Stream 100K+ rows with constant memory |

## Providers

| # | Script | What it does |
|---|---|---|
| 05 | [`05_groq_example.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/05_groq_example.py) | Groq (fast inference) |
| 10 | [`10_mlx_qwen3_local.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/10_mlx_qwen3_local.py) | Local MLX on Apple Silicon -- zero API cost |
| 13 | [`13_custom_client.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/13_custom_client.py) | Custom HTTP client for any API |
| 14 | [`14_provider_presets.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/14_provider_presets.py) | Provider preset configurations |
| 15 | [`15_custom_llm_provider.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/15_custom_llm_provider.py) | Build your own LLM provider class |
| 19 | [`19_azure_managed_identity_complete.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/19_azure_managed_identity_complete.py) | Azure AD auth without API keys |

## Observability

| # | Script | What it does |
|---|---|---|
| 15 | [`15_observability_logging.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/15_observability_logging.py) | Structured logging setup |
| 16 | [`16_observability_opentelemetry.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/16_observability_opentelemetry.py) | OpenTelemetry traces and spans |
| 17 | [`17_observability_langfuse.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/17_observability_langfuse.py) | Langfuse integration |
| 18 | [`18_observability_multi.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/18_observability_multi.py) | Multiple observability backends at once |

## Advanced

| # | Script | What it does |
|---|---|---|
| 06 | [`06_from_config_file.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/06_from_config_file.py) | YAML-driven pipeline configuration |
| 09 | [`09_system_prompts.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/09_system_prompts.py) | System prompts for prefix caching |
| 16 | [`16_custom_pipeline_stage.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/16_custom_pipeline_stage.py) | Write custom pipeline stages |
| 17 | [`17_plugin_architecture_demo.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/17_plugin_architecture_demo.py) | Plugin system for extensibility |
| | [`context_store_example.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/context_store_example.py) | Anti-hallucination with Context Store |
| | [`rag_knowledge_base_example.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/rag_knowledge_base_example.py) | RAG pipeline with KnowledgeStore |
| | [`multi_column_composition_example.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/multi_column_composition_example.py) | Multi-column output + pipeline composition |

## YAML Configs

These configs work with `06_from_config_file.py` or the CLI (`ondine process --config`):

| Config | Provider |
|---|---|
| [`config_example.yaml`](https://github.com/ptimizeroracle/ondine/blob/main/examples/config_example.yaml) | OpenAI |
| [`azure_managed_identity_config.yaml`](https://github.com/ptimizeroracle/ondine/blob/main/examples/azure_managed_identity_config.yaml) | Azure (managed identity) |
| [`azure_api_key_config.yaml`](https://github.com/ptimizeroracle/ondine/blob/main/examples/azure_api_key_config.yaml) | Azure (API key) |
| [`10_mlx_qwen3_local.yaml`](https://github.com/ptimizeroracle/ondine/blob/main/examples/10_mlx_qwen3_local.yaml) | Local MLX |
| [`10_ollama_local.yaml`](https://github.com/ptimizeroracle/ondine/blob/main/examples/10_ollama_local.yaml) | Ollama |
| [`11_together_ai.yaml`](https://github.com/ptimizeroracle/ondine/blob/main/examples/11_together_ai.yaml) | Together AI |
| [`12_vllm_custom.yaml`](https://github.com/ptimizeroracle/ondine/blob/main/examples/12_vllm_custom.yaml) | vLLM |
| [`composition_example.yaml`](https://github.com/ptimizeroracle/ondine/blob/main/examples/composition_example.yaml) | Pipeline composition |
