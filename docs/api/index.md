# API Reference

Complete API documentation for Ondine, auto-generated from source code docstrings.

## Module Organization

Ondine is organized into clear layers:

### High-Level APIs

User-facing interfaces for building pipelines:

- **[ondine.api.pipeline](api/ondine/api/pipeline.md)** - Main `Pipeline` class
- **[ondine.api.pipeline_builder](api/ondine/api/pipeline_builder.md)** - Fluent builder for pipelines
- **[ondine.api.quick](api/ondine/api/quick.md)** - `QuickPipeline` with smart defaults
- **[ondine.api.pipeline_composer](api/ondine/api/pipeline_composer.md)** - Multi-column composition
- **[ondine.api.dataset_processor](api/ondine/api/dataset_processor.md)** - Direct dataset processing

### Core Models and Specifications

Configuration models and data structures:

- **[ondine.core.specifications](api/ondine/core/specifications.md)** - `LLMSpec`, `DatasetSpec`, `PromptSpec`, `ProcessingSpec`
- **[ondine.core.models](api/ondine/core/models.md)** - `ExecutionResult`, `CostEstimate`, `ProcessingStats`
- **[ondine.core.error_handler](api/ondine/core/error_handler.md)** - Error handling utilities

### Pipeline Stages

Composable processing stages:

- **[ondine.stages.pipeline_stage](api/ondine/stages/pipeline_stage.md)** - Base `PipelineStage` class
- **[ondine.stages.data_loader_stage](api/ondine/stages/data_loader_stage.md)** - Load data from various sources
- **[ondine.stages.prompt_formatter_stage](api/ondine/stages/prompt_formatter_stage.md)** - Format prompts with variables
- **[ondine.stages.llm_invocation_stage](api/ondine/stages/llm_invocation_stage.md)** - Invoke LLM APIs
- **[ondine.stages.response_parser_stage](api/ondine/stages/response_parser_stage.md)** - Parse and validate responses
- **[ondine.stages.parser_factory](api/ondine/stages/parser_factory.md)** - Parser utilities (`JSONParser`, `PydanticParser`)
- **[ondine.stages.result_writer_stage](api/ondine/stages/result_writer_stage.md)** - Write results to storage
- **[ondine.stages.streaming_loader_stage](api/ondine/stages/streaming_loader_stage.md)** - Streaming data loading

### Orchestration

Execution strategies and pipeline execution:

- **[ondine.orchestration.pipeline_executor](api/ondine/orchestration/pipeline_executor.md)** - Main executor
- **[ondine.orchestration.execution_strategy](api/ondine/orchestration/execution_strategy.md)** - Execution strategy interface
- **[ondine.orchestration.sync_executor](api/ondine/orchestration/sync_executor.md)** - Synchronous execution
- **[ondine.orchestration.async_executor](api/ondine/orchestration/async_executor.md)** - Asynchronous execution
- **[ondine.orchestration.streaming_executor](api/ondine/orchestration/streaming_executor.md)** - Streaming execution
- **[ondine.orchestration.execution_context](api/ondine/orchestration/execution_context.md)** - Execution context and state
- **[ondine.orchestration.state_manager](api/ondine/orchestration/state_manager.md)** - State management
- **[ondine.orchestration.observers](api/ondine/orchestration/observers.md)** - Observer pattern for monitoring

### Adapters

External integrations and abstractions:

- **[ondine.adapters.llm_client](api/ondine/adapters/llm_client.md)** - LLM client abstraction
- **[ondine.adapters.data_io](api/ondine/adapters/data_io.md)** - Data I/O utilities
- **[ondine.adapters.checkpoint_storage](api/ondine/adapters/checkpoint_storage.md)** - Checkpoint persistence

### Utilities

Cross-cutting concerns:

- **[ondine.utils.cost_tracker](api/ondine/utils/cost_tracker.md)** - Cost tracking
- **[ondine.utils.cost_calculator](api/ondine/utils/cost_calculator.md)** - Cost calculation
- **[ondine.utils.budget_controller](api/ondine/utils/budget_controller.md)** - Budget enforcement
- **[ondine.utils.rate_limiter](api/ondine/utils/rate_limiter.md)** - Rate limiting
- **[ondine.utils.retry_handler](api/ondine/utils/retry_handler.md)** - Retry logic
- **[ondine.utils.input_preprocessing](api/ondine/utils/input_preprocessing.md)** - Text preprocessing
- **[ondine.utils.logging_utils](api/ondine/utils/logging_utils.md)** - Logging configuration
- **[ondine.utils.metrics_exporter](api/ondine/utils/metrics_exporter.md)** - Metrics export

### Configuration

Configuration loading and validation:

- **[ondine.config.config_loader](api/ondine/config/config_loader.md)** - YAML configuration loader

### CLI

Command-line interface:

- **[ondine.cli.main](api/ondine/cli/main.md)** - CLI commands

### Integrations

Workflow orchestration integrations:

- **[ondine.integrations.airflow](api/ondine/integrations/airflow.md)** - Apache Airflow operators
- **[ondine.integrations.prefect](api/ondine/integrations/prefect.md)** - Prefect tasks

## Quick Reference

### Most Common Classes

```python
from ondine import Pipeline, PipelineBuilder, QuickPipeline
from ondine.core.specifications import LLMSpec, DatasetSpec, ProcessingSpec
from ondine.core.models import ExecutionResult, CostEstimate
from ondine.stages.response_parser_stage import PydanticParser
from ondine.stages.parser_factory import JSONParser
```

### Building a Pipeline

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Process: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .build()
)
```

### Using Specifications

```python
from ondine.core.specifications import LLMSpec, PipelineSpecifications

llm_spec = LLMSpec(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.0
)

pipeline = (
    PipelineBuilder.create()
    .from_csv(...)
    .with_llm_spec(llm_spec)
    .build()
)
```

### Execution Result

```python
result = pipeline.execute()

# Access data
result.data  # pandas DataFrame

# Metrics
result.metrics.processed_rows
result.metrics.successful_rows
result.metrics.failed_rows
result.metrics.elapsed_time

# Costs
result.costs.total_cost
result.costs.input_tokens
result.costs.output_tokens
```

## Type Hints

Ondine uses comprehensive type hints throughout. Use a type checker like mypy for static type checking:

```bash
pip install mypy
mypy your_script.py
```

## IDE Support

All classes and functions include docstrings compatible with IDE autocomplete:

- VSCode: Pylance provides IntelliSense
- PyCharm: Built-in type hints and documentation
- Jupyter: `?` and `??` show documentation

Example:
```python
from ondine import PipelineBuilder

# Type Shift+Tab in Jupyter or Ctrl+Space in IDEs for autocomplete
PipelineBuilder.create().<tab>
```

## Examples

See the [examples directory](https://github.com/ptimizeroracle/ondine/tree/main/examples) for complete working examples.

## Navigation

Browse the full API documentation using the navigation sidebar, or jump directly to specific modules:

- [Pipeline Building](api/ondine/api/pipeline_builder.md)
- [Specifications](api/ondine/core/specifications.md)
- [Execution Strategies](api/ondine/orchestration/execution_strategy.md)
- [LLM Adapters](api/ondine/adapters/llm_client.md)
- [Response Parsing](api/ondine/stages/response_parser_stage.md)

