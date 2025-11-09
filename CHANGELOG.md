# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2025-11-09

### Added
- **Documentation Quality Tools**
  - `tools/check_docstring_coverage.py`: Scans and reports docstring coverage (93.62% achieved)
  - `tools/generate_docstring_report.py`: Analyzes docstring quality with scoring system
  - `tools/validate_docs_examples.py`: Validates code examples in documentation
- **CI/CD Enhancements**
  - `.github/workflows/docstring-quality.yml`: Automated docstring quality checks (80% threshold)
  - `.github/workflows/validate-docs.yml`: Documentation example validation
  - Integrated `pydocstyle` and `interrogate` tools
- **Comprehensive API Documentation**
  - Google-style docstrings with real-world examples for all core APIs
  - `PipelineBuilder`: Complete examples for all builder methods
  - `Pipeline`: Execution examples with error handling
  - `QuickPipeline`: Simple and advanced usage patterns
  - `DatasetSpec`, `LLMSpec`, `ProcessingSpec`: Detailed field descriptions
  - `ExecutionResult`, `CostEstimate`, `ProcessingStats`: Result inspection examples
  - `PipelineStage`: Template Method pattern explanation with custom stage example
- **Example Script**
  - `multi_stage_classification_groq.py`: 491-line multi-stage classification pipeline demonstrating scalability features

### Fixed
- **Critical API Bug Fixes**
  - Removed usage of non-existent `with_processing()` method in examples and documentation
  - Replaced with individual `.with_batch_size()` and `.with_concurrency()` calls
  - Fixed in: `examples/azure_managed_identity.py`, `examples/19_azure_managed_identity_complete.py`, `docs/guides/azure-managed-identity.md`
- **Result Access Corrections**
  - Updated from `result.rows_processed` to `result.metrics.total_rows`
  - Updated from `result.cost.total_cost` to `result.costs.total_cost`
- **Documentation Fixes**
  - Fixed logo paths in documentation (`../assets/images/` → `assets/images/`)
  - Corrected broken internal links

### Changed
- **Branding & Messaging**
  - Removed "Production-grade" marketing language throughout codebase
  - Replaced with more accurate, modest language ("SDK", "Fault Tolerant", etc.)
  - Toned down claims (removed "99.9% completion rate in production workloads")
  - Updated README, docs/index.md, ondine/__init__.py, ondine/cli/main.py

### Technical Details
- 24 files changed
- +1,849 lines of documentation and examples
- -75 lines of outdated/incorrect content
- 60% code coverage maintained
- 378 tests passing, 3 skipped
- Docstring coverage: 93.62% (threshold: 80%)

## [1.1.0] - 2025-10-29

### Added
- **Azure Managed Identity Authentication**
  - Native support for Azure Managed Identity (System-assigned and User-assigned)
  - Automatic token acquisition and refresh for Azure OpenAI
  - No API keys required when running on Azure infrastructure (VMs, App Service, Functions, AKS)
  - `AzureManagedIdentityClient` for seamless Azure integration
- **Examples**
  - `examples/azure_managed_identity.py`: Basic Azure Managed Identity usage
  - `examples/19_azure_managed_identity_complete.py`: Complete Azure integration example
- **Documentation**
  - `docs/guides/azure-managed-identity.md`: Comprehensive Azure Managed Identity guide
  - Setup instructions for Azure VMs, App Service, Functions, and AKS
  - Troubleshooting and best practices

### Changed
- Enhanced Azure OpenAI provider to support both API key and Managed Identity authentication
- Updated logo to transparent background version (1.9MB)
- Moved logo to `assets/images/` directory for better organization

### Technical Details
- All unit tests pass (378 passed, 3 skipped)
- Backward compatible with existing Azure OpenAI API key authentication
- Zero breaking changes

## [1.0.4] - 2025-10-28

### Added
- **Plugin-based observability system** leveraging LlamaIndex's built-in instrumentation
  - `with_observer()` method in PipelineBuilder for one-line observability configuration
  - Three official observers: OpenTelemetry, Langfuse, LoggingObserver
  - Observer registry with `@observer` decorator for plugin system
  - Automatic LLM call tracking (prompts, completions, tokens, costs, latency)
  - Multiple observers can run simultaneously
  - Fault-tolerant: observer failures never crash pipeline
- **PII sanitization** module with comprehensive regex patterns
  - Email, SSN, credit card, phone numbers, API keys, IP addresses
  - `sanitize_text()` and `sanitize_event()` functions
  - Custom patterns support
- **LlamaIndex handler integration**
  - Delegates to LlamaIndex's `set_global_handler()` for OpenTelemetry, Langfuse, Simple handlers
  - Zero manual instrumentation required
  - Production-ready, battle-tested observability
- **4 new example scripts**:
  - `examples/15_observability_logging.py` - Simple console logging
  - `examples/16_observability_opentelemetry.py` - OpenTelemetry + Jaeger
  - `examples/17_observability_langfuse.py` - Langfuse integration
  - `examples/18_observability_multi.py` - Multiple observers
- **Dependencies**: Added `opentelemetry-api`, `opentelemetry-sdk`, `langfuse` as required dependencies

### Changed
- **Simplified observers** by delegating to LlamaIndex (70% code reduction)
  - OpenTelemetryObserver: 200+ → 73 lines
  - LangfuseObserver: 240+ → 86 lines
  - LoggingObserver: 170+ → 69 lines
- **Removed manual event emission** from LLMInvocationStage (LlamaIndex auto-instruments)
- Updated documentation to emphasize LlamaIndex integration

### Technical Details
- Net code reduction: ~400 lines deleted
- All unit tests pass (366/366)
- Backward compatible with existing ExecutionObserver interface
- Observer failures isolated (try/except per observer)

## [1.0.0] - 2025-10-27

### Initial Release

**Ondine** - Production-grade SDK for batch processing tabular datasets with LLMs.

#### Core Features

- **Quick API**: 3-line hello world with smart defaults and auto-detection
- **Simple API**: Fluent builder pattern for full control
- **Reliability**: Automatic retries, checkpointing, error policies (99.9% completion rate)
- **Cost Control**: Pre-execution estimation, budget limits, real-time tracking
- **Production Ready**: Zero data loss on crashes, resume from checkpoint

#### LLM Provider Support

- OpenAI (GPT-4, GPT-3.5, etc.)
- Azure OpenAI
- Anthropic Claude
- Groq (fast inference)
- MLX (Apple Silicon local inference)
- Ollama (local models)
- Custom OpenAI-compatible APIs (Together.AI, vLLM, etc.)

#### Architecture

- **Plugin System**: `@provider` and `@stage` decorators for extensibility
- **Multi-Column Processing**: Generate multiple outputs with composition or JSON parsing
- **Observability**: OpenTelemetry integration with PII sanitization
- **Streaming**: Process large datasets without loading into memory
- **Async Execution**: Parallel processing with configurable concurrency

#### APIs

- `QuickPipeline.create()` - Simplified API with smart defaults
- `PipelineBuilder` - Full control with fluent builder pattern
- `PipelineComposer` - Multi-column composition from YAML
- CLI: `ondine process`, `ondine inspect`, `ondine validate`, `ondine estimate`

#### Quality

- 95%+ test coverage
- Type hints throughout
- Pre-commit hooks (ruff, bandit, detect-secrets)
- CI/CD with GitHub Actions
- Security scanning with TruffleHog

#### Documentation

- Comprehensive README with examples
- 18 example scripts covering all features
- Technical reference documentation
- Architecture Decision Records (ADRs)

#### Use Cases

- Data cleaning and standardization
- Content categorization and tagging
- Sentiment analysis at scale
- Entity extraction and enrichment
- Data quality assessment
- Batch translation
- Custom data transformations

---

## [Unreleased]

### Upcoming Features

- **RAG Integration**: Retrieval-Augmented Generation for context-aware processing
- **Enhanced Observability**: More metrics and tracing options
- **Additional Providers**: More LLM provider integrations

---

[1.0.0]: https://github.com/ptimizeroracle/Ondine/releases/tag/v1.0.0
