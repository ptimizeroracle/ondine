# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.1](https://github.com/ptimizeroracle/ondine/compare/v1.3.0...v1.3.1) (2025-11-16)


### Bug Fixes

* Make CLI version test dynamic and restore PyPI workflow ([#35](https://github.com/ptimizeroracle/ondine/issues/35)) ([e9cda06](https://github.com/ptimizeroracle/ondine/commit/e9cda064e6e5a9577ce76c36423e78f8149a5dc9))
* Make CLI version test dynamic instead of hardcoded ([#33](https://github.com/ptimizeroracle/ondine/issues/33)) ([31f780c](https://github.com/ptimizeroracle/ondine/commit/31f780c5a00a187586caf7ad1573c492def2cc5e))

## [1.3.0](https://github.com/ptimizeroracle/ondine/compare/v1.2.1...v1.3.0) (2025-11-16)


### Features

* Add automated versioning with Python Semantic Release ([#28](https://github.com/ptimizeroracle/ondine/issues/28)) ([3f2695d](https://github.com/ptimizeroracle/ondine/commit/3f2695dc8083bdf77c5ec50be7f3ef3e55ad1bb7))
* Add Multi-Row Batching for 100× Speedup ([#27](https://github.com/ptimizeroracle/ondine/issues/27)) ([4df836e](https://github.com/ptimizeroracle/ondine/commit/4df836e1a9a12f03cdc88224e45fc1b1951ac5bb))
* Add Prefix Caching Support for 40-50% Cost Reduction ([#25](https://github.com/ptimizeroracle/ondine/issues/25)) ([63b46b3](https://github.com/ptimizeroracle/ondine/commit/63b46b3d2defffb02af30da8ad2a78cdb3c43cfe))
* Switch to Release Please for automated versioning ([#30](https://github.com/ptimizeroracle/ondine/issues/30)) ([e5d2cdb](https://github.com/ptimizeroracle/ondine/commit/e5d2cdb7e91be941e4a3b2649e92a3acbafd88c3))


### Bug Fixes

* Update release workflow to use uv run semantic-release ([#29](https://github.com/ptimizeroracle/ondine/issues/29)) ([6dbfb16](https://github.com/ptimizeroracle/ondine/commit/6dbfb1617878db4a97c941bec3150723d2887743))


### Documentation

* Make prefix caching example generic instead of product-specific ([#26](https://github.com/ptimizeroracle/ondine/issues/26)) ([e00d510](https://github.com/ptimizeroracle/ondine/commit/e00d510d61d7ca4a15fdce0463561fc36a5756f1))
* remove outdated reference to non-existent DESIGN_IMPROVEMENT.md ([#23](https://github.com/ptimizeroracle/ondine/issues/23)) ([4bf72e1](https://github.com/ptimizeroracle/ondine/commit/4bf72e127c5ef36b338c980f2de5a13b0abd394e))

## [Unreleased]

### Added
- **Multi-Row Batching for 100× Speedup**
  - Process N rows in a single API call (up to 100× reduction in API calls)
  - `with_batch_size(N)` API for configuring batch size
  - `with_batch_strategy("json")` for batch formatting strategy
  - `BatchAggregatorStage` and `BatchDisaggregatorStage` for batch processing
  - Strategy Pattern for extensible batch formatting (JSON, CSV)
  - Automatic context window validation against model limits
  - Partial failure handling with row-by-row retry fallback
  - Model context limits registry (50+ models)
  - Flatten-then-concurrent pattern for true parallel batch processing

### Fixed
- **Concurrency Architecture**
  - Fixed sequential batch processing (batches were processed one-by-one)
  - Implemented flatten-then-concurrent pattern for parallel execution
  - All batches now process concurrently regardless of aggregation
  - Result: 50× speedup for large datasets

- **Prefix Caching with Batching**
  - Fixed system_message not being preserved in batch aggregation
  - BatchAggregatorStage now merges all custom fields from original metadata
  - Caching now works correctly with multi-row batching
  - Cache hits visible: "✅ Cache hit! 1152/8380 tokens cached (14%)"

- **Row Count Tracking**
  - Fixed off-by-one error in processed_rows count
  - Correct handling of aggregated vs non-aggregated batches
  - Progress tracking now shows accurate row counts

- **Rate Limiting**
  - Added burst_size parameter to RateLimiter to prevent rate limit errors
  - Set burst_size=min(20, concurrency) to limit initial burst
  - Prevents overwhelming provider burst limits

### Changed
- **Performance Optimizations**
  - Replaced df.iterrows() with df.itertuples() for 10× speedup in PromptFormatterStage
  - Added progress logging with hybrid strategy (10% OR 30s)
  - Added ETA and throughput metrics to progress messages
  - Suppress progress logs for fast operations (<5s)
  - Moved DEBUG content to actual DEBUG level (cleaner INFO logs)

- **API Improvements**
  - Renamed old `with_batch_size()` to `with_processing_batch_size()` for clarity
  - New `with_batch_size()` for multi-row batching (user-facing)
  - `with_processing_batch_size()` for internal batching (advanced users)

### Testing
- **New Tests**
  - 24 new unit tests for batch strategies and stages
  - Integration tests with real OpenAI API
  - Concurrent batch processing tests
  - All 435 tests passing with 60% coverage

### Documentation
- **New Guides**
  - `docs/guides/batch-processing.md` - Comprehensive multi-row batching guide
  - Updated `docs/guides/cost-control.md` with batching strategies
  - Updated `docs/getting-started/core-concepts.md` with batch stages
  - Updated `docs/architecture/technical-reference.md` with batch architecture
  - New example: `examples/21_multi_row_batching.py`

### Performance Impact
- **5.4M rows (4 stages):**
  - Without batching: 21.6M API calls, ~276 hours
  - With batch_size=100: 216K API calls, ~5.6 hours (50× faster!)
  - With caching + batching: ~$150 total cost (vs $800 without optimizations)

## [1.2.1] - 2025-11-12

### Added
- **Progress Tracking System**
  - `ProgressTracker`: Pluggable abstraction layer for progress tracking
  - `RichProgressTracker`: Rich terminal UI with real-time progress bars, cost tracking, and ETA
  - `LoggingProgressTracker`: Fallback for non-TTY environments
  - `NoopProgressTracker`: Disable progress tracking entirely
  - Auto-detection of terminal capabilities with graceful fallback
  - `.with_progress_mode()` API for configuring progress tracking ("auto", "rich", "logging", "none")
  - Per-row progress updates with cost and throughput metrics

- **Non-Retryable Error Classification**
  - `NonRetryableError`: Base class for fatal errors that should fail fast
  - `ModelNotFoundError`: Decommissioned or invalid models
  - `InvalidAPIKeyError`: Authentication failures
  - `ConfigurationError`: File not found, invalid config
  - `QuotaExceededError`: Credits exhausted (not rate limit)
  - `_classify_error()`: Leverages LlamaIndex native exceptions for error classification
  - Automatic cancellation of remaining futures when fatal error occurs
  - Prevents wasting time and money on retrying non-retryable errors

### Fixed
- **Cost Tracking**
  - Fixed double-counting of costs in `LLMInvocationStage` (removed batch-level `context.add_cost()`)
  - Costs now tracked accurately per-row only

- **Error Handling**
  - Fixed `AttributeError` when using SKIP policy: changed `self.llm_client.spec.model` to `self.llm_client.model`
  - Fatal errors now fail after 1 attempt instead of retrying indefinitely

### Changed
- **Git Ignore**
  - Added `titles_classified*.csv` and `*_stage*.csv` to `.gitignore` to prevent committing generated output files

### Testing
- **End-to-End Tests**
  - Added comprehensive E2E integration tests covering API contract and behavior
  - 25 new unit tests for non-retryable error classification
  - 403 total tests passing with 100% backward compatibility

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
