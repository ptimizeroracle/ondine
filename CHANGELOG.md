# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
