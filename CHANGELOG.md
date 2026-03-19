# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.1](https://github.com/ptimizeroracle/ondine/compare/v1.6.0...v1.6.1) (2026-03-19)


### Bug Fixes

* Add Python 3 ([38dab87](https://github.com/ptimizeroracle/ondine/commit/38dab87d0d3e53fca38bc7ea8b21566f5524c0ed)), closes [#93](https://github.com/ptimizeroracle/ondine/issues/93)
* fail fast when all Router deployments are in cooldown ([bc97232](https://github.com/ptimizeroracle/ondine/commit/bc97232ffa28061ef56df8888c1571c7f91a59ab))
* treat Router cooldown as retryable with consecutive-failure breaker ([bb184b8](https://github.com/ptimizeroracle/ondine/commit/bb184b87edc25a90701730c5b115369b92495052))

## [1.6.0](https://github.com/ptimizeroracle/ondine/compare/v1.5.3...v1.6.0) (2026-03-12)


### Features

* add content-hash routing strategy for cache-friendly load balancing ([f744caf](https://github.com/ptimizeroracle/ondine/commit/f744cafb805a1c99f8b71bcb122712a173400295))
* checkpoint resilience — gzip compression and optional cleanup ([e028882](https://github.com/ptimizeroracle/ondine/commit/e0288822e491e937a38e2acfff6bc327dff52062))
* framed ASCII pipeline report for logging progress mode ([ea4f093](https://github.com/ptimizeroracle/ondine/commit/ea4f093e66ee493ab3bfdc87a8768fddf57861c4))
* router-aware ASCII scoreboard for logging progress mode ([1b10ba7](https://github.com/ptimizeroracle/ondine/commit/1b10ba7b87b707a024499963e234bbaa8398e311))
* router-aware ASCII scoreboard for logging progress mode ([e169ad3](https://github.com/ptimizeroracle/ondine/commit/e169ad3e46687bc2e311eb1ff579e40b4a946a96))


### Bug Fixes

* accurate pipeline report labels and deduplicated logs ([b3e117a](https://github.com/ptimizeroracle/ondine/commit/b3e117a1673dda23635dad6c533f0c179e10ed75))
* add missing ensure_deployment_task to NoOpProgressTracker ([124b0b8](https://github.com/ptimizeroracle/ondine/commit/124b0b800408cc794daadbcc0308b6f94ac7247e))
* add missing ensure_deployment_task to NoOpProgressTracker ([ab804e3](https://github.com/ptimizeroracle/ondine/commit/ab804e3066b19d32747f92c1a4805cc50f113061))
* emit each router endpoint as separate log line ([1f52c1e](https://github.com/ptimizeroracle/ondine/commit/1f52c1e764c64a3beb5aeedc50a2df4f378b34d9))
* emit pipeline report lines individually to fix alignment ([fee9d6c](https://github.com/ptimizeroracle/ondine/commit/fee9d6c9ff7e52e4e17936f0bf004cb657ceeba0))
* improve LoggingProgressTracker with percentage, ETA, throughput, and per-deployment stats ([0ee9d18](https://github.com/ptimizeroracle/ondine/commit/0ee9d18ca0cb45da982396ad15c0c9a7cfbc5bbd))
* update checkpoint tests for gzip format and seed heartbeat timer ([0dbe438](https://github.com/ptimizeroracle/ondine/commit/0dbe438f1f28fc291048c6657fb58962e163d5cd))

## [1.5.3](https://github.com/ptimizeroracle/ondine/compare/v1.5.2...v1.5.3) (2026-03-11)


### Bug Fixes

* correct pandas 1.5 CI job — install into venv and pin compatible numpy ([37072b0](https://github.com/ptimizeroracle/ondine/commit/37072b0e1a2d3bf07c7e12dd6319df8c4019c05a))
* relax pandas constraint from &gt;=2.0.0 to &gt;=1.5.0 ([4187c52](https://github.com/ptimizeroracle/ondine/commit/4187c52dcdf54a33b624bc5efffdc8cfcf8cdb22))
* relax pandas constraint from &gt;=2.0.0 to &gt;=1.5.0 ([aeab1e4](https://github.com/ptimizeroracle/ondine/commit/aeab1e45e779fd06524ed5dccc272d06270451a2))
* skip observability tests in pandas 1.5 compat job ([9aff3a8](https://github.com/ptimizeroracle/ondine/commit/9aff3a8514708bce0c4becd7e7247ad065a241ae))
* use --no-sync in pandas compat CI to prevent uv from restoring lockfile ([58e6690](https://github.com/ptimizeroracle/ondine/commit/58e6690f554f940a394830bbc882d9c23b9a22aa))


### Documentation

* rewrite changelog for v1.5.0, v1.5.1, v1.5.2 with detailed release notes ([729d742](https://github.com/ptimizeroracle/ondine/commit/729d742eaaf083befb4c0e56bd4243695096f7d0))
* rewrite changelog for v1.5.0, v1.5.1, v1.5.2 with detailed release notes ([36068b9](https://github.com/ptimizeroracle/ondine/commit/36068b93140c872870d0801cdba85fa9bc4750d8))

## [1.5.2](https://github.com/ptimizeroracle/ondine/compare/v1.5.1...v1.5.2) (2026-03-11)

Scalene/cProfile profiling revealed per-API-call overhead in the structured output path. This release caches Pydantic schema generation and Instructor model preparation, cutting CPU waste and speeding up pipeline execution for large datasets.

### Performance Improvements

* **Pydantic JSON Schema Caching**: `BaseModel.model_json_schema()` is now cached by class identity. Previously called twice per API call by Instructor, this alone saved ~1.2ms × 2 × N calls of pure CPU time ([52cdb2f](https://github.com/ptimizeroracle/ondine/commit/52cdb2f151faef8b38d9f7c75ccf6d7af74d7143))
* **Instructor `prepare_response_model` Caching**: The wrapper class created by `openai_schema()` on every call is now cached, avoiding repeated class construction. Benchmark (mock LLM, 5000 rows): `handle_response_model` dropped from 1.65s to 0.21s (**-87%**), pipeline p50 from 3.3s to 2.5s (**-25%**) ([52cdb2f](https://github.com/ptimizeroracle/ondine/commit/52cdb2f151faef8b38d9f7c75ccf6d7af74d7143))
* **Lazy-Load `UnifiedLiteLLMClient`**: The heavy `litellm`/`instructor` import tree is now deferred via `__getattr__` in `ondine.adapters`, making `import ondine` **87% faster** for scripts that don't immediately invoke LLM calls ([52cdb2f](https://github.com/ptimizeroracle/ondine/commit/52cdb2f151faef8b38d9f7c75ccf6d7af74d7143))
* **Optimized `_calc_cost_from_response`**: Replaced `hasattr()` chains with `getattr()` defaults for cleaner and faster attribute access on LLM response objects ([52cdb2f](https://github.com/ptimizeroracle/ondine/commit/52cdb2f151faef8b38d9f7c75ccf6d7af74d7143))

### Bug Fixes

* **Thread-Safe Schema Patching** (CodeRabbit): The `_install_json_schema_cache()` monkey-patch was not atomic — under high concurrency, multiple coroutines could race to patch `BaseModel.model_json_schema`, potentially wrapping an already-patched function recursively. Fixed with a `threading.Lock` and double-checked locking pattern ([953ef4f](https://github.com/ptimizeroracle/ondine/commit/953ef4f2e570271fef7c84bfa321b420da8a2602))
* **Defensive Schema Copy** (CodeRabbit): The cached schema dict was returned by reference, allowing downstream consumers (e.g. Instructor) to mutate it in-place and corrupt the cache for subsequent calls. Now returns `cached.copy()` ([953ef4f](https://github.com/ptimizeroracle/ondine/commit/953ef4f2e570271fef7c84bfa321b420da8a2602))
* **CI Fix — Cache Hit Detection**: Removed a `logger.isEnabledFor(logging.DEBUG)` early-exit guard in `_check_cache_hit` that prevented cache-hit detection from running when the test mocked `logger.debug` without setting the log level to DEBUG, causing `test_prefix_caching_e2e` to fail on Python 3.11 ([b4d5076](https://github.com/ptimizeroracle/ondine/commit/b4d5076ed40bf4572afb347af7e5c3e5041835b8))

### Added

* **`perf` dependency group**: New optional dependency group for profiling tools — `memray`, `py-spy`, `pytest-benchmark`, `scalene`. Install with `uv sync --group perf`

## [1.5.1](https://github.com/ptimizeroracle/ondine/compare/v1.5.0...v1.5.1) (2026-03-10)

Stabilizes the Textual TUI introduced in v1.5.0, unifies progress state across trackers, and removes a confusing public API method.

### Bug Fixes

* **TUI Layout Overlap**: Wrapped progress widgets in a `Vertical` container to prevent the progress bar panel from overlapping the log panel. Simplified CSS by removing unused selectors and tightening margins ([cc92f9d](https://github.com/ptimizeroracle/ondine/commit/cc92f9d9c22f5dc71b45476e3aa206dd057ac872))
* **Dropped Row Visibility**: The pipeline summary report now shows a "Dropped rows" line when `processed + failed + skipped < total`, making silent row loss always visible ([cc92f9d](https://github.com/ptimizeroracle/ondine/commit/cc92f9d9c22f5dc71b45476e3aa206dd057ac872))
* **Unified Progress State**: Cost and progress totals are now routed through shared `ExecutionContext` state so all tracker implementations (Rich, Textual, Logging) stay consistent. Previously, each tracker maintained independent counters that could drift ([38369b9](https://github.com/ptimizeroracle/ondine/commit/38369b924d056f44305fdf60736148dbb1c7c8b3))
* **Simplified Deployment Bars**: Replaced deployment sub-bars with lightweight text labels so the TUI stays readable and responsive under high-concurrency runs with many deployments ([38369b9](https://github.com/ptimizeroracle/ondine/commit/38369b924d056f44305fdf60736148dbb1c7c8b3))

### Removed

* **`with_processing_batch_size()` from public API**: This builder method only controlled internal prompt grouping in `PromptFormatterStage` with no effect on API calls, cost, or throughput. It confused users by having a nearly identical name to `with_batch_size()` (which controls actual multi-row batching). The internal `ProcessingSpec.batch_size` default (100) remains for internal use ([90f8c81](https://github.com/ptimizeroracle/ondine/commit/90f8c81))

## [1.5.0](https://github.com/ptimizeroracle/ondine/compare/v1.4.3...v1.5.0) (2026-03-09)

Adds a Textual-based TUI for interactive progress monitoring, a generic pipeline summary report, and fixes a critical bug where Pydantic structured results were lost during checkpoint restore and batch retry.

### Features

* **Textual TUI Progress Tracker**: New pluggable progress mode (`"textual"`) providing a full terminal UI with fixed progress bars at top and a scrollable `RichLog` widget at bottom for real-time log inspection. Runs Textual in a daemon thread with `call_from_thread()` communication. Supports headless mode for CI. Install with `pip install ondine[tui]` ([5d7017d](https://github.com/ptimizeroracle/ondine/commit/5d7017d3ed5df8ee70a7cf54cf853d6262e210af))
* **Split-Panel Rich Progress**: The existing Rich progress tracker now uses a `Layout`-based split panel — progress bars pinned at top, scrolling log messages at bottom. Deployment sub-bars show actual contribution percentage instead of always displaying 100% ([22075b9](https://github.com/ptimizeroracle/ondine/commit/22075b9652e1a096bf30bbec7a92c96148e40dd2))
* **Generic Pipeline Summary Report**: `ProgressTracker` ABC now has a `show_summary()` method so every progress mode (Rich, Textual, Logging) prints a final report with rows processed, duration, cost, and token usage at pipeline completion ([f91ad34](https://github.com/ptimizeroracle/ondine/commit/f91ad3404ea71349a1f91af6496f27bc3069591a))
* **Enhanced Textual TUI**: Header/footer, bordered panels, keybindings (Q to quit, S to toggle auto-scroll), and thread-safe signal patching for daemon-thread execution ([f91ad34](https://github.com/ptimizeroracle/ondine/commit/f91ad3404ea71349a1f91af6496f27bc3069591a))

### Bug Fixes

* **Critical — Structured Result Lost on Restore**: `_structured_result` Pydantic objects were not carried through `completed_responses` checkpoint records, causing `_restore_completed_response_batches` to lose them. The disaggregator then fell back to fragile JSON text parsing instead of using the fast path. Now stores the `_structured_result` reference in checkpoint data ([5e0e087](https://github.com/ptimizeroracle/ondine/commit/5e0e087d56d8adad2a957a4eca6a7a78b29e5ae7))
* **Batch Retry Amplification**: When a batch had fewer response items than input rows, the disaggregator silently dropped positions and marked everything as failed, triggering unnecessary retries. Now maps items by ID and pads missing positions with `"null"` instead of dropping them ([5e0e087](https://github.com/ptimizeroracle/ondine/commit/5e0e087d56d8adad2a957a4eca6a7a78b29e5ae7))
* **Textual Test Compatibility**: Tests requiring the `textual` package now use `pytest.mark.skipif` to gracefully skip in CI environments that don't install the `[tui]` extra ([07c53eb](https://github.com/ptimizeroracle/ondine/commit/07c53eb97131830391820b9c9d7a8c7913f3f806))

### Testing

* Tightened unit tests per TDD Enterprise skill audit: removed tautological roundtrip test, strengthened fast-path assertions, added test exercising `Pipeline._restore_completed_response_batches`, consolidated test setup into shared `_make_batch` builder
* 17 new unit tests for Textual TUI covering factory, lifecycle, context manager, and app interface

### Added

* `textual>=1.0.0` as optional `[tui]` dependency

## [1.4.3](https://github.com/ptimizeroracle/ondine/compare/v1.4.2...v1.4.3) (2026-03-09)


### Bug Fixes

* prefer JSON_SCHEMA mode for structured output and disable parallel tool calls ([#66](https://github.com/ptimizeroracle/ondine/issues/66)) ([faa2b2f](https://github.com/ptimizeroracle/ondine/commit/faa2b2f77a96e2f91f74c39d35b21365e97f7cf5))

## [1.4.2](https://github.com/ptimizeroracle/ondine/compare/v1.4.1...v1.4.2) (2026-03-08)


### Bug Fixes

* restore structured output compatibility for Anthropic ([#64](https://github.com/ptimizeroracle/ondine/issues/64)) ([f09d8b2](https://github.com/ptimizeroracle/ondine/commit/f09d8b21755c2ae76526968c8204d3845e6a43d1))

## [1.4.1](https://github.com/ptimizeroracle/ondine/compare/v1.4.0...v1.4.1) (2026-03-08)


### Bug Fixes

* harden streaming, checkpoint, and router regressions ([#61](https://github.com/ptimizeroracle/ondine/issues/61)) ([dee5f3c](https://github.com/ptimizeroracle/ondine/commit/dee5f3cb1f00c369b5a36833dfeb51a74e52b503))

## [1.4.0](https://github.com/ptimizeroracle/ondine/compare/v1.3.4...v1.4.0) (2025-12-02)


### Features

* LiteLLM Integration - 348 Lines Removed  ([#46](https://github.com/ptimizeroracle/ondine/issues/46)) ([b9bd5ea](https://github.com/ptimizeroracle/ondine/commit/b9bd5ea2bbdca4903ecfd8de74a95dedc28bdee4))
* LiteLLM Native Integration - Aggressive Refactor ([#48](https://github.com/ptimizeroracle/ondine/issues/48)) ([a2afc1d](https://github.com/ptimizeroracle/ondine/commit/a2afc1d197a618576651fd277d3ce31116636690))
* Native LlamaIndex Structured Output with Multi-Provider E2E Tests ([#44](https://github.com/ptimizeroracle/ondine/issues/44)) ([f6969d7](https://github.com/ptimizeroracle/ondine/commit/f6969d7750b6e988d1dae587e93eb6fe6ddfbcd9))

## [1.4.2] - 2025-12-01

### Features
* **Enhanced Resilience & Observability**:
  * Added `ProviderCooldownEvent` and `ProviderRecoveredEvent` for tracking circuit breaker state
  * Improved health checks: now treats `ContentPolicyViolationError` and `BadRequestError` as healthy connectivity (avoids false negatives)
  * Deployment progress bars now show accurate per-deployment totals and costs

* **Robust Structured Output**:
  * **Layered Instructor Mode Detection**: Intelligent detection of best mode (User Override > Special Case > LiteLLM Info > Registry > JSON Default)
  * Added `instructor_mode` parameter to `with_structured_output()` for manual control
  * Automatically uses `JSON` mode for Azure reasoning models (fixing compatibility issues)
  * Added **position-based fallback** for batch processing (no `id` field required in user schemas!)

### Bug Fixes
* **Critical**: Fixed bug where `structured_result` Pydantic objects were lost during batch processing
* **Critical**: Fixed indentation bug in `BatchDisaggregatorStage` that overwrote valid results
* **Clean Logs**: Removed all emojis from logs for cleaner production output
* **Testing**: Fixed mock iteration issues in unit tests

## [1.4.1](https://github.com/ptimizeroracle/ondine/compare/v1.4.0...v1.4.1) (2025-11-27)

### Features
* **Native LiteLLM Caching**:
  * Replaced custom `RedisResponseCache` with LiteLLM's native `litellm.cache` integration
  * Added support for **Disk Caching** (`diskcache`) for local persistence without Redis
  * Simplified `PipelineBuilder` API: `with_disk_cache()` and updated `with_redis_cache()`
  * Removed 100+ lines of redundant caching code while gaining support for S3, Momento, and more via LiteLLM

### Refactoring
* **Code Cleanup**: Deleted `ondine/adapters/cache_client.py` in favor of `litellm.caching`
* **Standardization**: Standardized `LLMSpec.cache_config` structure to match LiteLLM's native format

## [1.4.0](https://github.com/ptimizeroracle/ondine/compare/v1.3.4...v1.4.0) (2025-11-27)

### Features
* **LiteLLM Router Optimizations**
  * Support for `latency-based-routing` strategy
  * Router-aware error handling: treat `ModelNotFoundError` as transient node failure to trigger fallback
  * Enhanced multi-provider support (Groq, OpenAI, Moonshot, Cerebras)
  * Robust handling of `InstructorRetryException` in Router context

* **Batch Processing Enhancements**
  * Minified JSON prompt payload (indent=None) to save tokens and context
  * Graceful handling of empty/failed batches in `BatchDisaggregatorStage`
  * Relaxed Pydantic validation for failed batches to prevent pipeline crashes
  * Smarter `auto_retry_failed` logic: only retries rows where *all* output columns are null

### Bug Fixes
* **Progress Tracking**: Fixed freezing animations and "stuck at 1%" visual glitches in `RichProgressTracker`
* **Cost Tracking**: Fixed duplicate cost counting and improved per-provider cost attribution in Router mode
* **Error Handling**: Fixed false positive "Model Not Found" on network timeouts
* **Response Parsing**: Fixed bug where single-column output parsers grabbed `id` instead of data
* **Testing**: Fixed regression in `test_non_retryable_errors` by explicitly mocking Router behavior

## [1.3.4](https://github.com/ptimizeroracle/ondine/compare/v1.3.3...v1.3.4) (2025-11-20)


### Bug Fixes

* Critical bugs in auto_retry and quality metrics (v1.3.4) ([#42](https://github.com/ptimizeroracle/ondine/issues/42)) ([80e9319](https://github.com/ptimizeroracle/ondine/commit/80e93195a0358c46bdc3ed9b648137bf6b270ed5))

## [1.3.3](https://github.com/ptimizeroracle/ondine/compare/v1.3.2...v1.3.3) (2025-11-16)


### Bug Fixes

* Prevent division by zero in batch aggregator (Windows CI) ([#38](https://github.com/ptimizeroracle/ondine/issues/38)) ([89c7480](https://github.com/ptimizeroracle/ondine/commit/89c7480396207398ca0936c1bbac1f671cb6611c))

## [1.3.2](https://github.com/ptimizeroracle/ondine/compare/v1.3.1...v1.3.2) (2025-11-16)


### Bug Fixes

* Resolve integration test regressions from multi-row batching ([#36](https://github.com/ptimizeroracle/ondine/issues/36)) ([37931bf](https://github.com/ptimizeroracle/ondine/commit/37931bf0182d2ab5463a3a317c922baff7e3864c))

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
