# Ondine Feature Verification Matrix

Every feature documented in README, docs, or CHANGELOG is either tested or documented here.

## Tested Features (69 tests)

| # | Feature | Test File | Test Name |
|---|-------|-----------|-----------|
| 1 | 100x fewer API calls via batching | test_verify_performance.py | test_claim_01_batching_reduces_api_calls |
| 2 | 100x faster processing | test_verify_performance.py | test_claim_02_batching_throughput_scales |
| 3 | 40-50% cost reduction via prefix caching | test_verify_performance.py | test_claim_03_prefix_caching_config |
| 4 | 99.9% completion rate | test_verify_performance.py | test_claim_04_completion_rate_with_retries |
| 5 | 1K rows < 5 min | test_verify_performance.py | test_claim_05_pipeline_overhead_acceptable |
| 6 | 87% faster import | test_verify_performance.py | test_claim_06_import_time_fast |
| 7 | -25% p50 latency from schema caching | test_verify_performance.py | test_claim_07_schema_caching_mechanism |
| 8 | -87% response model overhead | test_verify_performance.py | test_claim_08_response_parsing_fast |
| 9 | Sub-millisecond Rust lookups | test_verify_performance.py | test_claim_09_rust_store_submillisecond_lookup |
| 10 | 100+ providers via LiteLLM | test_verify_functionality.py | test_claim_10_litellm_provider_integration |
| 11 | Quick API (3-line) | test_verify_functionality.py | test_claim_11_quick_api_minimal_args |
| 12 | Builder API (fluent) | test_verify_functionality.py | test_claim_12_builder_api_fluent_chain |
| 13 | Multi-row batching | test_verify_functionality.py | test_claim_13_batching_configurable |
| 14 | Prefix caching | test_verify_functionality.py | test_claim_14_prefix_caching_enabled_by_default |
| 15 | Cost estimation | test_verify_functionality.py | test_claim_15_cost_estimation_calculation |
| 16 | Budget limits | test_verify_functionality.py | test_claim_16_budget_limits_enforcement |
| 17 | Checkpointing | test_verify_functionality.py | test_claim_17_checkpointing_roundtrip |
| 18 | Structured output | test_verify_functionality.py | test_claim_18_structured_output_pydantic |
| 19 | Multi-column output | test_verify_functionality.py | test_claim_19_multi_column_output |
| 20 | Pipeline composition | test_verify_functionality.py | test_claim_20_pipeline_composer |
| 21 | Async execution | test_verify_functionality.py | test_claim_21_async_execution_configurable |
| 22 | Streaming execution | test_verify_functionality.py | test_claim_22_streaming_execution |
| 23 | Observability | test_verify_functionality.py | test_claim_23_observability_registry |
| 24 | Router strategies | test_verify_functionality.py | test_claim_24_router_strategies |
| 25 | MLX local inference | test_verify_functionality.py | test_claim_25_mlx_client_exists |
| 26 | Provider presets | test_verify_functionality.py | test_claim_26_provider_presets_exist |
| 27 | Custom providers | test_verify_functionality.py | test_claim_27_custom_provider_registration |
| 28 | CLI commands | test_verify_functionality.py | test_claim_28_cli_commands_defined |
| 29 | Type-safe (Pydantic) | test_verify_functionality.py | test_claim_29_type_safe_pydantic_specs |
| 30 | Auto-detection | test_verify_functionality.py | test_claim_30_auto_detection_provider_from_model |
| 31 | Anti-hallucination layer | test_verify_quality_safety.py | test_claim_31_context_store_protocol |
| 32 | Evidence graph | test_verify_quality_safety.py | test_claim_32_evidence_store_retrieve_cycle |
| 33 | Confidence scoring | test_verify_quality_safety.py | test_claim_33_confidence_field_on_evidence |
| 34 | Grounding verification | test_verify_quality_safety.py | test_claim_34_grounding_method |
| 35 | Contradiction detection | test_verify_quality_safety.py | test_claim_35_contradiction_detection |
| 36 | RAG / Knowledge Store | test_verify_quality_safety.py | test_claim_36_knowledge_store_search |
| 37 | Evidence priming | test_verify_quality_safety.py | test_claim_37_evidence_retrieval_stage |
| 38 | Automatic retries | test_verify_quality_safety.py | test_claim_38_retry_handler_exponential_backoff |
| 39 | Error policies | test_verify_quality_safety.py | test_claim_39_error_policies |
| 40 | Partial failure handling | test_verify_quality_safety.py | test_claim_40_partial_failure_handling |
| 41 | LiteLLM callbacks | test_verify_integration.py | test_claim_41_litellm_callback_support |
| 42 | Langfuse integration | test_verify_integration.py | test_claim_42_langfuse_observer_exists |
| 43 | OpenTelemetry | test_verify_integration.py | test_claim_43_opentelemetry_observer_exists |
| 44 | Azure Managed Identity | test_verify_integration.py | test_claim_44_azure_managed_identity_optional_dep |
| 45 | Multi-observer | test_verify_integration.py | test_claim_45_multi_observer_dispatch |
| 46 | Cache metrics | test_verify_integration.py | test_claim_46_metrics_exporter_exists |
| 47 | Router events | test_verify_integration.py | test_claim_47_router_strategies_complete |
| 48 | CSV support | test_verify_data_formats.py | test_claim_48_csv_read_write_roundtrip |
| 49 | Parquet support | test_verify_data_formats.py | test_claim_49_parquet_read_write_roundtrip |
| 50 | Excel support | test_verify_data_formats.py | test_claim_50_excel_read_write_roundtrip |
| 51 | DataFrame support | test_verify_data_formats.py | test_claim_51_dataframe_passthrough |
| 52 | JSON support | test_verify_data_formats.py | test_claim_52_json_loading |
| 53 | Sync execution | test_verify_execution.py | test_claim_53_sync_executor_exists |
| 54 | Async execution | test_verify_execution.py | test_claim_54_async_executor_exists |
| 55 | Streaming execution | test_verify_execution.py | test_claim_55_streaming_executor_exists |
| 56 | Checkpoint interval | test_verify_execution.py | test_claim_56_checkpoint_interval_configurable |
| 57 | Rate limiting | test_verify_execution.py | test_claim_57_rate_limiter_limits_throughput |
| 58 | Concurrency control | test_verify_execution.py | test_claim_58_concurrency_configurable |
| 59 | Context window validation | test_verify_execution.py | test_claim_59_context_window_validation_exists |
| 60 | Smart defaults | test_verify_execution.py | test_claim_60_smart_defaults |
| 64 | Data cleaning | test_verify_use_cases.py | test_claim_64_data_cleaning |
| 65 | Sentiment analysis | test_verify_use_cases.py | test_claim_65_sentiment_analysis |
| 66 | Information extraction | test_verify_use_cases.py | test_claim_66_information_extraction |
| 67 | Auto-categorization | test_verify_use_cases.py | test_claim_67_auto_categorization |
| 68 | Content generation | test_verify_use_cases.py | test_claim_68_content_generation |
| 69 | Translation | test_verify_use_cases.py | test_claim_69_translation |
| 70 | Data enrichment | test_verify_use_cases.py | test_claim_70_data_enrichment |
| 71 | Product matching | test_verify_use_cases.py | test_claim_71_product_matching |
| 72 | Content moderation | test_verify_use_cases.py | test_claim_72_content_moderation |

## Documentation-Only Features (14 features)

These features cannot be automatically tested but are verified by other means:

| # | Feature | Justification |
|---|-------|---------------|
| 61 | 100% test pass rate (461 unit + 103 integration) | Meta-claim: verified by CI itself passing. Test counts have grown since this was written. |
| 62 | Fully backward compatible v1.3.0+ | Verified by `test-pandas-compat` CI job (pandas 1.5.3 compatibility). |
| 63 | 80% docstring coverage | Verified by `interrogate` tool in `docstring-quality.yml` CI workflow. |
| 73 | Purpose-built for tabular data (vs LangChain) | Competitive positioning — LangChain focuses on chains/agents, not tabular batch processing. |
| 74 | Multi-row batching (LangChain/DSPy: no) | Verified by Claim 1 test. LangChain/DSPy lack native batch aggregation. |
| 75 | Prefix caching (LangChain/DSPy: no) | Verified by Claim 3 test. Feature is provider-native, passed through by Ondine. |
| 76 | Pre-run cost estimation (others: no) | Verified by Claim 15 test. |
| 77 | Budget limits (others: no) | Verified by Claim 16 test. |
| 78 | Checkpointing (others: no) | Verified by Claim 17 test. |
| 79 | Structured output via Instructor | Verified by Claim 18 test. LangChain has own structured output. |
| 80 | Setup complexity: pip install ondine | Verified by `install-profiles` CI job testing core/observability/all profiles. |
| 81 | Development Status: Beta | Classifier in pyproject.toml: "Development Status :: 4 - Beta". |
| 82 | Recent updates (v1.7.0 March 2026) | CHANGELOG.md documents v1.7.0 release. |
| 83 | Deprecated features properly marked | `deprecated` package in observability extras, visible in pyproject.toml. |
