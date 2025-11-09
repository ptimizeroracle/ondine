"""
Observability Example - Distributed tracing with OpenTelemetry.

This example demonstrates how to enable distributed tracing for
production debugging and performance monitoring.

Requirements:
    pip install ondine[observability]
"""

import pandas as pd

from ondine import PipelineBuilder

# Example 1: Console Exporter (Development/Testing)
print("=" * 80)
print("Example 1: Console Exporter (for development)")
print("=" * 80)

from ondine.observability import disable_tracing, enable_tracing

# Enable tracing with console output
enable_tracing(exporter="console")

# Create sample data
data = pd.DataFrame(
    {
        "product": [
            "MacBook Pro 16-inch",
            "Dell XPS 15",
        ],
    }
)

# Build pipeline
pipeline = (
    PipelineBuilder.create()
    .from_dataframe(data, input_columns=["product"], output_columns=["summary"])
    .with_prompt("Summarize this product in 10 words: {product}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .build()
)

# Execute - traces will be printed to console
print("\nExecuting pipeline with console tracing...\n")
result = pipeline.execute()

print("\n✅ Pipeline complete!")
print(f"Processed: {result.metrics.processed_rows} rows")
print(f"Cost: ${result.costs.total_cost:.6f}")

# Cleanup
disable_tracing()

# Example 2: Jaeger Exporter (Production)
print("\n" + "=" * 80)
print("Example 2: Jaeger Exporter (for production monitoring)")
print("=" * 80)
print("""
To use Jaeger:
1. Start Jaeger locally:
   docker run -d --name jaeger \\
     -p 14268:14268 \\
     -p 16686:16686 \\
     jaegertracing/all-in-one:latest

2. Enable tracing with Jaeger endpoint:
   enable_tracing(
       exporter="jaeger",
       endpoint="http://localhost:14268/api/traces"
   )

3. Execute your pipeline

4. View traces at: http://localhost:16686

Example trace hierarchy:
```
pipeline.execute (root span)
├── stage.DataLoaderStage
├── stage.PromptFormatterStage
├── stage.LLMInvocationStage
│   └── llm.invoke (nested)
├── stage.ResponseParserStage
└── stage.ResultWriterStage
```

Each span includes:
- Duration (latency)
- Attributes (model, tokens, cost, rows processed)
- Error details (if any)
- Span context for distributed tracing
""")

# Example 3: PII Sanitization
print("\n" + "=" * 80)
print("Example 3: PII Sanitization (safe by default)")
print("=" * 80)


print("""
By default, prompts and responses are sanitized (PII-safe):

    # Default: prompts are hashed
    observer = TracingObserver(include_prompts=False)
    # Trace will show: "<sanitized-1234>" instead of actual prompt

To include prompts (opt-in for debugging):

    # Opt-in: include actual prompts (USE WITH CAUTION)
    observer = TracingObserver(include_prompts=True)
    # pipeline.add_observer(observer)  # TODO: Implement observer attachment

⚠️  Warning: Only use include_prompts=True in non-production environments
    or when you're certain prompts don't contain PII.
""")

# Example 4: Benefits of Tracing
print("\n" + "=" * 80)
print("Example 4: Why use tracing?")
print("=" * 80)
print("""
Production Debugging Benefits:

1. **Find bottlenecks**: Identify which stage is slow
   - DataLoader: 0.5s
   - LLMInvocation: 5.2s  ← Bottleneck!
   - Parser: 0.1s

2. **Debug failures**: See exact row and stage where failure occurred
   - Row 8,347 failed in LLMInvocationStage
   - Error: Rate limit exceeded
   - Context: Timestamp, session ID, retry attempt

3. **Cost tracking**: Per-stage cost breakdown
   - Stage 1: $0.05
   - Stage 2 (LLM): $1.25  ← Most expensive
   - Stage 3: $0.00

4. **Distributed systems**: Trace requests across services
   - Frontend → API → Ondine Pipeline → LLM → Database
   - Single trace ID connects all steps

5. **Performance trends**: Historical analysis
   - P95 latency: 12s (last week) → 8s (this week)
   - Cost per row: $0.001 (stable)
""")

print("\n" + "=" * 80)
print("✅ Observability example complete!")
print("=" * 80)
print("\nNext steps:")
print("1. Try with your own pipeline")
print("2. Start Jaeger and view traces")
print("3. Integrate with your monitoring stack")
