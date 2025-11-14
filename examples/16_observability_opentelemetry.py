"""
Example: Infrastructure observability with OpenTelemetry.

Demonstrates how to use OpenTelemetry for distributed tracing
and send traces to Jaeger, Datadog, or other OTEL-compatible backends.

Setup:
    1. Start Jaeger locally (for testing):
       docker run -d --name jaeger \
         -p 16686:16686 \
         -p 4318:4318 \
         jaegertracing/all-in-one:latest

    2. Install OpenTelemetry exporters:
       pip install opentelemetry-exporter-jaeger

    3. Run this example and view traces at http://localhost:16686
"""

import pandas as pd
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from ondine import PipelineBuilder

# Configure OpenTelemetry to export to Jaeger
resource = Resource(attributes={SERVICE_NAME: "ondine-example"})
tracer_provider = TracerProvider(resource=resource)

# Add Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))

# Set as global tracer provider
trace.set_tracer_provider(tracer_provider)

print("OpenTelemetry configured to export to Jaeger at localhost:6831")

# Sample data
data = pd.DataFrame(
    {
        "review": [
            "This product is amazing! Best purchase ever.",
            "Terrible quality, broke after one week.",
            "It's okay, nothing special but does the job.",
        ]
    }
)

# Build pipeline with OpenTelemetry observer
pipeline = (
    PipelineBuilder.create()
    .from_dataframe(data, input_columns=["review"], output_columns=["sentiment"])
    .with_prompt(
        """
        Classify the sentiment of this review as: positive, negative, or neutral.

        Review: {review}
        """
    )
    .with_llm(provider="openai", model="gpt-4o-mini", temperature=0.0)
    # Add OpenTelemetry observer
    .with_observer(
        "opentelemetry",
        config={
            "tracer_name": "ondine.example.sentiment",
            "include_prompts": False,  # Don't include prompts for PII safety
        },
    )
    .build()
)

# Execute
print("\nExecuting pipeline with OpenTelemetry tracing...")
result = pipeline.execute()

print("\nResults:")
print(result.data)

print(f"\nTotal cost: ${result.costs.total_cost:.4f}")
print("\nView traces at: http://localhost:16686")
print('Search for service "ondine-example" in Jaeger UI')

# Flush traces before exit
tracer_provider.force_flush()
