"""
Example: Multiple observers simultaneously.

Demonstrates how to use multiple observers at once for comprehensive
observability across different dimensions.
"""

import pandas as pd

from ondine import PipelineBuilder

# Sample data
data = pd.DataFrame(
    {
        "email": [
            "Just received the product, looks great!",
            "This is spam, click here for free money!!!",
            "Meeting tomorrow at 3pm, please confirm.",
        ]
    }
)

# Build pipeline with multiple observers
pipeline = (
    PipelineBuilder.create()
    .from_dataframe(data, input_columns=["email"], output_columns=["category", "spam"])
    .with_prompt(
        """
        Classify this email and detect if it's spam.
        Return JSON: {"category": "personal|work|spam", "spam": true|false}

        Email: {email}
        """
    )
    .with_llm(provider="openai", model="gpt-4o-mini", temperature=0.0)
    # Add multiple observers
    .with_observer(
        "logging",
        config={
            "log_level": "INFO",
            "include_prompts": True,
        },
    )
    .with_observer(
        "opentelemetry",
        config={
            "tracer_name": "ondine.multi_observer_example",
            "include_prompts": False,  # Don't include prompts in traces
        },
    )
    # Uncomment if you have Langfuse credentials
    # .with_observer(
    #     "langfuse",
    #     config={
    #         "public_key": "pk-lf-...",
    #         "secret_key": "sk-lf-...",
    #     }
    # )
    .build()
)

# Execute
print("Executing pipeline with multiple observers...")
print("- LoggingObserver: Console output")
print("- OpenTelemetryObserver: Distributed tracing")
print("- (Optionally) LangfuseObserver: LLM metrics\n")

result = pipeline.execute()

print("\nResults:")
print(result.data)

print(f"\nTotal cost: ${result.costs.total_cost:.4f}")

print("\nObservability data available in:")
print("- Console logs (via LoggingObserver)")
print("- OpenTelemetry traces (if exporter configured)")
print("- Langfuse dashboard (if enabled)")
