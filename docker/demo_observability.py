"""
Demo script for Unified Observability (OpenTelemetry + Scalene).

This script demonstrates:
1.  **OpenTelemetry**: Tracing pipeline execution via LiteLLM native callbacks (view in Jaeger).
2.  **Scalene**: Profiling CPU/Memory usage (view in generated HTML).

Usage:
    ./docker/profile.sh docker/demo_observability.py
"""

import logging
import os

import pandas as pd

from ondine import PipelineBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run a sample pipeline with observability enabled."""
    print("🚀 Starting Observability Demo...")

    # 1. Verify Environment (Docker sets these automatically)
    otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    print(
        f"📡 OpenTelemetry Endpoint: {otel_endpoint or 'Not Set (will fallback to default)'}"
    )

    # 2. Create Sample Data
    df = pd.DataFrame(
        {
            "text": [
                "The quick brown fox jumps over the lazy dog.",
                "To be or not to be, that is the question.",
                "I think, therefore I am.",
            ]
        }
    )

    # 3. Build Pipeline with OpenTelemetry Observer
    # Note: The observer now automatically picks up the environment configuration!
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["text"], output_columns=["analysis"])
        .with_prompt(
            """
            Analyze the sentiment and complexity of this text.
            Return a short summary.

            Text: {text}
            """
        )
        # Using a fake model for demo (or OpenAI if key provided)
        # For testing without keys, we can use a mock provider or handle errors
        .with_llm(provider="openai", model="gpt-4o-mini", temperature=0.7)
        .with_observer("opentelemetry")  # Activate the OTel observer
        .build()
    )

    # 4. Execute (Traces will be sent to Jaeger)
    try:
        print("⚡ Executing pipeline...")
        result = pipeline.execute()

        if result.success:
            print("\n✅ Pipeline Success!")
            print(result.data)
            print(f"\n💰 Total Cost: ${result.costs.total_cost:.6f}")
        else:
            print("\n❌ Pipeline Failed")
            print(result.error)

    except Exception as e:
        print(f"\n⚠️ Execution Error (Expected if no API Key): {e}")
        print("NOTE: Even with errors, traces should appear in Jaeger!")

    print("\n--------------------------------------------------")
    print("🔍 View Traces: http://localhost:16686")
    print("   Service Name: ondine-profiling")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
