"""
Example: LLM-specific observability with Langfuse.

Demonstrates how to use Langfuse for tracking prompts, completions,
tokens, costs, and other LLM-specific metrics.

Setup:
    1. Get Langfuse API keys:
       - Sign up at https://cloud.langfuse.com
       - Or self-host: https://langfuse.com/docs/deployment/self-host

    2. Install Langfuse SDK:
       pip install langfuse

    3. Set environment variables (or pass in config):
       export LANGFUSE_PUBLIC_KEY="pk-lf-..."
       export LANGFUSE_SECRET_KEY="sk-lf-..."

    4. Run this example and view traces in Langfuse UI
"""

import os

import pandas as pd

from ondine import PipelineBuilder

# Get Langfuse credentials from environment
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")

if not public_key or not secret_key:
    raise ValueError(
        "Please set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables. "
        "Get your keys from https://cloud.langfuse.com"
    )

# Sample data
data = pd.DataFrame(
    {
        "question": [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is the speed of light?",
        ]
    }
)

# Build pipeline with Langfuse observer
pipeline = (
    PipelineBuilder.create()
    .from_dataframe(data, input_columns=["question"], output_columns=["answer"])
    .with_prompt(
        """
        Answer the following question concisely.

        Question: {question}
        Answer:"""
    )
    .with_llm(provider="openai", model="gpt-4o-mini", temperature=0.0)
    # Add Langfuse observer for LLM observability
    .with_observer(
        "langfuse",
        config={
            "public_key": public_key,
            "secret_key": secret_key,
            # Optional: "host": "https://cloud.langfuse.com",
        },
    )
    .build()
)

# Execute
print("Executing pipeline with Langfuse observability...")
result = pipeline.execute()

print("\nResults:")
print(result.data)

print(f"\nTotal cost: ${result.costs.total_cost:.4f}")
print(f"Total tokens: {result.costs.total_tokens}")

print("\nView detailed metrics in Langfuse:")
print("https://cloud.langfuse.com")
print("\nLangfuse provides:")
print("- Full prompt and completion tracking")
print("- Token usage and cost analysis")
print("- Latency metrics")
print("- Quality evaluations")
