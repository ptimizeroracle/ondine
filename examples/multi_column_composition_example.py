"""
Example: Multi-Column Composition with Independent Prompts

Demonstrates how to generate multiple output columns, each with its own
processing logic, using PipelineComposer.

Use Cases:
1. One-to-Many: Single prompt generates multiple columns (JSON)
2. One-to-One: Each column has independent prompt
3. Dependencies: Column B uses Column A as input
"""

import pandas as pd

from ondine.api import PipelineBuilder, PipelineComposer

# Sample data: E-commerce product matching
product_data = pd.DataFrame(
    {
        "current_product": [
            "Apple iPhone 14 Pro Max 256GB Space Black",
            "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
        ],
        "candidate_product": [
            "Apple iPhone 14 Pro 256GB Midnight",
            "Sony WH-1000XM4 Wireless Noise Cancelling Over-Ear Headphones",
        ],
    }
)


def example_1_python_api():
    """Example 1: Python API (programmatic composition)"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Python API - E-commerce Product Matching")
    print("=" * 70)

    # Build Pipeline 1: Match score
    match_pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            product_data,
            input_columns=["current_product", "candidate_product"],
            output_columns=["match_score"],
        )
        .with_prompt(
            template="Rate product match similarity (0-100%): '{current_product}' vs '{candidate_product}'",
        )
        .with_llm(
            provider="groq",
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=10,
        )
        .with_batch_size(100)
        .build()
    )

    # Build Pipeline 2: Match explanation (uses match score)
    explanation_pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            product_data,
            input_columns=["current_product", "candidate_product", "match_score"],
            output_columns=["explanation"],
        )
        .with_prompt(
            template="Explain why '{current_product}' and '{candidate_product}' have a {match_score}% match score. Consider: brand, model, features, specs.",
        )
        .with_llm(
            provider="groq",
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=200,
        )
        .with_batch_size(50)
        .build()
    )

    # Compose pipelines
    composer = PipelineComposer(input_data=product_data)

    result = (
        composer.add_column("match_score", match_pipeline)
        .add_column("explanation", explanation_pipeline, depends_on=["match_score"])
        .execute()
    )

    print(f"\n✓ Processed {len(result.data)} rows")
    print(f"✓ Generated columns: {list(result.data.columns)}")
    print(f"✓ Total cost: ${result.costs.total_cost:.4f}")
    print(f"✓ Errors: {len(result.errors)}")
    print("\nSample output:")
    print(result.data.head())


def example_2_yaml_api():
    """Example 2: YAML API (declarative composition)"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: YAML API - Declarative Composition")
    print("=" * 70)

    # Load composition config
    composer = PipelineComposer.from_yaml("examples/composition_example.yaml")

    # Execute
    result = composer.execute()

    print(f"\n✓ Processed {len(result.data)} rows")
    print(f"✓ Generated columns: {list(result.data.columns)}")
    print(f"✓ Total cost: ${result.costs.total_cost:.4f}")


def example_3_json_multi_output():
    """Example 3: JSON multi-output (single prompt, multiple columns)"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: JSON Multi-Output (1 prompt → 2 columns)")
    print("=" * 70)

    # Single pipeline with JSON output
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            product_data,
            input_columns=["current_product", "candidate_product"],
            output_columns=["match_score", "explanation"],  # Both columns!
        )
        .with_prompt(
            template="""
Analyze product match and explain in JSON format:

Current: {current_product}
Candidate: {candidate_product}

Return JSON:
{{
  "match_score": "85%",
  "explanation": "Same brand and product line, but different generation/model..."
}}
""",
        )
        .with_llm(
            provider="groq",
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=300,
        )
        .build()
    )

    # Execute single pipeline
    result = pipeline.execute()

    print(f"\n✓ Single LLM call generated {len(result.data.columns)} columns")
    print(f"✓ Columns: {list(result.data.columns)}")
    print(f"✓ Cost: ${result.costs.total_cost:.4f}")
    print("\nSample output:")
    print(result.data.head())


def main():
    """Run all examples."""
    print("\n" + "Multi-Column Processing Examples")

    # Example 1: Python API
    example_1_python_api()

    # Example 2: YAML API (template only - requires creating referenced config files)
    # example_2_yaml_api()  # Uncomment when you create the referenced config files

    # Example 3: JSON multi-output
    # example_3_json_multi_output()  # Uncomment to test JSON mode

    print("\n" + "=" * 70)
    print("✓ All examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
