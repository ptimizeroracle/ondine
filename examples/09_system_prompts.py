"""
System Prompts Example - Using system messages with transformation prompts.

This example demonstrates how to use system messages to guide LLM behavior
while keeping the transformation prompt separate.

Use cases span multiple industries:
- Customer Support (ticket classification)
- E-commerce (product data cleaning)
- Finance (transaction categorization)
"""

import pandas as pd

from ondine import PipelineBuilder

# Sample customer feedback data
data = pd.DataFrame(
    {
        "customer_feedback": [
            "The app crashes every time I try to upload a photo",
            "I was charged twice for the same order last month",
            "How do I change my email address on my account?",
            "My package says delivered but I never received it",
        ],
    }
)


def example_1_basic_system_prompt():
    """Example 1: Basic system prompt for customer support classification."""
    print("=" * 70)
    print("EXAMPLE 1: Customer Support Ticket Classification")
    print("=" * 70)

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            data,
            input_columns=["customer_feedback"],
            output_columns=["category"],
        )
        .with_prompt(
            template="Classify this customer feedback: {customer_feedback}",
            system_message="You are a customer support specialist. "
            "Classify feedback into these categories: "
            "Technical, Billing, Account, Shipping, or General. "
            "Rules: "
            "1) Use only one category "
            "2) Be consistent "
            "3) Choose the most specific category",
        )
        .with_llm(
            provider="groq",
            model="openai/gpt-oss-120b",
            temperature=0.0,
        )
        .build()
    )

    print("\nSystem Message:")
    print(pipeline.specifications.prompt.system_message)
    print("\nTransformation Prompt:")
    print(pipeline.specifications.prompt.template)

    # Estimate cost first
    estimate = pipeline.estimate_cost()
    print(f"\nEstimated cost: ${estimate.total_cost}")
    print(f"Rows to process: {estimate.rows}")

    # Execute
    result = pipeline.execute()

    print("\n✅ Results:")
    for i, row in result.data.iterrows():
        print(f"\nFeedback: {row['customer_feedback']}")
        print(f"Category: {row['category']}")


def example_2_different_personas():
    """Example 2: Different system prompts for different personas."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Different Personas via System Prompts")
    print("=" * 70)

    review = "The product arrived late and the packaging was damaged."

    # Professional response
    pipeline_professional = (
        PipelineBuilder.create()
        .from_dataframe(
            pd.DataFrame({"review": [review]}),
            input_columns=["review"],
            output_columns=["response"],
        )
        .with_prompt(
            template="Write a response to this customer review: {review}",
            system_message="You are a professional customer service representative. "
            "Be empathetic, apologetic, and offer solutions. "
            "Keep responses concise and actionable.",
        )
        .with_llm(provider="groq", model="openai/gpt-oss-120b", temperature=0.3)
        .build()
    )

    # Casual response
    pipeline_casual = (
        PipelineBuilder.create()
        .from_dataframe(
            pd.DataFrame({"review": [review]}),
            input_columns=["review"],
            output_columns=["response"],
        )
        .with_prompt(
            template="Write a response to this customer review: {review}",
            system_message="You are a friendly, casual customer support rep. "
            "Be warm, understanding, and conversational. "
            "Use a casual tone but remain helpful.",
        )
        .with_llm(provider="groq", model="openai/gpt-oss-120b", temperature=0.3)
        .build()
    )

    print("\nReview:", review)
    print("\n--- PROFESSIONAL PERSONA ---")
    result_pro = pipeline_professional.execute()
    print(result_pro.data.iloc[0]["response"])

    print("\n--- CASUAL PERSONA ---")
    result_casual = pipeline_casual.execute()
    print(result_casual.data.iloc[0]["response"])


def example_3_json_extraction_with_system():
    """Example 3: System prompt for structured output."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Structured Output with System Prompt")
    print("=" * 70)

    from ondine.stages import JSONParser

    products = pd.DataFrame(
        {
            "text": [
                "Wireless Mouse Logitech MX Master 3, $99.99",
                "USB-C Cable 6ft Braided Nylon, $12.99",
            ]
        }
    )

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            products,
            input_columns=["text"],
            output_columns=["product", "quantity", "price"],
        )
        .with_prompt(
            template="Extract product info from: {text}",
            system_message="You are a data extraction specialist. "
            "Always respond with valid JSON only, no explanations. "
            'Format: {"product": "name", "quantity": "amount", "price": "$X.XX"}',
        )
        .with_llm(provider="groq", model="openai/gpt-oss-120b", temperature=0.0)
        .with_parser(JSONParser(strict=False))
        .build()
    )

    result = pipeline.execute()

    print("\n✅ Extracted Data:")
    print(result.data.to_string(index=False))


def example_4_yaml_config_with_system():
    """Example 4: System prompt in YAML configuration for finance."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Finance Transaction Categorization")
    print("=" * 70)

    # Create YAML config with system message
    yaml_config = """
dataset:
  source_type: dataframe
  input_columns: [transaction_description]
  output_columns: [category]

prompt:
  template: "Categorize this transaction: {transaction_description}"
  system_message: |
    You are a financial categorization expert.
    Categories: Software, Travel, Marketing, Office, Equipment, Dining
    Rules:
    - Choose only one category
    - Be consistent
    - Consider business expense type

llm:
  provider: groq
  model: openai/gpt-oss-120b
  temperature: 0.0

processing:
  batch_size: 10
  concurrency: 2
"""

    import tempfile

    from ondine.config import ConfigLoader

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_config)
        config_path = f.name

    # Load from config
    specs = ConfigLoader.from_yaml(config_path)

    print("\nLoaded system message from YAML:")
    print(specs.prompt.system_message)

    # Use it
    df = pd.DataFrame(
        {
            "transaction_description": [
                "AWS Cloud Services - Monthly",
                "United Airlines - SFO to NYC",
                "Google Ads Campaign Q1",
            ]
        }
    )

    from ondine.api import Pipeline

    pipeline = Pipeline(specs, dataframe=df)
    result = pipeline.execute()

    print("\n✅ Categorization Results:")
    for i, row in result.data.iterrows():
        print(f"{row['transaction_description']}: {row['category']}")


def example_5_cli_with_system_prompt():
    """Example 5: Using system prompts via CLI."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: System Prompt via CLI")
    print("=" * 70)

    print("\nYou can use system prompts in the CLI by including them in your config:")
    print("""
# config.yaml
dataset:
  source_type: csv
  input_columns: [text]
  output_columns: [result]

prompt:
  template: "Process: {text}"
  system_message: "You are a helpful assistant."

llm:
  provider: groq
  model: openai/gpt-oss-120b
""")

    print("\nThen run:")
    print("$ llm-dataset process -c config.yaml -i data.csv -o result.csv")
    print("\n✅ The system message will be automatically included!")


# Main execution
if __name__ == "__main__":
    import os

    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  Set GROQ_API_KEY environment variable to run examples")
        print("Example: export GROQ_API_KEY='your-key-here'")
        exit(1)

    print("🚀 System Prompts Examples - LLM Dataset Engine\n")

    # Run examples
    try:
        example_1_basic_system_prompt()
        example_2_different_personas()
        example_3_json_extraction_with_system()
        example_4_yaml_config_with_system()
        example_5_cli_with_system_prompt()

        print("\n" + "=" * 70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
