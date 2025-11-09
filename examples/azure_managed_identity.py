"""
Azure OpenAI with Managed Identity Authentication

This example shows how to use Azure Managed Identity for keyless authentication.

Prerequisites:
1. Running on Azure infrastructure (VM, Container App, Function, etc.)
2. Managed Identity assigned to the resource
3. Identity has "Cognitive Services OpenAI User" role
4. Install: pip install ondine[azure]

For local development:
- Run `az login` to authenticate with Azure CLI
- DefaultAzureCredential will use your Azure CLI credentials
"""

from ondine import PipelineBuilder


def main():
    print("Azure OpenAI with Managed Identity Example")
    print("=" * 50)

    # Create pipeline with Managed Identity authentication
    pipeline = (
        PipelineBuilder.create()
        .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
        .with_prompt("Summarize: {text}")
        .with_llm(
            provider="azure_openai",
            model="gpt-4",
            azure_endpoint="https://your-resource.openai.azure.com/",
            azure_deployment="gpt-4-deployment",
            use_managed_identity=True,  # Keyless authentication!
            temperature=0.7,
        )
        .with_batch_size(10)
        .with_concurrency(5)
        .build()
    )

    print("\nExecuting pipeline with Managed Identity...")
    result = pipeline.execute()

    print(f"\nProcessed {result.metrics.total_rows} rows")
    print(f"Cost: ${result.costs.total_cost:.4f}")
    print(f"Success rate: {result.metrics.success_count}/{result.metrics.total_rows}")


if __name__ == "__main__":
    main()
