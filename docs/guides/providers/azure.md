# Azure OpenAI Provider

Configure and use Azure OpenAI Service with Ondine.

## Setup

```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

## Basic Usage

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Process: {text}")
    .with_llm(
        provider="azure_openai",
        model="gpt-4",
        azure_endpoint="https://your-resource.openai.azure.com/",
        azure_deployment="your-deployment-name",
        api_version="2024-02-15-preview"
    )
    .build()
)

result = pipeline.execute()
```

## Managed Identity Authentication (Recommended for Azure)

For applications running on Azure infrastructure (VMs, Container Apps, Functions), use Managed Identity for keyless authentication:

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Process: {text}")
    .with_llm(
        provider="azure_openai",
        model="gpt-4",
        azure_endpoint="https://your-resource.openai.azure.com/",
        azure_deployment="your-deployment-name",
        use_managed_identity=True  # No API key needed!
    )
    .build()
)

result = pipeline.execute()
```

### Prerequisites

1. Assign a Managed Identity to your Azure resource
2. Grant the identity "Cognitive Services OpenAI User" role on your Azure OpenAI resource
3. Install azure-identity: `pip install ondine[azure]`

### How It Works

- **Production (Azure VM/Container)**: Uses system-assigned or user-assigned Managed Identity
- **Local Development**: Uses Azure CLI credentials (`az login`)
- **CI/CD**: Uses service principal or federated credentials

No API keys stored in code or environment variables!

## Configuration

The deployment name in Azure OpenAI maps to the model:

```python
.with_llm(
    provider="azure_openai",
    model="gpt-4",  # Your base model
    azure_deployment="my-gpt4-deployment",  # Your Azure deployment name
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-15-preview"
)
```

## Rate Limits

Rate limits in Azure OpenAI are configured per deployment. Adjust concurrency based on your TPM (tokens per minute) limits.

## Related

- [OpenAI](openai.md)
- [Cost Control](../cost-control.md)
