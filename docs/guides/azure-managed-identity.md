# Azure Managed Identity Authentication

Complete guide for using Azure Managed Identity with Ondine for keyless, secure authentication.

## Overview

Azure Managed Identity provides keyless authentication to Azure OpenAI Service, eliminating the need to store API keys in your code or environment variables. This is the **recommended approach** for production deployments on Azure.

## Benefits

- 🔒 **No secrets in code** - Zero API keys to manage
- 🔄 **Automatic rotation** - Azure AD handles credential lifecycle
- 🎯 **Fine-grained access** - Use Azure RBAC for permissions
- 📊 **Audit trail** - All access logged through Azure AD
- 🚀 **Works everywhere** - Production, staging, and local development

## Quick Start

### 1. Install Azure Dependencies

```bash
pip install ondine[azure]
```

This installs `azure-identity>=1.15.0` for Managed Identity support.

### 2. Configure Azure Resources

```bash
# Assign Managed Identity to your Azure resource
az vm identity assign --name my-vm --resource-group my-rg

# Grant "Cognitive Services OpenAI User" role
az role assignment create \
  --assignee <managed-identity-principal-id> \
  --role "Cognitive Services OpenAI User" \
  --scope /subscriptions/<sub-id>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<openai-resource>
```

### 3. Use in Your Code

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
        azure_deployment="gpt-4-deployment",
        use_managed_identity=True  # ← No API key needed!
    )
    .build()
)

result = pipeline.execute()
```

## Usage Examples

### Programmatic API (Python)

#### Example 1: Basic Usage

```python
from ondine import PipelineBuilder
import pandas as pd

# Sample data
data = pd.DataFrame({
    "product_description": [
        "Wireless Bluetooth headphones with noise cancellation",
        "Stainless steel water bottle, 32oz capacity",
    ]
})

# Build pipeline with Managed Identity
pipeline = (
    PipelineBuilder.create()
    .from_dataframe(
        data,
        input_columns=["product_description"],
        output_columns=["category", "features"]
    )
    .with_prompt("""
Analyze: {product_description}

Provide:
1. category: Product category
2. features: Key features (comma-separated)

Format as JSON: {{"category": "...", "features": "..."}}
""")
    .with_llm(
        provider="azure_openai",
        model="gpt-4",
        azure_endpoint="https://your-resource.openai.azure.com/",
        azure_deployment="gpt-4-deployment",
        use_managed_identity=True,
        temperature=0.7,
    )
    .with_batch_size(10)
    .with_concurrency(5)
    .build()
)

result = pipeline.execute()
print(f"Processed: {result.metrics.total_rows} rows")
print(f"Cost: ${result.costs.total_cost:.4f}")
```

#### Example 2: Environment-Aware Configuration

```python
import os
from ondine import PipelineBuilder

# Detect if running on Azure
is_azure = os.getenv("WEBSITE_INSTANCE_ID") or os.getenv("MSI_ENDPOINT")

builder = PipelineBuilder.create().from_csv("data.csv", ...)

if is_azure:
    # Production: Use Managed Identity
    builder.with_llm(
        provider="azure_openai",
        model="gpt-4",
        azure_endpoint="https://your-resource.openai.azure.com/",
        azure_deployment="gpt-4-deployment",
        use_managed_identity=True
    )
else:
    # Development: Use API Key
    builder.with_llm(
        provider="azure_openai",
        model="gpt-4",
        azure_endpoint="https://your-resource.openai.azure.com/",
        azure_deployment="gpt-4-deployment"
        # Falls back to AZURE_OPENAI_API_KEY env var
    )

pipeline = builder.build()
```

### CLI with YAML Configuration

#### Example 1: Managed Identity Config

Create `azure_managed_identity.yaml`:

```yaml
name: "product_enrichment_pipeline"
version: "1.0"

dataset:
  source_type: "csv"
  source_path: "products.csv"
  input_columns:
    - "product_description"
  output_columns:
    - "category"
    - "key_features"

prompt:
  template: "Analyze this product: {product_description}\n\nProvide category and key features as JSON."
  response_format: "json"

llm:
  provider: "azure_openai"
  model: "gpt-4"
  azure_endpoint: "https://your-resource.openai.azure.com/"
  azure_deployment: "gpt-4-deployment"
  use_managed_identity: true  # ← Keyless authentication!
  temperature: 0.7

processing:
  batch_size: 50
  concurrency: 10

output:
  destination_type: "csv"
  destination_path: "enriched_products.csv"
```

Run with CLI:

```bash
ondine process --config azure_managed_identity.yaml
```

#### Example 2: API Key Config (Traditional)

Create `azure_api_key.yaml`:

```yaml
name: "support_ticket_classification"
version: "1.0"

dataset:
  source_type: "csv"
  source_path: "tickets.csv"
  input_columns:
    - "ticket_description"
  output_columns:
    - "category"

prompt:
  template: "Classify: {ticket_description}"

llm:
  provider: "azure_openai"
  model: "gpt-4"
  azure_endpoint: "https://your-resource.openai.azure.com/"
  azure_deployment: "gpt-4-deployment"
  # No use_managed_identity → falls back to AZURE_OPENAI_API_KEY

processing:
  batch_size: 100

output:
  destination_type: "csv"
  destination_path: "classified_tickets.csv"
```

Run with CLI:

```bash
export AZURE_OPENAI_API_KEY="your-key-here"  # pragma: allowlist secret
ondine process --config azure_api_key.yaml
```

## Environment Setup

### Local Development

```bash
# 1. Install dependencies
pip install ondine[azure]

# 2. Login with Azure CLI
az login

# 3. Run your pipeline
python your_script.py
# or
ondine process --config your_config.yaml
```

**What happens**: `DefaultAzureCredential` uses your Azure CLI credentials automatically.

### Azure VM / Container Apps

```bash
# 1. Assign Managed Identity (one-time setup)
az vm identity assign --name my-vm --resource-group my-rg

# 2. Grant RBAC role (one-time setup)
az role assignment create \
  --assignee <managed-identity-principal-id> \
  --role "Cognitive Services OpenAI User" \
  --scope <openai-resource-id>

# 3. Deploy and run your application
# No az login needed - Managed Identity works automatically!
```

### Azure Container Apps

```bash
# Enable Managed Identity
az containerapp identity assign \
  --name my-app \
  --resource-group my-rg \
  --system-assigned

# Grant RBAC role
az role assignment create \
  --assignee <managed-identity-principal-id> \
  --role "Cognitive Services OpenAI User" \
  --scope <openai-resource-id>
```

### Azure Functions

```bash
# Enable Managed Identity
az functionapp identity assign \
  --name my-function \
  --resource-group my-rg

# Grant RBAC role
az role assignment create \
  --assignee <managed-identity-principal-id> \
  --role "Cognitive Services OpenAI User" \
  --scope <openai-resource-id>
```

## Multi-Region Deployment

For global deployments, use region-specific endpoints:

```python
import os

# Define region-specific endpoints
AZURE_REGIONS = {
    "eastus": "https://eastus-openai.openai.azure.com/",
    "westeurope": "https://westeurope-openai.openai.azure.com/",
    "swedencentral": "https://swedencentral-openai.openai.azure.com/",
    "japaneast": "https://japaneast-openai.openai.azure.com/",
}

# Get region from environment
region = os.getenv("AZURE_REGION", "eastus")
endpoint = AZURE_REGIONS[region]

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", ...)
    .with_llm(
        provider="azure_openai",
        model="gpt-4",
        azure_endpoint=endpoint,  # Region-specific
        azure_deployment="gpt-4-deployment",
        use_managed_identity=True
    )
    .build()
)
```

## Troubleshooting

### Error: "azure-identity not installed"

```
ImportError: Azure Managed Identity requires azure-identity.
Install with: pip install ondine[azure]
```

**Solution**:
```bash
pip install ondine[azure]
```

### Error: "Failed to authenticate with Azure Managed Identity"

```
ValueError: Failed to authenticate with Azure Managed Identity: <error>.
Ensure the resource has a Managed Identity assigned with
'Cognitive Services OpenAI User' role.
```

**Solutions**:

1. **Check Managed Identity is assigned**:
```bash
az vm identity show --name my-vm --resource-group my-rg
```

2. **Check RBAC role assignment**:
```bash
az role assignment list \
  --assignee <managed-identity-principal-id> \
  --scope <openai-resource-id>
```

3. **For local development, ensure you're logged in**:
```bash
az login
az account show  # Verify you're logged in
```

### Error: "No authentication provided"

```
ValueError: Azure OpenAI requires either:
  1. use_managed_identity=True (for keyless auth), or
  2. api_key parameter, or
  3. AZURE_OPENAI_API_KEY environment variable
```

**Solution**: Choose one authentication method:

```python
# Option 1: Managed Identity
.with_llm(..., use_managed_identity=True)

# Option 2: API Key  # pragma: allowlist secret
.with_llm(..., api_key="your-key")  # pragma: allowlist secret

# Option 3: Environment variable
export AZURE_OPENAI_API_KEY="your-key"  # pragma: allowlist secret
.with_llm(...)  # Reads from env var
```

## Testing

### Unit Tests (No Azure Required)

```bash
# Run unit tests with mocks
uv run pytest tests/unit/test_azure_managed_identity.py -v
```

All tests use mocks - no Azure credentials needed!

### Integration Tests (Azure Required)

```bash
# Setup Azure resources first
az login

# Run integration tests
uv run pytest tests/integration/ -k azure
```

### Manual Testing

```bash
# 1. Login with Azure CLI
az login

# 2. Run example script
python examples/19_azure_managed_identity_complete.py

# 3. Or use CLI with YAML config
ondine process --config examples/azure_managed_identity_config.yaml
```

## Security Best Practices

### ✅ Do's

1. **Use Managed Identity in production** - No secrets to manage
2. **Use `az login` for local development** - Personal credentials
3. **Grant minimum required RBAC roles** - Principle of least privilege
4. **Use separate identities per environment** - Dev, staging, prod
5. **Monitor Azure AD logs** - Track authentication events

### ❌ Don'ts

1. **Don't hardcode API keys** - Use Managed Identity instead
2. **Don't commit credentials** - Never in source control
3. **Don't use API keys in production** - Use Managed Identity
4. **Don't share Managed Identities** - One per application
5. **Don't skip RBAC** - Always use role-based access control

## Comparison: API Key vs. Managed Identity

| Aspect | API Key | Managed Identity |
|--------|---------|------------------|
| **Security** | ⚠️ Secret to manage | ✅ No secrets |
| **Rotation** | ❌ Manual | ✅ Automatic |
| **Setup** | ✅ Simple | ⚠️ Requires Azure setup |
| **Local Dev** | ✅ Easy (env var) | ✅ Easy (`az login`) |
| **Production** | ❌ Risk of exposure | ✅ Secure by design |
| **Audit Trail** | ⚠️ Limited | ✅ Full Azure AD logs |
| **RBAC** | ❌ No | ✅ Yes |
| **Cost** | Free | Free |

**Recommendation**: Use Managed Identity for production, API key for quick prototyping.

## Examples

### Complete Python Example

See: `examples/19_azure_managed_identity_complete.py`

Includes:
- Managed Identity authentication
- API key authentication
- Pre-fetched token
- Environment-aware configuration
- Multi-region setup

### YAML Configuration Examples

- **Managed Identity**: `examples/azure_managed_identity_config.yaml`
- **API Key**: `examples/azure_api_key_config.yaml`

## Related Documentation

- [Azure OpenAI Provider Guide](providers/azure.md)
- [Installation Guide](../getting-started/installation.md)
- [Technical Reference](../architecture/technical-reference.md#53-azure-openai-provider-enterprise)

## FAQ

### Q: Do I need `az login` in production?

**A**: No! Managed Identity works automatically on Azure infrastructure. `az login` is only for local development.

### Q: Can I use Managed Identity with OpenAI (not Azure OpenAI)?

**A**: No. Managed Identity only works with Azure OpenAI Service. For OpenAI's public API, use API keys.

### Q: How long do tokens last?

**A**: Azure AD tokens typically last 1 hour. For pipelines running longer than 1 hour, consider using API keys or implementing token refresh (future feature).

### Q: Can I use User-Assigned Managed Identity?

**A**: Yes! `DefaultAzureCredential` supports both System-Assigned and User-Assigned Managed Identities. Set the `AZURE_CLIENT_ID` environment variable to specify a User-Assigned identity.

### Q: Does this work with Azure Government or Azure China?

**A**: Yes, but you may need to specify custom token scopes. This is an advanced scenario - contact support for guidance.

## Support

For issues or questions:
- GitHub Issues: https://github.com/ptimizeroracle/Ondine/issues
- Documentation: https://ptimizeroracle.github.io/ondine
