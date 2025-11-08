# 🔐 Azure Managed Identity - Complete Usage Guide

This guide shows you exactly how to use the new Azure Managed Identity feature in Ondine.

## 📋 Table of Contents

1. [Python Code Examples](#python-code-examples)
2. [CLI with YAML Configuration](#cli-with-yaml-configuration)
3. [Testing Without Azure](#testing-without-azure)
4. [Real Azure Deployment](#real-azure-deployment)

---

## 🐍 Python Code Examples

### Example 1: Managed Identity (Recommended)

```python
from ondine import PipelineBuilder
import pandas as pd

# Sample data
data = pd.DataFrame({
    "product_description": [
        "Wireless Bluetooth headphones with noise cancellation",
        "Stainless steel water bottle, 32oz capacity",
        "Organic cotton t-shirt, multiple colors available",
    ]
})

# Build pipeline with Managed Identity
pipeline = (
    PipelineBuilder.create()
    .from_dataframe(
        data,
        input_columns=["product_description"],
        output_columns=["category", "key_features"]
    )
    .with_prompt("""
Analyze this product: {product_description}

Provide:
1. category: Product category
2. key_features: Main features (comma-separated)

Format as JSON:
{{"category": "...", "key_features": "..."}}
""")
    .with_llm(
        provider="azure_openai",
        model="gpt-4",
        azure_endpoint="https://your-resource.openai.azure.com/",
        azure_deployment="gpt-4-deployment",
        use_managed_identity=True,  # ← NO API KEY NEEDED!
        temperature=0.7,
    )
    .with_processing(batch_size=10, concurrency=5)
    .build()
)

# Execute
result = pipeline.execute()

print(f"✅ Processed: {result.total_rows} rows")
print(f"💰 Cost: ${result.total_cost:.4f}")
print(f"📊 Results:\n{result.output_data}")
```

**When to use:**
- ✅ Production deployments on Azure
- ✅ Maximum security (no secrets)
- ✅ Enterprise compliance requirements

**Setup required:**
```bash
# 1. Install dependencies
pip install ondine[azure]

# 2. For local dev: Login with Azure CLI
az login

# 3. For production: Assign Managed Identity to your Azure resource
```

---

### Example 2: API Key (Traditional - Backward Compatible)

```python
from ondine import PipelineBuilder

# Build pipeline with API Key (existing behavior)
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Process: {text}")
    .with_llm(
        provider="azure_openai",
        model="gpt-4",
        azure_endpoint="https://your-resource.openai.azure.com/",
        azure_deployment="gpt-4-deployment",
        # No use_managed_identity → falls back to AZURE_OPENAI_API_KEY env var
    )
    .build()
)

result = pipeline.execute()
```

**When to use:**
- ✅ Quick prototyping
- ✅ Non-Azure environments
- ✅ Existing code (backward compatible)

**Setup required:**
```bash
export AZURE_OPENAI_API_KEY="your-key-from-azure-portal"
```

---

### Example 3: Environment-Aware (Best Practice)

```python
import os
from ondine import PipelineBuilder

# Automatically detect environment
is_azure = (
    os.getenv("WEBSITE_INSTANCE_ID") or      # Azure App Service
    os.getenv("CONTAINER_APP_NAME") or       # Azure Container Apps
    os.getenv("MSI_ENDPOINT")                # Any Managed Identity
)

builder = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Process: {text}")
)

if is_azure:
    # Production: Use Managed Identity
    builder.with_llm(
        provider="azure_openai",
        model="gpt-4",
        azure_endpoint="https://your-resource.openai.azure.com/",
        azure_deployment="gpt-4-deployment",
        use_managed_identity=True
    )
    print("🔒 Using Managed Identity (production)")
else:
    # Development: Use API Key
    builder.with_llm(
        provider="azure_openai",
        model="gpt-4",
        azure_endpoint="https://your-resource.openai.azure.com/",
        azure_deployment="gpt-4-deployment"
    )
    print("🔑 Using API Key (development)")

pipeline = builder.build()
result = pipeline.execute()
```

**When to use:**
- ✅ Same code for dev and production
- ✅ No configuration changes between environments
- ✅ Best practice for enterprise deployments

---

## 🖥️ CLI with YAML Configuration

### Example 1: Managed Identity Config

Create `azure_managed_identity.yaml`:

```yaml
name: "product_enrichment_pipeline"
version: "1.0"

# Input data
dataset:
  source_type: "csv"
  source_path: "products.csv"
  input_columns:
    - "product_description"
  output_columns:
    - "category"
    - "key_features"
    - "target_audience"

# Prompt template
prompt:
  template: |
    Analyze this product: {product_description}
    
    Provide:
    1. category: Product category
    2. key_features: Main features (comma-separated)
    3. target_audience: Target customer segment
    
    Format as JSON:
    {{"category": "...", "key_features": "...", "target_audience": "..."}}
  
  system_message: "You are a product categorization expert."
  response_format: "json"

# LLM configuration - MANAGED IDENTITY
llm:
  provider: "azure_openai"
  model: "gpt-4"
  
  # Azure-specific
  azure_endpoint: "https://your-resource.openai.azure.com/"
  azure_deployment: "gpt-4-deployment"
  api_version: "2024-02-15-preview"
  
  # Managed Identity (NO API KEY!)
  use_managed_identity: true
  
  # LLM parameters
  temperature: 0.7
  max_tokens: 500

# Processing configuration
processing:
  batch_size: 50
  concurrency: 10
  checkpoint_interval: 100
  max_retries: 3
  error_policy: "skip"
  rate_limit_rpm: 60
  max_budget: "5.00"

# Output configuration
output:
  destination_type: "csv"
  destination_path: "enriched_products.csv"
  merge_strategy: "replace"
  atomic_write: true
```

**Run with CLI:**

```bash
# Install dependencies
pip install ondine[azure]

# For local development
az login

# Run pipeline
ondine process --config azure_managed_identity.yaml

# Output:
# ✅ Processed 1000 rows
# 💰 Total cost: $2.45
# 📁 Output: enriched_products.csv
```

---

### Example 2: API Key Config (Traditional)

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
    - "priority"

prompt:
  template: |
    Classify this support ticket: {ticket_description}
    
    Provide:
    1. category: technical, billing, account, shipping, or general
    2. priority: high, medium, or low
    
    Format as JSON:
    {{"category": "...", "priority": "..."}}
  
  response_format: "json"

llm:
  provider: "azure_openai"
  model: "gpt-4"
  azure_endpoint: "https://your-resource.openai.azure.com/"
  azure_deployment: "gpt-4-deployment"
  # No use_managed_identity → uses API key
  temperature: 0.3

processing:
  batch_size: 100
  concurrency: 5

output:
  destination_type: "csv"
  destination_path: "classified_tickets.csv"
```

**Run with CLI:**

```bash
# Set API key
export AZURE_OPENAI_API_KEY="your-key-here"

# Run pipeline
ondine process --config azure_api_key.yaml
```

---

## 🧪 Testing Without Azure

### Unit Tests (No Azure Required)

All unit tests use mocks - no Azure credentials needed:

```bash
# Run Azure Managed Identity tests
uv run pytest tests/unit/test_azure_managed_identity.py -v

# Output:
# ✅ test_managed_identity_enabled PASSED
# ✅ test_managed_identity_authentication PASSED
# ✅ test_pre_fetched_token PASSED
# ✅ test_backward_compatibility_api_key PASSED
# ... 11 tests passed in 0.59s
```

**How it works:**
- Tests use `@patch` to mock Azure libraries
- No real Azure calls are made
- Tests verify our code logic, not Azure's authentication

### Run Example Scripts Locally

```bash
# 1. Login with Azure CLI (uses your personal credentials)
az login

# 2. Install dependencies
pip install ondine[azure]

# 3. Run example (replace with your Azure endpoint)
python examples/19_azure_managed_identity_complete.py
```

---

## 🚀 Real Azure Deployment

### Step 1: Create Azure OpenAI Resource

```bash
# Create resource group
az group create --name my-rg --location eastus

# Create Azure OpenAI resource
az cognitiveservices account create \
  --name my-openai-resource \
  --resource-group my-rg \
  --kind OpenAI \
  --sku S0 \
  --location eastus

# Create deployment
az cognitiveservices account deployment create \
  --name my-openai-resource \
  --resource-group my-rg \
  --deployment-name gpt-4-deployment \
  --model-name gpt-4 \
  --model-version "0613" \
  --model-format OpenAI \
  --sku-capacity 10 \
  --sku-name Standard
```

### Step 2: Setup Managed Identity

**For Azure VM:**

```bash
# Create VM with Managed Identity
az vm create \
  --name my-vm \
  --resource-group my-rg \
  --image Ubuntu2204 \
  --assign-identity \
  --admin-username azureuser \
  --generate-ssh-keys

# Get Managed Identity principal ID
PRINCIPAL_ID=$(az vm identity show \
  --name my-vm \
  --resource-group my-rg \
  --query principalId -o tsv)

echo "Managed Identity Principal ID: $PRINCIPAL_ID"
```

**For Azure Container Apps:**

```bash
# Create Container App with Managed Identity
az containerapp create \
  --name my-app \
  --resource-group my-rg \
  --environment my-env \
  --image myregistry.azurecr.io/my-app:latest \
  --system-assigned

# Get principal ID
PRINCIPAL_ID=$(az containerapp identity show \
  --name my-app \
  --resource-group my-rg \
  --query principalId -o tsv)
```

### Step 3: Grant RBAC Permissions

```bash
# Get Azure OpenAI resource ID
OPENAI_RESOURCE_ID=$(az cognitiveservices account show \
  --name my-openai-resource \
  --resource-group my-rg \
  --query id -o tsv)

# Grant "Cognitive Services OpenAI User" role
az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Cognitive Services OpenAI User" \
  --scope $OPENAI_RESOURCE_ID

# Verify role assignment
az role assignment list \
  --assignee $PRINCIPAL_ID \
  --scope $OPENAI_RESOURCE_ID
```

### Step 4: Deploy Your Application

**Create `app.py`:**

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Process: {text}")
    .with_llm(
        provider="azure_openai",
        model="gpt-4",
        azure_endpoint="https://my-openai-resource.openai.azure.com/",
        azure_deployment="gpt-4-deployment",
        use_managed_identity=True  # ← Keyless!
    )
    .build()
)

result = pipeline.execute()
print(f"Processed: {result.total_rows} rows")
```

**Deploy to Azure VM:**

```bash
# SSH into VM
ssh azureuser@<vm-ip>

# Install Ondine
pip install ondine[azure]

# Run your application
python app.py

# ✅ Works automatically - no API keys needed!
```

**Deploy to Azure Container Apps:**

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install ondine[azure]

CMD ["python", "app.py"]
```

```bash
# Build and push
docker build -t myregistry.azurecr.io/my-app:latest .
docker push myregistry.azurecr.io/my-app:latest

# Deploy (Managed Identity already assigned)
az containerapp update \
  --name my-app \
  --resource-group my-rg \
  --image myregistry.azurecr.io/my-app:latest
```

---

## 🎯 Quick Reference

### Python API

```python
# Managed Identity
.with_llm(
    provider="azure_openai",
    model="gpt-4",
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_deployment="gpt-4-deployment",
    use_managed_identity=True  # ← This is the key!
)

# API Key (backward compatible)
.with_llm(
    provider="azure_openai",
    model="gpt-4",
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_deployment="gpt-4-deployment"
    # Falls back to AZURE_OPENAI_API_KEY env var
)
```

### YAML Config

```yaml
# Managed Identity
llm:
  provider: "azure_openai"
  model: "gpt-4"
  azure_endpoint: "https://your-resource.openai.azure.com/"
  azure_deployment: "gpt-4-deployment"
  use_managed_identity: true  # ← This is the key!

# API Key (backward compatible)
llm:
  provider: "azure_openai"
  model: "gpt-4"
  azure_endpoint: "https://your-resource.openai.azure.com/"
  azure_deployment: "gpt-4-deployment"
  api_key: ${AZURE_OPENAI_API_KEY}  # From environment
```

### CLI Commands

```bash
# With Managed Identity config
ondine process --config azure_managed_identity.yaml

# With API Key config
export AZURE_OPENAI_API_KEY="your-key"
ondine process --config azure_api_key.yaml

# Estimate cost before running
ondine estimate --config azure_managed_identity.yaml --sample-size 100
```

---

## 🔍 How It Works

### DefaultAzureCredential Chain

When you use `use_managed_identity=True`, Ondine uses Azure's `DefaultAzureCredential`, which tries these methods in order:

```
1. Environment Variables
   ↓ (not found)
2. Managed Identity (on Azure VM/Container/Function)
   ↓ (not found)
3. Azure CLI (az login)
   ↓ (not found)
4. Visual Studio Code
   ↓ (not found)
5. Azure PowerShell
   ↓ (not found)
❌ Error: No credentials available
```

**This means:**
- ✅ **Production (Azure)**: Uses Managed Identity automatically
- ✅ **Local Dev**: Uses your `az login` credentials
- ✅ **CI/CD**: Uses Service Principal from environment variables
- ✅ **No code changes** between environments!

---

## 📊 Comparison Table

| Feature | API Key | Managed Identity |
|---------|---------|------------------|
| **Setup Complexity** | ✅ Simple (1 env var) | ⚠️ Medium (Azure setup) |
| **Security** | ❌ Secret to manage | ✅ No secrets |
| **Rotation** | ❌ Manual | ✅ Automatic |
| **Local Development** | ✅ Easy | ✅ Easy (`az login`) |
| **Production** | ⚠️ Risk of exposure | ✅ Secure |
| **RBAC** | ❌ No | ✅ Yes |
| **Audit Trail** | ⚠️ Limited | ✅ Full Azure AD logs |
| **Cost** | Free | Free |
| **Works Outside Azure** | ✅ Yes | ❌ No (requires Azure) |

---

## 🎓 Complete Examples

### Full Example Files

1. **`examples/19_azure_managed_identity_complete.py`**
   - All authentication methods
   - Environment-aware configuration
   - Multi-region setup
   - Production-ready patterns

2. **`examples/azure_managed_identity_config.yaml`**
   - Complete YAML configuration
   - Managed Identity setup
   - All configuration options

3. **`examples/azure_api_key_config.yaml`**
   - Traditional API key configuration
   - Backward compatible

### Run Examples

```bash
# Python example
python examples/19_azure_managed_identity_complete.py

# CLI with Managed Identity
ondine process --config examples/azure_managed_identity_config.yaml

# CLI with API Key
export AZURE_OPENAI_API_KEY="your-key"
ondine process --config examples/azure_api_key_config.yaml
```

---

## 🔧 Troubleshooting

### "azure-identity not installed"

```bash
pip install ondine[azure]
```

### "Failed to authenticate with Azure Managed Identity"

**Check 1: Managed Identity assigned?**
```bash
az vm identity show --name my-vm --resource-group my-rg
```

**Check 2: RBAC role granted?**
```bash
az role assignment list --assignee <principal-id>
```

**Check 3: For local dev, logged in?**
```bash
az login
az account show
```

### "No authentication provided"

Choose one authentication method:
- `use_managed_identity: true` (Managed Identity)
- `api_key: "..."` (explicit key)
- `export AZURE_OPENAI_API_KEY="..."` (env var)

---

## 📚 Additional Resources

- **Documentation**: [Azure Managed Identity Guide](docs/guides/azure-managed-identity.md)
- **Provider Guide**: [Azure OpenAI Provider](docs/guides/providers/azure.md)
- **Technical Reference**: [Architecture Docs](docs/architecture/technical-reference.md#53-azure-openai-provider-enterprise)
- **Azure Docs**: [Managed Identity Overview](https://learn.microsoft.com/en-us/entra/identity/managed-identities-azure-resources/overview)

---

## ✅ Summary

**To use Azure Managed Identity with Ondine:**

1. **Install**: `pip install ondine[azure]`
2. **Setup Azure**: Assign Managed Identity + grant RBAC role
3. **Local Dev**: `az login`
4. **Code**: Add `use_managed_identity=True` to `.with_llm()`
5. **Run**: Works automatically on Azure, uses `az login` locally

**That's it!** No API keys to manage. 🎉

