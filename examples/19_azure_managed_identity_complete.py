"""
Complete Azure Managed Identity Usage Examples

This example demonstrates all authentication methods for Azure OpenAI:
1. Managed Identity (recommended for production)
2. API Key from environment variable (traditional)
3. Pre-fetched token (advanced)

Prerequisites:
- Azure OpenAI resource created
- For Managed Identity: pip install ondine[azure]
- For API Key: export AZURE_OPENAI_API_KEY="..."

Author: Ondine Team
"""

import os

import pandas as pd

from ondine import PipelineBuilder


# ============================================================================
# Example 1: Managed Identity (Recommended for Production)
# ============================================================================
def example_managed_identity():
    """
    Use Managed Identity for keyless authentication.

    Works automatically when running on:
    - Azure VM
    - Azure Container Apps
    - Azure Functions
    - Azure Kubernetes Service (AKS)
    - Azure App Service

    For local development:
    - Run: az login
    - DefaultAzureCredential will use your Azure CLI credentials
    """
    print("\n" + "=" * 70)
    print("Example 1: Managed Identity Authentication")
    print("=" * 70)

    # Create sample data
    data = pd.DataFrame(
        {
            "product_description": [
                "Wireless Bluetooth headphones with noise cancellation",
                "Stainless steel water bottle, 32oz capacity",
                "Organic cotton t-shirt, available in multiple colors",
            ]
        }
    )

    # Build pipeline with Managed Identity
    _pipeline = (  # noqa: F841
        PipelineBuilder.create()
        .from_dataframe(
            data,
            input_columns=["product_description"],
            output_columns=["category", "key_features"],
        )
        .with_prompt(
            """
Analyze this product description: {product_description}

Provide:
1. category: The product category
2. key_features: Main features (comma-separated)

Format as JSON:
{{"category": "...", "key_features": "..."}}
"""
        )
        .with_llm(
            provider="azure_openai",
            model="gpt-4",
            azure_endpoint="https://your-resource.openai.azure.com/",
            azure_deployment="gpt-4-deployment",
            use_managed_identity=True,  # ← Keyless authentication!
            temperature=0.7,
        )
        .with_batch_size(10)
        .with_concurrency(3)
        .build()
    )

    print("\n✅ Pipeline configured with Managed Identity")
    print("   - No API key needed")
    print("   - Uses DefaultAzureCredential")
    print("   - Works on Azure infrastructure or with 'az login'\n")

    # Uncomment to execute (requires Azure setup):
    # result = _pipeline.execute()
    # print(f"Processed {result.metrics.total_rows} rows")
    # print(f"Cost: ${result.costs.total_cost:.4f}")
    # print(result.output_data.head())


# ============================================================================
# Example 2: API Key from Environment Variable (Traditional)
# ============================================================================
def example_api_key():
    """
    Use API key from environment variable (backward compatible).

    Setup:
    export AZURE_OPENAI_API_KEY="your-api-key-here"  # pragma: allowlist secret
    """
    print("\n" + "=" * 70)
    print("Example 2: API Key Authentication (Traditional)")
    print("=" * 70)

    data = pd.DataFrame(
        {
            "customer_feedback": [
                "Great product, fast shipping!",
                "Item arrived damaged, very disappointed",
                "Good quality but overpriced",
            ]
        }
    )

    _pipeline = (  # noqa: F841
        PipelineBuilder.create()
        .from_dataframe(
            data,
            input_columns=["customer_feedback"],
            output_columns=["sentiment", "rating"],
        )
        .with_prompt(
            """
Analyze this customer feedback: {customer_feedback}

Provide:
1. sentiment: positive, negative, or neutral
2. rating: estimated rating (1-5)

Format as JSON:
{{"sentiment": "...", "rating": "..."}}
"""
        )
        .with_llm(
            provider="azure_openai",
            model="gpt-4",
            azure_endpoint="https://your-resource.openai.azure.com/",
            azure_deployment="gpt-4-deployment",
            # No use_managed_identity → falls back to AZURE_OPENAI_API_KEY env var
            temperature=0.5,
        )
        .build()
    )

    print("\n✅ Pipeline configured with API Key")
    print("   - Reads from AZURE_OPENAI_API_KEY environment variable")
    print("   - Traditional authentication method\n")

    # Uncomment to execute (requires AZURE_OPENAI_API_KEY):
    # result = _pipeline.execute()


# ============================================================================
# Example 3: Pre-fetched Token (Advanced)
# ============================================================================
def example_pre_fetched_token():
    """
    Use a pre-fetched Azure AD token (advanced use case).

    Useful when:
    - You manage token lifecycle manually
    - You need custom token scopes
    - You integrate with existing auth systems
    """
    print("\n" + "=" * 70)
    print("Example 3: Pre-fetched Azure AD Token (Advanced)")
    print("=" * 70)

    # Fetch token manually (example)
    try:
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential()
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        azure_ad_token = token.token

        print(f"✅ Token fetched: {azure_ad_token[:20]}...")
    except Exception as e:
        print(f"⚠️  Could not fetch token: {e}")
        azure_ad_token = "dummy_token_for_demo"  # nosec B105

    data = pd.DataFrame({"text": ["Sample text for processing"]})

    _pipeline = (  # noqa: F841
        PipelineBuilder.create()
        .from_dataframe(data, input_columns=["text"], output_columns=["result"])
        .with_prompt("Process: {text}")
        .with_llm(
            provider="azure_openai",
            model="gpt-4",
            azure_endpoint="https://your-resource.openai.azure.com/",
            azure_deployment="gpt-4-deployment",
            azure_ad_token=azure_ad_token,  # ← Pre-fetched token!
            temperature=0.7,
        )
        .build()
    )

    print("\n✅ Pipeline configured with pre-fetched token")
    print("   - Token fetched manually")
    print("   - Useful for custom token management\n")


# ============================================================================
# Example 4: Environment-Aware Configuration (Best Practice)
# ============================================================================
def example_environment_aware():
    """
    Automatically use Managed Identity in production, API key in dev.

    This is the BEST PRACTICE for real-world deployments.
    """
    print("\n" + "=" * 70)
    print("Example 4: Environment-Aware Configuration (Best Practice)")
    print("=" * 70)

    # Detect environment
    is_azure_environment = (
        os.getenv("WEBSITE_INSTANCE_ID") is not None
    )  # Azure App Service
    is_azure_environment = (
        is_azure_environment or os.getenv("CONTAINER_APP_NAME") is not None
    )  # Container Apps
    is_azure_environment = (
        is_azure_environment or os.getenv("MSI_ENDPOINT") is not None
    )  # Any Managed Identity

    print(
        f"\n🔍 Detected environment: {'Azure (Managed Identity)' if is_azure_environment else 'Local (API Key)'}"
    )

    data = pd.DataFrame(
        {
            "email_subject": [
                "Order #12345 - Shipping Delay",
                "Billing Question - Duplicate Charge",
                "Product Return Request",
            ]
        }
    )

    # Build pipeline with environment-aware auth
    builder = (
        PipelineBuilder.create()
        .from_dataframe(
            data,
            input_columns=["email_subject"],
            output_columns=["category", "priority"],
        )
        .with_prompt("""
Categorize this email: {email_subject}

Provide:
1. category: shipping, billing, returns, or general
2. priority: high, medium, or low

Format as JSON:
{{"category": "...", "priority": "..."}}
""")
    )

    if is_azure_environment:
        # Production: Use Managed Identity
        builder.with_llm(
            provider="azure_openai",
            model="gpt-4",
            azure_endpoint="https://your-resource.openai.azure.com/",
            azure_deployment="gpt-4-deployment",
            use_managed_identity=True,  # ← Keyless!
        )
        print("✅ Using Managed Identity (production mode)")
    else:
        # Development: Use API Key
        builder.with_llm(
            provider="azure_openai",
            model="gpt-4",
            azure_endpoint="https://your-resource.openai.azure.com/",
            azure_deployment="gpt-4-deployment",
            # Falls back to AZURE_OPENAI_API_KEY env var
        )
        print("✅ Using API Key (development mode)")

    _pipeline = builder.build()  # noqa: F841

    print("   - Same code works in both environments!")
    print("   - No configuration changes needed\n")


# ============================================================================
# Example 5: Multi-Region Setup
# ============================================================================
def example_multi_region():
    """
    Use different Azure OpenAI endpoints per region.

    Common in enterprise deployments for:
    - Data residency compliance
    - Latency optimization
    - High availability
    """
    print("\n" + "=" * 70)
    print("Example 5: Multi-Region Setup")
    print("=" * 70)

    # Detect region (example)
    region = os.getenv("AZURE_REGION", "eastus")

    # Region-specific endpoints
    endpoints = {
        "eastus": "https://eastus-resource.openai.azure.com/",
        "westeurope": "https://westeurope-resource.openai.azure.com/",
        "swedencentral": "https://core-mffmjv6z-swedencentral.openai.azure.com/",
    }

    endpoint = endpoints.get(region, endpoints["eastus"])

    print(f"\n🌍 Using region: {region}")
    print(f"   Endpoint: {endpoint}")

    data = pd.DataFrame({"text": ["Sample text"]})

    _pipeline = (  # noqa: F841
        PipelineBuilder.create()
        .from_dataframe(data, input_columns=["text"], output_columns=["result"])
        .with_prompt("Process: {text}")
        .with_llm(
            provider="azure_openai",
            model="gpt-4",
            azure_endpoint=endpoint,  # ← Region-specific
            azure_deployment="gpt-4-deployment",
            use_managed_identity=True,
        )
        .build()
    )

    print("✅ Pipeline configured for multi-region deployment\n")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("\n" + "🔐 Azure Managed Identity - Complete Usage Examples".center(70))

    # Run all examples
    example_managed_identity()
    example_api_key()
    example_pre_fetched_token()
    example_environment_aware()
    example_multi_region()

    print("\n" + "=" * 70)
    print("📚 Summary")
    print("=" * 70)
    print("""
✅ Managed Identity (Recommended):
   - No API keys needed
   - Automatic on Azure infrastructure
   - Requires: pip install ondine[azure]
   - Local dev: az login

✅ API Key (Traditional):
   - Works everywhere
   - Requires: export AZURE_OPENAI_API_KEY="..."
   - Backward compatible

✅ Pre-fetched Token (Advanced):
   - Manual token management
   - Custom token lifecycle
   - For specialized use cases

📖 Documentation:
   - See: docs/guides/providers/azure.md
   - Example: examples/azure_managed_identity.py
""")
