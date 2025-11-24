"""
Tests for Azure OpenAI via UnifiedLiteLLMClient.

NOTE: Azure Managed Identity is now handled by LiteLLM natively.
LiteLLM supports Azure via environment variables:
- AZURE_API_KEY for API key auth
- AZURE_AD_TOKEN for token-based auth

The complex Managed Identity logic (DefaultAzureCredential) is delegated to LiteLLM.
We just need to test that our client correctly formats the model identifier.
"""

from unittest.mock import patch

from ondine.adapters.unified_litellm_client import UnifiedLiteLLMClient
from ondine.core.specifications import LLMProvider, LLMSpec


class TestAzureViaUnifiedClient:
    """Test Azure OpenAI support in UnifiedLiteLLMClient."""

    def test_azure_model_identifier_format(self):
        """Test that Azure uses correct model identifier format."""
        spec = LLMSpec(
            provider=LLMProvider.AZURE_OPENAI,
            model="gpt-4",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="gpt-4-deployment",
            api_key="test-key",  # pragma: allowlist secret  # pragma: allowlist secret
        )

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

            # LiteLLM expects: azure/deployment-name
            assert client.model_identifier == "azure/gpt-4-deployment"

    def test_azure_sets_api_key_env_var(self):
        """Test that Azure API key is set in environment."""
        spec = LLMSpec(
            provider=LLMProvider.AZURE_OPENAI,
            model="gpt-4",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="gpt-4",
            api_key="azure-test-key",  # pragma: allowlist secret
        )

        env_dict = {}
        with patch("ondine.adapters.unified_litellm_client.os.environ", env_dict):
            UnifiedLiteLLMClient(spec)

            assert "AZURE_API_KEY" in env_dict
            assert env_dict["AZURE_API_KEY"] == "azure-test-key"  # noqa: S105 # pragma: allowlist secret

    def test_azure_with_deployment_and_endpoint(self):
        """Test that Azure spec accepts deployment and endpoint."""
        spec = LLMSpec(
            provider=LLMProvider.AZURE_OPENAI,
            model="gpt-4",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="gpt-4-deployment",
            api_key="test-key",  # pragma: allowlist secret  # pragma: allowlist secret
        )

        # Should be valid
        assert spec.azure_deployment == "gpt-4-deployment"
        assert spec.azure_endpoint == "https://test.openai.azure.com/"


# NOTE: Advanced Azure Managed Identity tests removed
# Rationale: LiteLLM handles Managed Identity natively via environment variables.
# Users should set AZURE_AD_TOKEN environment variable for token-based auth.
# We don't need to test DefaultAzureCredential logic - that's LiteLLM's responsibility.
