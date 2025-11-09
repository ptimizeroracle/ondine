"""Tests for Azure Managed Identity authentication."""

from unittest.mock import Mock, patch

import pytest

from ondine.adapters.llm_client import AzureOpenAIClient
from ondine.core.specifications import LLMProvider, LLMSpec


class TestAzureManagedIdentity:
    """Test Azure Managed Identity authentication."""

    def test_managed_identity_enabled(self):
        """Test that use_managed_identity flag is accepted."""
        spec = LLMSpec(
            provider=LLMProvider.AZURE_OPENAI,
            model="gpt-4",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="gpt-4",
            use_managed_identity=True,
        )
        assert spec.use_managed_identity is True

    @patch("ondine.adapters.llm_client.AzureOpenAI")
    @patch("azure.identity.DefaultAzureCredential")
    def test_managed_identity_authentication(self, mock_credential, mock_azure_client):
        """Test authentication with Managed Identity."""
        # Mock Azure credential and token
        mock_token = Mock()
        mock_token.token = "mock_azure_ad_token"  # noqa: S105
        mock_credential.return_value.get_token.return_value = mock_token

        spec = LLMSpec(
            provider=LLMProvider.AZURE_OPENAI,
            model="gpt-4",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="gpt-4",
            use_managed_identity=True,
        )

        AzureOpenAIClient(spec)

        # Verify DefaultAzureCredential was used
        mock_credential.assert_called_once()
        mock_credential.return_value.get_token.assert_called_once_with(
            "https://cognitiveservices.azure.com/.default"
        )

        # Verify AzureOpenAI client was initialized with token
        mock_azure_client.assert_called_once()
        call_kwargs = mock_azure_client.call_args.kwargs
        assert call_kwargs["azure_ad_token"] == "mock_azure_ad_token"  # noqa: S105
        assert "api_key" not in call_kwargs

    def test_managed_identity_missing_dependency(self):
        """Test error when azure-identity not installed."""
        spec = LLMSpec(
            provider=LLMProvider.AZURE_OPENAI,
            model="gpt-4",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="gpt-4",
            use_managed_identity=True,
        )

        with (
            patch.dict("sys.modules", {"azure.identity": None}),
            pytest.raises(ImportError, match="azure-identity"),
        ):
            AzureOpenAIClient(spec)

    @patch("ondine.adapters.llm_client.AzureOpenAI")
    def test_pre_fetched_token(self, mock_azure_client):
        """Test authentication with pre-fetched Azure AD token."""
        spec = LLMSpec(
            provider=LLMProvider.AZURE_OPENAI,
            model="gpt-4",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="gpt-4",
            azure_ad_token="pre_fetched_token_123",  # noqa: S106
        )

        AzureOpenAIClient(spec)

        # Verify token was used
        call_kwargs = mock_azure_client.call_args.kwargs
        assert call_kwargs["azure_ad_token"] == "pre_fetched_token_123"  # noqa: S105

    @patch("ondine.adapters.llm_client.AzureOpenAI")
    def test_backward_compatibility_api_key(self, mock_azure_client):
        """Test that API key authentication still works (backward compatible)."""
        spec = LLMSpec(
            provider=LLMProvider.AZURE_OPENAI,
            model="gpt-4",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="gpt-4",
            api_key="test_api_key",  # pragma: allowlist secret
        )

        AzureOpenAIClient(spec)

        # Verify API key was used
        call_kwargs = mock_azure_client.call_args.kwargs
        assert call_kwargs["api_key"] == "test_api_key"  # pragma: allowlist secret
        assert "azure_ad_token" not in call_kwargs

    @patch("ondine.adapters.llm_client.AzureOpenAI")
    @patch("azure.identity.DefaultAzureCredential")
    def test_managed_identity_authentication_failure(
        self, mock_credential, mock_azure_client
    ):
        """Test error handling when Managed Identity authentication fails."""
        # Mock credential failure
        mock_credential.return_value.get_token.side_effect = Exception(
            "No Managed Identity found"
        )

        spec = LLMSpec(
            provider=LLMProvider.AZURE_OPENAI,
            model="gpt-4",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="gpt-4",
            use_managed_identity=True,
        )

        with pytest.raises(ValueError, match="Failed to authenticate"):
            AzureOpenAIClient(spec)

    def test_missing_azure_endpoint(self):
        """Test error when azure_endpoint is missing."""
        spec = LLMSpec(
            provider=LLMProvider.AZURE_OPENAI,
            model="gpt-4",
            azure_deployment="gpt-4",
            use_managed_identity=True,
        )

        with pytest.raises(ValueError, match="azure_endpoint required"):
            AzureOpenAIClient(spec)

    def test_missing_azure_deployment(self):
        """Test error when azure_deployment is missing."""
        spec = LLMSpec(
            provider=LLMProvider.AZURE_OPENAI,
            model="gpt-4",
            azure_endpoint="https://test.openai.azure.com/",
            use_managed_identity=True,
        )

        with pytest.raises(ValueError, match="azure_deployment required"):
            AzureOpenAIClient(spec)

    def test_no_authentication_provided(self):
        """Test error when no authentication method is provided."""
        spec = LLMSpec(
            provider=LLMProvider.AZURE_OPENAI,
            model="gpt-4",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="gpt-4",
            # No api_key, no use_managed_identity, no azure_ad_token
        )

        with pytest.raises(ValueError, match="Azure OpenAI requires either"):
            AzureOpenAIClient(spec)

    @patch("ondine.adapters.llm_client.AzureOpenAI")
    @patch("azure.identity.DefaultAzureCredential")
    def test_priority_managed_identity_over_api_key(
        self, mock_credential, mock_azure_client
    ):
        """Test that Managed Identity takes priority over API key."""
        # Mock Azure credential and token
        mock_token = Mock()
        mock_token.token = "mock_azure_ad_token"  # noqa: S105
        mock_credential.return_value.get_token.return_value = mock_token

        spec = LLMSpec(
            provider=LLMProvider.AZURE_OPENAI,
            model="gpt-4",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="gpt-4",
            use_managed_identity=True,
            api_key="should_not_be_used",  # Should be ignored  # pragma: allowlist secret
        )

        AzureOpenAIClient(spec)

        # Verify token was used, not API key
        call_kwargs = mock_azure_client.call_args.kwargs
        assert call_kwargs["azure_ad_token"] == "mock_azure_ad_token"  # noqa: S105
        assert "api_key" not in call_kwargs

    @patch("ondine.adapters.llm_client.AzureOpenAI")
    def test_priority_token_over_api_key(self, mock_azure_client):
        """Test that azure_ad_token takes priority over API key."""
        spec = LLMSpec(
            provider=LLMProvider.AZURE_OPENAI,
            model="gpt-4",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="gpt-4",
            azure_ad_token="pre_fetched_token",  # noqa: S106
            api_key="should_not_be_used",  # Should be ignored  # pragma: allowlist secret
        )

        AzureOpenAIClient(spec)

        # Verify token was used, not API key
        call_kwargs = mock_azure_client.call_args.kwargs
        assert call_kwargs["azure_ad_token"] == "pre_fetched_token"  # noqa: S105
        assert "api_key" not in call_kwargs
