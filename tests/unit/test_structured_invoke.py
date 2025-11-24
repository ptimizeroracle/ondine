from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from ondine.adapters.llm_client import OpenAIClient
from ondine.core.specifications import LLMProvider, LLMSpec


# Define a sample Pydantic model for testing
class TestModel(BaseModel):
    field1: str
    field2: int


@pytest.fixture
def mock_openai_client():
    spec = LLMSpec(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        api_key="dummy",
        temperature=0.0,
    )
    # Patch OpenAI constructor to avoid actual API connection
    with patch("ondine.adapters.llm_client.OpenAI"):
        client = OpenAIClient(spec)
        # Set up the mock LlamaIndex client
        client.client = MagicMock()
        yield client


def test_structured_invoke_with_structured_predict(mock_openai_client):
    """Test structured_invoke when underlying client supports structured_predict."""
    # Setup mock
    mock_result = TestModel(field1="test", field2=123)

    # Mock structured_predict method on the LlamaIndex client wrapper
    mock_openai_client.client.structured_predict.return_value = mock_result

    # Mock estimate_tokens
    mock_openai_client.estimate_tokens = MagicMock(return_value=10)

    # Execute
    response = mock_openai_client.structured_invoke("prompt", TestModel)

    # Assertions - prompt is now wrapped in PromptTemplate
    assert mock_openai_client.client.structured_predict.called
    call_args = mock_openai_client.client.structured_predict.call_args
    assert call_args[0][0] == TestModel  # First positional arg is output_cls
    # Prompt is wrapped in PromptTemplate, so check the template string
    assert call_args[1]["prompt"].template == "prompt"

    assert response.text == mock_result.model_dump_json()
    assert response.model == "gpt-4o-mini"
    assert response.tokens_in == 10
    assert response.tokens_out == 10


def test_structured_invoke_fallback(mock_openai_client):
    """Test structured_invoke fallback to LLMTextCompletionProgram."""
    # Remove structured_predict attribute to simulate older version/unsupported LLM
    del mock_openai_client.client.structured_predict

    mock_result = TestModel(field1="fallback", field2=456)

    # Mock LLMTextCompletionProgram
    with patch(
        "llama_index.core.program.LLMTextCompletionProgram.from_defaults"
    ) as MockProgram:
        program_instance = MagicMock()
        program_instance.return_value = mock_result
        MockProgram.return_value = program_instance

        # Mock estimate_tokens
        mock_openai_client.estimate_tokens = MagicMock(return_value=10)

        # Execute
        response = mock_openai_client.structured_invoke("prompt", TestModel)

        # Assertions
        MockProgram.assert_called_once_with(
            output_cls=TestModel,
            llm=mock_openai_client.client,
            prompt_template_str="{prompt}",
        )
        program_instance.assert_called_once_with(prompt="prompt")
        assert response.text == mock_result.model_dump_json()


def test_structured_invoke_error_handling(mock_openai_client):
    """Test error handling in structured_invoke."""
    # Setup mock to raise exception
    mock_openai_client.client.structured_predict.side_effect = Exception("API Error")

    # Execute and assert
    with pytest.raises(ValueError) as exc:
        mock_openai_client.structured_invoke("prompt", TestModel)

    assert "Structured prediction failed: API Error" in str(exc.value)
