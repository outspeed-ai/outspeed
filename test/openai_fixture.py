from unittest import mock
import pytest
import asyncio
import json


@pytest.fixture
def mock_openai_client():
    """Fixture to mock the AsyncOpenAI client with dummy responses."""

    async def mock_create_side_effect(*args, **kwargs):
        """Mock the `chat.completions.create` method to return dummy response chunks."""

        response_format = kwargs.get("response_format")
        tool_choice = kwargs.get("tool_choice")

        if response_format and response_format.get("type") == "json":
            MockChatCompletion = mock.Mock(
                choices=[
                    mock.Mock(message=mock.Mock(content=json.dumps({"message": "Hello, this is a mocked response."})))
                ]
            )
        else:
            MockChatCompletion = mock.Mock(
                choices=[mock.Mock(message=mock.Mock(content="Hello, this is a mocked response."))]
            )

        if tool_choice not in ["auto", "none", "required", None]:
            raise ValueError(f"Invalid tool_choice: {tool_choice}")

        MockChatCompletionStream = mock.Mock(
            choices=[mock.Mock(message=mock.Mock(content="Hello, this is a mocked response."))]
        )

        if "messages" not in kwargs:
            raise TypeError("messages is required")
        if "model" not in kwargs:
            raise TypeError("model is required")

        if kwargs.get("tool_choice") not in ["auto", "none", "required", None]:
            raise ValueError(f"Invalid tool_choice: {kwargs.get('tool_choice')}")

        if kwargs.get("stream"):
            return MockChatCompletionStream
        return MockChatCompletion

    with mock.patch("outspeed.plugins.openai_llm.AsyncOpenAI", new_callable=mock.MagicMock) as MockAsyncOpenAI:
        mock_client_instance = MockAsyncOpenAI.return_value
        mock_client_instance.chat.completions.create = mock_create_side_effect
        yield mock_client_instance
