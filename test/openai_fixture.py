from unittest import mock
import pytest
import asyncio


@pytest.fixture
def mock_openai_client():
    """Fixture to mock the AsyncOpenAI client with dummy responses."""

    async def mock_create_side_effect(*args, **kwargs):
        """Mock the `chat.completions.create` method to return dummy response chunks."""

        class MockChatCompletionStream:
            """Mock stream returned by `chat.completions.create`."""

            async def __aiter__(self):
                # Define a sequence of dummy chunks
                chunks = [
                    mock.Mock(
                        choices=[
                            mock.Mock(
                                delta=mock.Mock(content="Hello, "),
                            )
                        ]
                    ),
                    mock.Mock(
                        choices=[
                            mock.Mock(
                                delta=mock.Mock(content="this is a "),
                            )
                        ]
                    ),
                    mock.Mock(
                        choices=[
                            mock.Mock(
                                delta=mock.Mock(content="mocked response."),
                            )
                        ]
                    ),
                ]
                for chunk in chunks:
                    yield chunk

        MockChatCompletion = mock.Mock(
            choices=[
                mock.Mock(
                    message=mock.Mock(content="Hello, this is a mocked response."),
                )
            ]
        )

        if "messages" not in kwargs:
            raise TypeError("messages is required")
        if "model" not in kwargs:
            raise TypeError("model is required")

        if kwargs.get("stream"):
            return MockChatCompletionStream()
        return MockChatCompletion

    with mock.patch("outspeed.plugins.openai_llm.AsyncOpenAI", new_callable=mock.MagicMock) as MockAsyncOpenAI:
        mock_client_instance = MockAsyncOpenAI.return_value
        mock_client_instance.chat.completions.create = mock_create_side_effect
        yield mock_client_instance
