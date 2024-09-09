import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from realtime.plugins.fireworks_llm import FireworksLLM
from realtime.streams import TextStream


@pytest.fixture
def mock_env_api_key(monkeypatch):
    """Fixture to mock the FIREWORKS_API_KEY environment variable."""
    monkeypatch.setenv("FIREWORKS_API_KEY", "test_api_key")


@pytest.fixture
def mock_async_openai():
    """Fixture to mock AsyncOpenAI client."""
    with patch("realtime.plugins.fireworks_llm.AsyncOpenAI") as mock_client:
        yield mock_client


@pytest.fixture
def fireworks_llm(mock_env_api_key, mock_async_openai):
    """Fixture to create a FireworksLLM instance."""
    return FireworksLLM()


@pytest.mark.asyncio
async def test_fireworks_llm_initialization(fireworks_llm):
    """Test FireworksLLM initialization with default parameters."""
    assert fireworks_llm._model == "accounts/fireworks/models/llama-v3p1-8b-instruct"
    assert fireworks_llm._api_key == "test_api_key"
    assert isinstance(fireworks_llm.output_queue, TextStream)
    assert isinstance(fireworks_llm.chat_history_queue, TextStream)


@pytest.mark.asyncio
async def test_fireworks_llm_run(fireworks_llm):
    """Test FireworksLLM run method."""
    input_queue = TextStream()
    output_queue, chat_history_queue = fireworks_llm.run(input_queue)
    assert fireworks_llm.input_queue == input_queue
    assert output_queue == fireworks_llm.output_queue
    assert chat_history_queue == fireworks_llm.chat_history_queue
    assert fireworks_llm._task is not None


@pytest.mark.asyncio
async def test_fireworks_llm_close(fireworks_llm):
    """Test FireworksLLM close method."""
    fireworks_llm._task = AsyncMock()
    await fireworks_llm.close()
    fireworks_llm._task.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_fireworks_llm_stream_chat_completions(fireworks_llm, mock_async_openai):
    """Test FireworksLLM _stream_chat_completions method."""
    fireworks_llm.input_queue = asyncio.Queue()
    await fireworks_llm.input_queue.put("Hello, AI!")

    mock_chunk = ChatCompletionChunk(
        id="1",
        choices=[Choice(delta=ChoiceDelta(content="Hello, human!"), index=0, finish_reason=None)],
        model="test_model",
        object="chat.completion.chunk",
        created=1234567890,
    )

    # Create an AsyncMock for the chat.completions.create method
    mock_create = AsyncMock()
    mock_create.return_value = AsyncMock()
    mock_create.return_value.__aiter__.return_value = [mock_chunk]

    # Assign the AsyncMock to the client's chat.completions.create method
    fireworks_llm._client.chat.completions.create = mock_create

    # Run the _stream_chat_completions method for a short time
    task = asyncio.create_task(fireworks_llm._stream_chat_completions())
    await asyncio.sleep(0.1)

    # Cancel the task and wait for it to finish
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Assert that the method was called with the correct arguments
    mock_create.assert_called_once_with(
        model=fireworks_llm._model,
        stream=fireworks_llm._stream,
        messages=fireworks_llm._history,
        temperature=fireworks_llm._temperature,
    )

    assert fireworks_llm._history[-2] == {"role": "user", "content": "Hello, AI!"}
    assert fireworks_llm._history[-1] == {"role": "assistant", "content": "Hello, human!"}
    assert await fireworks_llm.output_queue.get() == "Hello, human!"


@pytest.mark.asyncio
async def test_fireworks_llm_interrupt(fireworks_llm):
    """Test FireworksLLM _interrupt method."""
    fireworks_llm.interrupt_queue = asyncio.Queue()
    fireworks_llm._generating = True

    # Create a mock task with a cancel method
    mock_task = MagicMock()
    mock_task.cancel = MagicMock()
    fireworks_llm._task = mock_task

    fireworks_llm.output_queue = AsyncMock()

    # Start the _interrupt method
    interrupt_task = asyncio.create_task(fireworks_llm._interrupt())

    # Simulate user speaking
    await fireworks_llm.interrupt_queue.put(True)
    await asyncio.sleep(0.1)

    assert not fireworks_llm._generating
    mock_task.cancel.assert_called_once()

    # Clean up
    interrupt_task.cancel()
    try:
        await interrupt_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_fireworks_llm_set_interrupt(fireworks_llm):
    """Test FireworksLLM set_interrupt method."""
    interrupt_queue = asyncio.Queue()
    await fireworks_llm.set_interrupt(interrupt_queue)
    assert fireworks_llm.interrupt_queue == interrupt_queue
    assert fireworks_llm._interrupt_task is not None


if __name__ == "__main__":
    pytest.main()
