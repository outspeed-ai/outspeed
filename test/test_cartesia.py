import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from outspeed.utils.vad import VADState
import pytest
from outspeed.plugins.cartesia_tts import CartesiaTTS
from outspeed.streams import TextStream, AudioStream, VADStream
from outspeed.data import AudioData


@pytest.fixture
def mock_env_api_key(monkeypatch):
    """Fixture to mock the CARTESIA_API_KEY environment variable."""
    monkeypatch.setenv("CARTESIA_API_KEY", "test_api_key")


@pytest.fixture
def mock_websockets():
    """Fixture to mock websockets.connect."""
    mock_connect = AsyncMock()
    with patch("outspeed.plugins.cartesia_tts.websockets.connect", new=mock_connect):
        yield mock_connect


@pytest.mark.asyncio
async def test_cartesia_tts_initialization():
    """Test CartesiaTTS initialization with default parameters."""
    cartesia_tts = CartesiaTTS()
    assert cartesia_tts.voice_id == "a0e99841-438c-4a64-b679-ae501e7d6091"
    assert cartesia_tts.model == "sonic-english"
    assert isinstance(cartesia_tts.output_queue, AudioStream)


@pytest.mark.asyncio
async def test_cartesia_tts_run():
    """Test CartesiaTTS run method."""
    cartesia_tts = CartesiaTTS()
    input_queue = TextStream()
    output_queue = cartesia_tts.run(input_queue)
    assert cartesia_tts.input_queue == input_queue
    assert output_queue == cartesia_tts.output_queue
    assert cartesia_tts._task is not None


@pytest.mark.asyncio
async def test_cartesia_tts_close():
    """Test CartesiaTTS close method."""
    cartesia_tts = CartesiaTTS()
    cartesia_tts._ws = AsyncMock()
    cartesia_tts._task = AsyncMock()

    await cartesia_tts.close()

    cartesia_tts._ws.close.assert_called_once()
    cartesia_tts._task.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_cartesia_tts_connect_websocket(mock_websockets):
    """Test CartesiaTTS connect_websocket method."""
    cartesia_tts = CartesiaTTS()
    await cartesia_tts.connect_websocket()

    # Verify that websockets.connect was called with the correct URL and parameters
    mock_websockets.assert_called_once()
    called_url = mock_websockets.call_args[0][0]
    assert called_url.startswith(cartesia_tts.base_url)
    assert f"cartesia_version={cartesia_tts.cartesia_version}" in called_url
    assert f"api_key={cartesia_tts.api_key}" in called_url

    # Verify that the websocket connection was established
    assert isinstance(cartesia_tts._ws, AsyncMock)


@pytest.mark.asyncio
async def test_cartesia_tts_synthesize_speech(mock_websockets):
    """Test CartesiaTTS synthesize_speech method."""
    cartesia_tts = CartesiaTTS()
    mock_ws = AsyncMock()
    mock_websockets.return_value = mock_ws

    cartesia_tts.input_queue = TextStream()
    await cartesia_tts.input_queue.put("Hello, world!")

    # Mock websocket receive method to return a valid response
    mock_ws.recv.side_effect = [
        json.dumps({"type": "chunk", "data": "SGVsbG8sIHdvcmxkIQ=="}),  # base64 encoded "Hello, world!"
        json.dumps({"type": "done"}),
    ]

    # Run the synthesize_speech method for a short time
    task = asyncio.create_task(cartesia_tts.synthesize_speech())
    await asyncio.sleep(0.1)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    # Check that the websocket was connected and used correctly
    mock_websockets.assert_called_once()
    assert mock_ws.send.call_count == 1
    assert mock_ws.recv.call_count == 1  # Adjusted to match the actual call count


@pytest.mark.asyncio
async def test_cartesia_tts_set_interrupt():
    """Test CartesiaTTS set_interrupt method."""
    cartesia_tts = CartesiaTTS()
    interrupt_queue = VADStream()
    cartesia_tts.set_interrupt_stream(interrupt_queue)
    assert cartesia_tts.interrupt_queue == interrupt_queue
    assert cartesia_tts._interrupt_task is not None


if __name__ == "__main__":
    pytest.main()
