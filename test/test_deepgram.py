import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from realtime.plugins.deepgram_stt import AudioData, DeepgramSTT, TextStream


@pytest.fixture
def mock_env_api_key(monkeypatch):
    """Fixture to mock the DEEPGRAM_API_KEY environment variable."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test_api_key")


@pytest.fixture
def mock_aiohttp_session():
    """Fixture to mock aiohttp.ClientSession."""
    with patch("aiohttp.ClientSession") as mock_session:
        yield mock_session


@pytest.fixture
def mock_ws():
    """Fixture to mock WebSocket connection."""
    mock = AsyncMock()
    mock.receive.return_value = MagicMock(type=aiohttp.WSMsgType.TEXT)
    return mock


@pytest.mark.asyncio
async def test_deepgram_stt_run(mock_env_api_key, mock_aiohttp_session):
    """Test DeepgramSTT run method."""
    stt = DeepgramSTT()
    input_queue = asyncio.Queue()
    output_stream = stt.run(input_queue)
    assert stt.input_queue == input_queue
    assert isinstance(output_stream, TextStream)
    assert stt._task is not None


@pytest.mark.asyncio
async def test_deepgram_stt_close(mock_env_api_key, mock_aiohttp_session):
    """Test DeepgramSTT close method."""
    stt = DeepgramSTT()
    stt.input_queue = asyncio.Queue()
    stt._task = AsyncMock()
    stt._session.close = AsyncMock()  # Make close method awaitable

    await stt.close()

    assert stt.input_queue.get_nowait() == '{"type": "CloseStream"}'
    stt._session.close.assert_called_once()
    stt._task.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_deepgram_stt_connect_ws(mock_env_api_key, mock_aiohttp_session, mock_ws):
    """Test DeepgramSTT _connect_ws method."""
    stt = DeepgramSTT()
    stt._session.ws_connect = AsyncMock(return_value=mock_ws)

    await stt._connect_ws()

    stt._session.ws_connect.assert_called_once()
    assert stt._ws == mock_ws


@pytest.mark.asyncio
async def test_deepgram_stt_send_task(mock_env_api_key, mock_aiohttp_session, mock_ws):
    """Test DeepgramSTT _send_task method."""
    stt = DeepgramSTT()
    stt._ws = mock_ws
    stt.input_queue = asyncio.Queue()

    # Simulate audio data
    audio_data = AudioData(b"dummy_audio_data")
    await stt.input_queue.put(audio_data)

    # Simulate close message
    await stt.input_queue.put('{"type": "CloseStream"}')

    await stt._send_task()

    assert stt._closed == True
    mock_ws.send_bytes.assert_called_once_with(b"dummy_audio_data")
    mock_ws.send_str.assert_called_once_with('{"type": "CloseStream"}')


@pytest.mark.asyncio
async def test_deepgram_stt_recv_task(mock_env_api_key, mock_aiohttp_session, mock_ws):
    """Test DeepgramSTT _recv_task method."""
    stt = DeepgramSTT()
    stt._ws = mock_ws
    stt.output_queue = AsyncMock()

    # Simulate Deepgram response
    responses = [
        MagicMock(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps(
                {
                    "is_final": True,
                    "channel": {"alternatives": [{"transcript": "Hello, world!", "confidence": 0.95}]},
                    "duration": 1.0,
                    "start": 0.0,
                }
            ),
        ),
        MagicMock(type=aiohttp.WSMsgType.CLOSED),
    ]

    mock_ws.receive = AsyncMock(side_effect=responses)
    # Run the receive task
    try:
        await stt._recv_task()
    except asyncio.CancelledError:
        pass
    # Assert that the output queue received the expected transcript
    stt.output_queue.put.assert_called_once_with("Hello, world!")

    # Verify that the method exited after receiving the CLOSED message
    assert mock_ws.receive.call_count == 2
