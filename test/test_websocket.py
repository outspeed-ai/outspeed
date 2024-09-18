import asyncio
import base64
import json
import pytest

import websockets

from outspeed.server import RealtimeServer
from outspeed.streams import AudioStream, TextStream
from outspeed.websocket import websocket

server = RealtimeServer()
server.PORT = 8000


# TODO: Add testing for other data formats (apart from just audio and text)

@websocket("/test")
async def test_handler(audio: AudioStream, text: TextStream):
    return audio, text


@pytest.mark.asyncio
async def test_websocket_decorator_basic_functionality():
    tasks = []
    tasks.append(asyncio.create_task(test_handler()))
    tasks.append(asyncio.create_task(server.start()))
    await asyncio.sleep(1)
    try:
        async with websockets.connect("ws://localhost:8000/test") as websocket:
            await websocket.send(json.dumps({"type": "audio_metadata"}))
            await websocket.send(json.dumps({"type": "message", "data": "Hello from test_handler"}))
            data = await websocket.recv()
            data = json.loads(data)
            data.pop("timestamp")
            assert data == {"type": "message", "data": "Hello from test_handler"}

            audio_data = {
                "type": "audio",
                "data": base64.b64encode(b"base64encodedaudiodata").decode("utf-8"),
                "sample_rate": 48000,
            }
            await websocket.send(json.dumps(audio_data))

            # Expect to receive the same audio data back
            received_data = await websocket.recv()
            received_data = json.loads(received_data)
            assert received_data["type"] == "audio"
            assert received_data["data"] == audio_data["data"]
            assert received_data["sample_rate"] == audio_data["sample_rate"]
        await server.shutdown()
        for task in tasks:
            try:
                task.cancel()
                await task
            except asyncio.CancelledError:
                continue

    except Exception as e:
        await server.shutdown()
        for task in tasks:
            try:
                task.cancel()
                await task
            except asyncio.CancelledError:
                continue

        raise e
