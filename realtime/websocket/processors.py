import asyncio
import base64
import io
import logging
import time
import wave

import av
import numpy as np
from fastapi import WebSocket

from realtime.streams import AudioStream, ByteStream, TextStream, VideoStream


class WebsocketInputStream:
    """
    Handles incoming WebSocket messages and streams audio and text data.

    Attributes:
        ws (WebSocket): The WebSocket connection.
    """

    def __init__(self, ws: WebSocket, sample_rate: int = 48000):
        self.ws = ws
        self.sample_rate = sample_rate

    async def run(self, audio_stream: AudioStream, message_stream: TextStream, video_stream: VideoStream):
        """
        Starts the task to process incoming WebSocket messages.

        Returns:
            Tuple[AudioStream, TextStream]: A tuple containing the audio and message streams.
        """
        self.audio_output_stream = audio_stream
        self.message_stream = message_stream

        # TODO: Implement video stream processing
        self.video_stream = video_stream

        audio_data = b""
        while True:
            try:
                data = await self.ws.receive_json()
                if data.get("type") == "message":
                    await self.message_stream.put(data.get("data"))
                elif data.get("type") == "audio":
                    audio_bytes = base64.b64decode(data.get("data"))
                    audio_data += audio_bytes

                    if len(audio_data) < 2:
                        continue
                    if len(audio_data) % 2 != 0:
                        # TODO: Get audio dtype from the frontend instead of hardcoding it
                        array = np.frombuffer(audio_data[:-1], dtype=np.int16).reshape(1, -1)
                        audio_data = audio_data[-1:]
                    else:
                        # TODO: Get audio dtype from the frontend instead of hardcoding it
                        array = np.frombuffer(audio_data, dtype=np.int16).reshape(1, -1)
                        audio_data = b""

                    frame = av.AudioFrame.from_ndarray(array, format="s16", layout="mono")
                    frame.sample_rate = self.sample_rate
                    await self.audio_output_stream.put(frame)
            except Exception as e:
                logging.error("websocket: Exception", e)
                raise asyncio.CancelledError()


class WebsocketOutputStream:
    """
    Handles outgoing WebSocket messages by streaming audio and text data.

    Attributes:
        ws (WebSocket): The WebSocket connection.
    """

    def __init__(self, ws: WebSocket):
        self.ws = ws

    async def run(
        self, audio_stream: AudioStream, message_stream: TextStream, video_stream: VideoStream, byte_stream: ByteStream
    ):
        """
        Starts tasks to process and send byte and text streams.

        Args:
            audio_stream (AudioStream): The audio stream to send.
            message_stream (TextStream): The text stream to send.
            video_stream (VideoStream): The video stream to send.
            byte_stream (ByteStream): The byte stream to send.
        """
        # TODO: Implement video stream and audio stream processing
        await asyncio.gather(self.task(byte_stream), self.task(message_stream))

    async def task(self, input_stream):
        """
        Sends data from the input stream over the WebSocket.

        Args:
            input_stream (Stream): The stream from which to send data.
        """
        while True:
            data = await input_stream.get()
            if data is None:
                json_data = {"type": "audio_end", "timestamp": time.time()}
                await self.ws.send_json(json_data)
            elif isinstance(data, bytes):
                output_bytes_io = io.BytesIO()
                in_memory_wav = wave.open(output_bytes_io, "wb")

                # TODO: Get channels, sample width, and sample rate from TTS module instead of hardcoding it
                in_memory_wav.setnchannels(1)
                in_memory_wav.setsampwidth(2)
                in_memory_wav.setframerate(16000)

                in_memory_wav.writeframes(data)
                output_bytes_io.seek(0)
                data = output_bytes_io.read()
                json_data = {"type": "audio", "data": base64.b64encode(data).decode(), "timestamp": time.time()}
                await self.ws.send_json(json_data)
            elif isinstance(data, str):
                json_data = {"type": "message", "data": data, "timestamp": time.time()}
                await self.ws.send_json(json_data)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
