import asyncio
import base64
import logging
import time

import numpy as np
import scipy.signal as signal
from fastapi import WebSocket

from realtime.data import AudioData
from realtime.streams import AudioStream, ByteStream, TextStream, VideoStream


def resample_wav_bytes(audio_data: AudioData, target_sample_rate: int) -> bytes:
    """
    Resample WAV bytes to a target sample rate.

    Args:
        wav_bytes (bytes): The input WAV file as bytes.
        target_sample_rate (int): The desired sample rate in Hz.

    Returns:
        bytes: The resampled WAV file as bytes.
    """
    wav_bytes = audio_data.get_bytes()
    if audio_data.sample_rate == target_sample_rate:
        return wav_bytes
    # Load WAV bytes into AudioSegment
    audio_array = np.frombuffer(wav_bytes, dtype=np.int16)

    # Calculate the resampling ratio
    ratio = target_sample_rate / audio_data.sample_rate

    # Resample the audio using scipy.signal.resample
    resampled_audio = signal.resample(audio_array, int(len(audio_array) * ratio))

    resampled_audio = resampled_audio.astype(np.int16).tobytes()

    return resampled_audio


class WebsocketInputProcessor:
    """
    Handles incoming WebSocket messages and streams audio and text data.

    Attributes:
        sample_rate (int): The sample rate for audio processing.
        ws (WebSocket): The WebSocket connection.
        audio_output_stream (AudioStream): The stream for input audio data.
        message_stream (TextStream): The stream for input text messages.
        video_stream (VideoStream): The stream for input video data.
    """

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Sample rate must be a positive integer")
        self._sample_rate = value

    def __init__(
        self, audio_stream: AudioStream, message_stream: TextStream, video_stream: VideoStream, sample_rate: int = 48000
    ):
        self.audio_output_stream = audio_stream
        self.message_stream = message_stream
        self.video_stream = video_stream
        self.sample_rate = sample_rate
        self._inputTrack = None

    def setInputTrack(self, track: TextStream):
        self._inputTrack = track

    async def run(self):
        """
        Starts the task to process incoming WebSocket messages.

        Returns:
            Tuple[AudioStream, TextStream]: A tuple containing the audio and message streams.
        """
        # TODO: Implement video stream processing

        while not self._inputTrack:
            await asyncio.sleep(0.2)
        audio_data = b""
        while True:
            try:
                data = await self._inputTrack.get()
                if data.get("type") == "message":
                    await self.message_stream.put(data.get("data"))
                elif data.get("type") == "audio":
                    audio_bytes = base64.b64decode(data.get("data"))
                    audio_data = AudioData(audio_bytes, sample_rate=self.sample_rate)
                    await self.audio_output_stream.put(audio_data)
            except Exception as e:
                logging.error("websocket: Exception", e)
                raise asyncio.CancelledError()


class WebsocketOutputProcessor:
    """
    Handles outgoing WebSocket messages by streaming audio and text data.

    Attributes:
        sample_rate (int): The sample rate for audio processing.
        audio_stream (AudioStream): The audio stream to send.
        message_stream (TextStream): The text stream to send.
        video_stream (VideoStream): The video stream to send.
        byte_stream (ByteStream): The byte stream to send.
    """

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Sample rate must be a positive integer")
        self._sample_rate = value

    def __init__(
        self, audio_stream: AudioStream, message_stream: TextStream, video_stream: VideoStream, byte_stream: ByteStream
    ):
        self.audio_stream = audio_stream
        self.message_stream = message_stream
        self.video_stream = video_stream
        self.byte_stream = byte_stream
        self._outputTrack = None

    def setOutputTrack(self, track: TextStream):
        self._outputTrack = track

    async def run(self):
        """
        Starts tasks to process and send byte and text streams.
        """
        # TODO: Implement video stream and audio stream processing
        await asyncio.gather(self.task(self.byte_stream), self.task(self.message_stream), self.task(self.audio_stream))

    async def task(self, input_stream):
        """
        Sends data from the input stream over the WebSocket.

        Args:
            input_stream (Stream): The stream from which to send data.
        """
        while not self._outputTrack:
            await asyncio.sleep(0.2)
        while True:
            if not input_stream:
                break
            audio_data = await input_stream.get()
            if audio_data is None:
                print("Sending audio end")
                json_data = {"type": "audio_end", "timestamp": time.time()}
                await self._outputTrack.put(json_data)
            elif isinstance(audio_data, AudioData):
                data = resample_wav_bytes(audio_data, self.sample_rate)
                json_data = {
                    "type": "audio",
                    "data": base64.b64encode(data).decode(),
                    "timestamp": time.time(),
                    "sample_rate": audio_data.sample_rate,
                }
                await self._outputTrack.put(json_data)
            elif isinstance(audio_data, str):
                json_data = {"type": "message", "data": audio_data, "timestamp": time.time()}
                await self._outputTrack.put(json_data)
            else:
                raise ValueError(f"Unsupported data type: {type(audio_data)}")
