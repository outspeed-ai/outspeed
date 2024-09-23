from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import aiohttp

from realtime.data import AudioData, SessionData
from realtime.plugins.base_plugin import Plugin
from realtime.streams import TextStream
from realtime.utils import tracing

# Constants for WebSocket messages
_KEEPALIVE_MSG: str = json.dumps({"type": "KeepAlive"})
_CLOSE_MSG: str = json.dumps({"type": "CloseStream"})

logger = logging.getLogger(__name__)


class DeepgramSTT(Plugin):
    """
    A plugin for real-time speech-to-text using Deepgram's API.

    This class handles the connection to Deepgram's WebSocket API, sends audio data,
    and processes the returned transcriptions.
    """

    def __init__(
        self,
        *,
        language: str = "en-US",
        detect_language: bool = False,
        interim_results: bool = True,
        punctuate: bool = True,
        smart_format: bool = True,
        model: str = "nova-2",
        api_key: Optional[str] = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        sample_width: int = 2,
        min_silence_duration: int = 100,
        confidence_threshold: float = 0.8,
    ) -> None:
        """
        Initialize the DeepgramSTT plugin.

        :param language: The language code for transcription.
        :param detect_language: Whether to enable language detection.
        :param interim_results: Whether to return interim results.
        :param punctuate: Whether to add punctuation to the transcript.
        :param smart_format: Whether to apply smart formatting to the transcript.
        :param model: The Deepgram model to use for transcription.
        :param api_key: The Deepgram API key. If None, it will be read from the DEEPGRAM_API_KEY environment variable.
        :param sample_rate: The sample rate of the audio in Hz.
        :param num_channels: The number of audio channels.
        :param sample_width: The width of each audio sample in bytes.
        :param min_silence_duration: The minimum duration of silence to trigger end of speech, in milliseconds.
        :param confidence_threshold: The minimum confidence score to accept a transcription.
        """
        api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if api_key is None:
            raise ValueError("Deepgram API key is required")
        self._api_key: str = api_key
        self.language: str = language
        self.detect_language: bool = detect_language
        self.interim_results: bool = interim_results
        self.punctuate: bool = punctuate
        self.smart_format: bool = smart_format
        self.model: str = model
        self.min_silence_duration: int = min_silence_duration
        self.endpointing: int = min_silence_duration

        self._sample_rate: int = sample_rate
        self._num_channels: int = num_channels
        self._sample_width: int = sample_width
        self._speaking: bool = False
        self.confidence_threshold: float = confidence_threshold

        self._session: aiohttp.ClientSession = aiohttp.ClientSession()

        self._closed: bool = False
        self.output_queue: TextStream = TextStream()
        self._audio_duration_received: float = 0.0
        self.input_queue: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None

    async def close(self) -> None:
        """Close the Deepgram connection and clean up resources."""
        if self.input_queue:
            self.input_queue.put_nowait(_CLOSE_MSG)
        await asyncio.sleep(0.2)

        await self._session.close()
        if self._task:
            self._task.cancel()

    def run(self, input_queue: asyncio.Queue) -> TextStream:
        """
        Start the Deepgram STT process.

        :param input_queue: The queue to receive audio data from.
        :return: The output queue for transcribed text.
        """
        self.input_queue = input_queue
        self._task = asyncio.create_task(self._run_ws())
        return self.output_queue

    async def _connect_ws(self) -> None:
        """Connect to the Deepgram WebSocket API."""
        live_config: Dict[str, Any] = {
            "model": self.model,
            "punctuate": self.punctuate,
            "smart_format": self.smart_format,
            "encoding": "linear16",
            "sample_rate": self._sample_rate,
            "channels": self._num_channels,
            "endpointing": self.endpointing,
            "language": self.language,
        }

        headers = {"Authorization": f"Token {self._api_key}"}

        url = f"wss://api.deepgram.com/v1/listen?{urlencode(live_config).lower()}"
        try:
            self._ws = await self._session.ws_connect(url, headers=headers)
        except Exception:
            logger.error("Deepgram connection failed", exc_info=True)
            raise asyncio.CancelledError()

    async def _run_ws(self) -> None:
        """Run the main WebSocket communication loop with Deepgram."""
        try:
            await asyncio.gather(self._keepalive_task(), self._send_task(), self._recv_task())
        except Exception:
            logger.error("Deepgram task failed", exc_info=True)

    async def _keepalive_task(self) -> None:
        """
        Send keepalive messages to maintain the WebSocket connection.

        :param ws: The WebSocket connection to Deepgram.
        """
        try:
            while True:
                if self._ws:
                    await self._ws.send_str(_KEEPALIVE_MSG)
                await asyncio.sleep(5)
        except Exception:
            logger.error("Keepalive task failed", exc_info=True)
            raise asyncio.CancelledError()

    async def _send_task(self) -> None:
        """
        Send audio data to Deepgram through the WebSocket.

        :param ws: The WebSocket connection to Deepgram.
        """
        try:
            while True:
                data: AudioData | SessionData = await self.input_queue.get()
                if not self._ws:
                    await self._connect_ws()

                if data == _CLOSE_MSG:
                    self._closed = True
                    await self._ws.send_str(data)
                    break

                if isinstance(data, SessionData):
                    await self.output_queue.put(data)
                    continue

                bytes_data = data.get_bytes()
                self._audio_duration_received += len(bytes_data) / (
                    self._sample_rate * self._num_channels * self._sample_width
                )
                await self._ws.send_bytes(bytes_data)
        except Exception:
            logger.error("Deepgram send task failed", exc_info=True)
            raise asyncio.CancelledError()

    async def _recv_task(self) -> None:
        """
        Receive and process transcription results from Deepgram.

        :param ws: The WebSocket connection to Deepgram.
        """
        try:
            while True:
                if not self._ws:
                    await asyncio.sleep(0.2)
                    continue

                msg = await self._ws.receive()
                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                    if self._closed:
                        return
                    raise Exception("Deepgram connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.error("Unexpected Deepgram message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                if "is_final" not in data:
                    continue
                is_final = data["is_final"]
                top_choice = data["channel"]["alternatives"][0]
                confidence = top_choice["confidence"]
                audio_processed_duration = data["duration"] + data["start"]

                if top_choice["transcript"] and confidence > self.confidence_threshold and is_final:
                    logger.info("Deepgram transcript: %s", top_choice["transcript"])
                    latency = self._audio_duration_received - audio_processed_duration
                    tracing.register_event(tracing.Event.USER_SPEECH_END, time.time() - latency)
                    tracing.register_event(tracing.Event.TRANSCRIPTION_RECEIVED)
                    await self.output_queue.put(top_choice["transcript"])
        except Exception:
            logger.error("Deepgram receive task failed", exc_info=True)
            raise asyncio.CancelledError()
