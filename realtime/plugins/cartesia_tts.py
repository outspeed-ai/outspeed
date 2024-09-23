import asyncio
import base64
import json
import logging
import os
import time
import uuid
from typing import Optional
from urllib.parse import urlencode

import websockets

from realtime.data import AudioData, SessionData
from realtime.plugins.base_plugin import Plugin
from realtime.streams import AudioStream, ByteStream, TextStream, VADStream
from realtime.utils.vad import VADState
from realtime.utils import tracing


class CartesiaTTS(Plugin):
    """
    A plugin for text-to-speech synthesis using the Cartesia API.

    This class handles the connection to the Cartesia TTS service, sends text for synthesis,
    and receives the generated audio data.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091",
        model: str = "sonic-english",
        output_encoding: str = "pcm_s16le",
        output_sample_rate: int = 16000,
        stream: bool = True,
        base_url: str = "wss://api.cartesia.ai/tts/websocket",
        cartesia_version: str = "2024-06-10",
    ):
        """
        Initialize the CartesiaTTS plugin.

        Args:
            api_key (Optional[str]): The API key for Cartesia. If not provided, it will be read from the CARTESIA_API_KEY environment variable.
            voice_id (str): The ID of the voice to use for synthesis.
            model (str): The model to use for synthesis.
            output_encoding (str): The encoding of the output audio.
            output_sample_rate (int): The sample rate of the output audio.
            stream (bool): Whether to stream the audio output.
            base_url (str): The base URL for the Cartesia API.
            cartesia_version (str): The version of the Cartesia API to use.
        """
        super().__init__()

        self._generating: bool = False
        self._task: Optional[asyncio.Task] = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._interrupt_task: Optional[asyncio.Task] = None

        # Set up API key
        self.api_key: str = api_key or os.environ.get("CARTESIA_API_KEY")
        if self.api_key is None:
            raise ValueError("Cartesia API key is required")

        # Set up TTS parameters
        self.voice_id: str = voice_id
        self.model: str = model
        self.output_encoding: str = output_encoding
        self.output_sample_rate: int = output_sample_rate
        self.output_queue: AudioStream = AudioStream()
        self.stream: bool = stream
        self.base_url: str = base_url
        self.cartesia_version: str = cartesia_version
        self._text_context_id: Optional[str] = None
        self._audio_context_id: Optional[str] = None
        self._ws = None

        # Initialize queues
        self.input_queue: Optional[TextStream] = None
        self.interrupt_queue: Optional[asyncio.Queue] = None

    def run(self, input_queue: TextStream) -> ByteStream:
        """
        Start the TTS synthesis process.

        Args:
            input_queue (TextStream): The input queue containing text to synthesize.

        Returns:
            ByteStream: The output queue containing synthesized audio data.
        """
        self.input_queue = input_queue
        self._task = asyncio.create_task(self.synthesize_speech())
        return self.output_queue

    async def connect_websocket(self):
        query_params = {
            "cartesia_version": self.cartesia_version,
            "api_key": self.api_key,
        }
        try:
            self._ws = await websockets.connect(f"{self.base_url}?{urlencode(query_params)}")
        except Exception as e:
            logging.error("Error connecting to Cartesia TTS: %s", e)
            raise asyncio.CancelledError()

    async def synthesize_speech(self):
        """
        Main method for speech synthesis. Connects to the Cartesia API and manages
        the send_text and receive_audio coroutines.
        """

        async def send_text():
            """Send text chunks to the Cartesia API for synthesis."""
            try:
                while True:
                    text_chunk = await self.input_queue.get()
                    if not self._ws:
                        await self.connect_websocket()
                    if isinstance(text_chunk, SessionData):
                        await self.output_queue.put(text_chunk)
                        continue
                    if text_chunk is None or text_chunk == "":
                        if self._text_context_id is None:
                            continue
                        payload = {
                            "transcript": "",
                            "context_id": self._text_context_id,
                            "model_id": self.model,
                            "continue": False,
                            "voice": {"mode": "id", "id": self.voice_id},
                            "output_format": {
                                "encoding": self.output_encoding,
                                "sample_rate": self.output_sample_rate,
                                "container": "raw",
                            },
                        }
                        await self._ws.send(json.dumps(payload))

                        self._text_context_id = None
                        continue
                    if self._text_context_id is None:
                        self._text_context_id = str(uuid.uuid4())
                        self._audio_context_id = self._text_context_id
                    tracing.register_event(tracing.Event.TTS_START)
                    logging.info("Generating TTS: %s", text_chunk)
                    payload = {
                        "voice": {"mode": "id", "id": self.voice_id},
                        "output_format": {
                            "encoding": self.output_encoding,
                            "sample_rate": self.output_sample_rate,
                            "container": "raw",
                        },
                        "transcript": text_chunk,
                        "model_id": self.model,
                        "context_id": self._text_context_id,
                        "continue": True,
                    }
                    self._generating = True
                    await self._ws.send(json.dumps(payload))
            except Exception as e:
                logging.error("Error sending text to Cartesia TTS: %s", e)
                self._generating = False
                self._text_context_id = None
                self._audio_context_id = None
                await self.output_queue.put(None)
                raise asyncio.CancelledError()

        async def receive_audio():
            """Receive synthesized audio from the Cartesia API and put it in the output queue."""
            try:
                total_audio_bytes = 0
                is_first_chunk = True
                while True:
                    if self._ws is None:
                        await asyncio.sleep(0.2)
                        continue
                    response = await self._ws.recv()
                    response = json.loads(response)
                    if response["type"] == "chunk":
                        if response["context_id"] != self._audio_context_id:
                            continue
                        audio_bytes = base64.b64decode(response["data"])
                        total_audio_bytes += len(audio_bytes)
                        if is_first_chunk:
                            tracing.register_event(tracing.Event.TTS_TTFB)
                            logging.info("Got TTS first chunk", time.time())
                            is_first_chunk = False
                        await self.output_queue.put(AudioData(audio_bytes, sample_rate=self.output_sample_rate))
                    elif response["type"] == "done":
                        tracing.register_event(tracing.Event.TTS_END)
                        tracing.register_metric(tracing.Metric.TTS_TOTAL_BYTES, total_audio_bytes)
                        total_audio_bytes = 0
                        self._generating = False
                        is_first_chunk = True
                        tracing.log_timeline()
                        await self.output_queue.put(None)
                    else:
                        logging.error("Unknown response type in Cartesia TTS: %s", response)
            except Exception as e:
                logging.error("Error receiving audio from Cartesia TTS: %s", e)
                self._generating = False
                self._audio_context_id = None
                self._text_context_id = None
                await self.output_queue.put(None)
                raise asyncio.CancelledError()

        try:
            await asyncio.gather(send_text(), receive_audio())
        except asyncio.CancelledError:
            logging.info("TTS cancelled")
            self._generating = False
            self._audio_context_id = None
            self._text_context_id = None

    async def close(self):
        """Close the websocket connection and cancel the main task."""
        if self._ws:
            await self._ws.close()
        if self._task:
            self._task.cancel()

    async def _interrupt(self):
        """
        Handle interruptions (e.g., when the user starts speaking).
        Cancels ongoing TTS generation and clears the output queue.
        """
        while True:
            vad_state: VADState = await self.interrupt_queue.get()
            self._audio_context_id = None
            self._text_context_id = None
            if vad_state == VADState.SPEAKING and (not self.input_queue.empty() or not self.output_queue.empty()):
                if self._task:
                    self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                while not self.output_queue.empty():
                    self.output_queue.get_nowait()
                while not self.input_queue.empty():
                    self.input_queue.get_nowait()
                logging.info("Done cancelling TTS")
                self._generating = False
                self._task = asyncio.create_task(self.synthesize_speech())

    def set_interrupt_stream(self, interrupt_stream: VADStream):
        """
        Set up the interrupt queue and start the interrupt handling task.

        Args:
            interrupt_stream (VADStream): The stream for receiving interrupt signals.
        """
        if isinstance(interrupt_stream, VADStream):
            self.interrupt_queue = interrupt_stream
        else:
            raise ValueError("Interrupt stream must be a VADStream")
        self._interrupt_task = asyncio.create_task(self._interrupt())
