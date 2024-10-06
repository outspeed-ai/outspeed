import asyncio
import base64
import json
import logging
import os
import uuid
import wave
from typing import Optional
from urllib.parse import urlencode

import websockets

from outspeed.data import AudioData, SessionData
from outspeed.ops.merge import merge
from outspeed.plugins.base_plugin import Plugin
from outspeed.streams import AudioStream, ByteStream, TextStream
from outspeed.utils import tracing


class OpenAIRealtimeBasic(Plugin):
    """
    A plugin for text-to-speech synthesis using the Cartesia API.

    This class handles the connection to the Cartesia TTS service, sends text for synthesis,
    and receives the generated audio data.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "alloy",
        model: str = "gpt-4o-realtime-preview-2024-10-01",
        output_encoding: str = "pcm16",
        output_sample_rate: int = 24000,
        base_url: str = "wss://api.openai.com/v1/realtime",
        turn_detection: bool = True,
        system_prompt=None,
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
        self.api_key: str = api_key or os.environ.get("OPENAI_API_KEY")
        if self.api_key is None:
            raise ValueError("OpenAI API key is required")

        # Set up TTS parameters
        self.voice_id: str = voice_id
        self.model: str = model
        self.output_encoding: str = output_encoding
        self.output_sample_rate: int = output_sample_rate
        self.output_queue: AudioStream = AudioStream()
        self.base_url: str = base_url
        self._current_context_id: Optional[str] = None
        self._ws = None

        # Initialize queues
        self.input_queue: Optional[TextStream] = None
        self.interrupt_queue: Optional[asyncio.Queue] = None

        self._initialize_handlers()

    def run(self, text_queue: TextStream, audio_queue: AudioStream) -> ByteStream:
        """
        Start the TTS synthesis process.

        Args:
            input_queue (TextStream): The input queue containing text to synthesize.

        Returns:
            ByteStream: The output queue containing synthesized audio data.
        """
        self.text_queue = text_queue
        self.audio_queue = audio_queue
        self.input_queue = merge([text_queue, audio_queue])
        self._task = asyncio.create_task(self.start_session())
        return self.output_queue

    async def connect_websocket(self):
        headers = {"Authorization": f"Bearer {self.api_key}", "OpenAI-Beta": "realtime=v1"}
        query_params = {
            "model": self.model,
        }
        try:
            self._ws = await websockets.connect(f"{self.base_url}?{urlencode(query_params)}", extra_headers=headers)
            logging.info("Connected to OpenAI Realtime")
        except Exception as e:
            logging.error("Error connecting to OpenAI Realtime: %s", e)
            raise asyncio.CancelledError()

    async def start_session(self):
        """
        Main method for speech synthesis. Connects to the Cartesia API and manages
        the send_text and receive_audio coroutines.
        """

        async def send_task():
            """Send text chunks to the Cartesia API for synthesis."""
            first_chunk = True
            try:
                while True:
                    text_chunk = await self.input_queue.get()
                    if first_chunk:
                        await self.connect_websocket()
                        first_chunk = False

                    if isinstance(text_chunk, SessionData):
                        await self.output_queue.put(text_chunk)
                        continue

                    if isinstance(text_chunk, AudioData):
                        await self._ws.send(
                            json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": text_chunk.resample(self.output_sample_rate).get_base64(),
                                }
                            )
                        )

                    self._generating = True
            except Exception as e:
                logging.error("Error sending text to Cartesia TTS: %s", e)
                self._generating = False
                await self.output_queue.put(None)
                raise asyncio.CancelledError()

        async def receive_task():
            """Receive synthesized audio from the Cartesia API and put it in the output queue."""
            try:
                total_audio_bytes = 0
                is_first_chunk = True
                while True:
                    if self._ws is None:
                        await asyncio.sleep(0.2)
                        continue
                    response = await self._ws.recv()
                    msg = json.loads(response)
                    msg_type = msg.get("type")

                    handler = self._handlers.get(msg_type, self._handle_unknown)
                    await handler(msg)
            except Exception as e:
                logging.error("Error receiving audio from Cartesia TTS: %s", e)
                self._generating = False
                await self.output_queue.put(None)
                raise asyncio.CancelledError()

        try:
            await asyncio.gather(send_task(), receive_task())
        except asyncio.CancelledError:
            logging.info("TTS cancelled")
            self._generating = False

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
            user_speaking = await self.interrupt_queue.get()
            if self._generating and user_speaking:
                if self._task:
                    self._task.cancel()
                while not self.output_queue.empty():
                    self.output_queue.get_nowait()
                logging.info("Done cancelling TTS")
                self._generating = False
                self._task = asyncio.create_task(self.start_session())

    async def set_interrupt(self, interrupt_queue: asyncio.Queue):
        """
        Set up the interrupt queue and start the interrupt handling task.

        Args:
            interrupt_queue (asyncio.Queue): The queue for receiving interrupt signals.
        """
        self.interrupt_queue = interrupt_queue
        self._interrupt_task = asyncio.create_task(self._interrupt())

    def _initialize_handlers(self):
        """Initialize the dispatch table for handling different message types."""
        self._handlers = {
            "session.created": self._handle_session_created,
            "session.updated": self._handle_session_updated,
            "input_audio_buffer.speech_started": self._handle_speech_started,
            "input_audio_buffer.speech_stopped": self._handle_speech_stopped,
            "conversation.item.created": self._handle_conversation_item_created,
            "conversation.item.input_audio_transcription.completed": self._handle_transcription_completed,
            "response.created": self._handle_response_created,
            "response.output_item.added": self._handle_output_item_added,
            "response.content_part.added": self._handle_content_part_added,
            "response.audio_transcript.delta": self._handle_audio_transcript_delta,
            "response.audio.delta": self._handle_audio_delta,
            "response.audio.done": self._handle_audio_done,
            "response.audio_transcript.done": self._handle_audio_transcript_done,
            "response.content_part.done": self._handle_content_part_done,
            "response.output_item.done": self._handle_output_item_done,
            "response.done": self._handle_response_done,
            "rate_limits.updated": self._handle_rate_limits_updated,
            "error": self._handle_error,
        }

    async def _handle_session_created(self, msg):
        pass  # Implement your logic here

    async def _handle_session_updated(self, msg):
        pass  # Implement your logic here

    async def _handle_speech_started(self, msg):
        pass  # Implement your logic here

    async def _handle_speech_stopped(self, msg):
        pass  # Implement your logic here

    async def _handle_conversation_item_created(self, msg):
        pass  # Implement your logic here

    async def _handle_response_created(self, msg):
        logging.debug(f"Received response created: {msg}")

    async def _handle_transcription_completed(self, msg):
        pass  # Implement your logic here

    async def _handle_output_item_added(self, msg):
        pass  # Implement your logic here

    async def _handle_content_part_added(self, msg):
        pass  # Implement your logic here

    async def _handle_audio_transcript_delta(self, msg):
        pass  # Implement your logic here

    async def _handle_audio_delta(self, msg):
        audio = AudioData(
            data=base64.b64decode(msg["delta"]),
            sample_rate=self.output_sample_rate,
            channels=1,
        )
        await self.output_queue.put(audio)

    async def _handle_audio_done(self, msg):
        pass  # Implement your logic here

    async def _handle_audio_transcript_done(self, msg):
        pass  # Implement your logic here

    async def _handle_content_part_done(self, msg):
        pass  # Implement your logic here

    async def _handle_output_item_done(self, msg):
        pass  # Implement your logic here

    async def _handle_response_done(self, msg):
        pass  # Implement your logic here

    async def _handle_rate_limits_updated(self, msg):
        pass  # Implement your logic here

    async def _handle_error(self, msg):
        raise Exception(f"Error: {msg}")

    async def _handle_unknown(self, msg):
        logging.error("Unknown response type in Cartesia TTS: %s", msg)
