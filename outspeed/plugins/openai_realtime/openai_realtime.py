import asyncio
import base64
import json
import logging
import os
import traceback
from typing import Optional
from urllib.parse import urlencode

import websockets

from outspeed.data import AudioData, SessionData
from outspeed.ops.merge import merge
from outspeed.plugins.base_plugin import Plugin
from outspeed.plugins.openai_realtime.events import ServerEvent
from outspeed.plugins.openai_realtime.session import RealtimeSession
from outspeed.plugins.openai_realtime.types import (
    ConversationCreated,
    ConversationItemCreated,
    ConversationItemInputAudioTranscriptionCompleted,
    InputAudioBufferSpeechStarted,
    ResponseAudioTranscriptDelta,
    ResponseContentPartAdded,
    ResponseDone,
    ResponseTextDeltaAdded,
    SessionCreated,
    SessionUpdated,
)
from outspeed.streams import AudioStream, ByteStream, TextStream


class OpenAIRealtime(Plugin):
    """
    A plugin for text-to-speech synthesis using the OpenAI Realtime API.

    This class handles the connection to the OpenAI Realtime API, sends text for synthesis,
    and receives the generated audio data.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "alloy",
        model: str = "gpt-4o-realtime-preview-2024-10-01",
        output_encoding: str = "pcm16",
        input_encoding: str = "pcm16",
        output_sample_rate: int = 24000,
        base_url: str = "wss://api.openai.com/v1/realtime",
        turn_detection: bool = True,
        system_prompt: Optional[str] = None,
        temperature: float = 0.8,
        max_output_tokens: Optional[int] = None,
    ):
        """
        Initialize the OpenAIRealtimeBasic plugin.

        Args:
            api_key (Optional[str]): The API key for OpenAI. If not provided, it will be read from the OPENAI_API_KEY environment variable.
            voice_id (str): The ID of the voice to use for synthesis.
            model (str): The model to use for synthesis.
            output_encoding (str): The encoding of the output audio.
            output_sample_rate (int): The sample rate of the output audio.
            stream (bool): Whether to stream the audio output.
            base_url (str): The base URL for the OpenAI Realtime API.
            turn_detection (bool): Whether to use turn detection.
            system_prompt (str): The system prompt to use.
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
        self.system_prompt = system_prompt
        self.turn_detection = turn_detection
        self.input_encoding = input_encoding
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

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
            session_update_msg = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "voice": self.voice_id,
                    "input_audio_format": self.input_encoding,
                    "output_audio_format": self.output_encoding,
                    "input_audio_transcription": {"model": "whisper-1"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 200,
                    },
                    "temperature": self.temperature,
                    "max_response_output_tokens": "inf" if self.max_output_tokens is None else self.max_output_tokens,
                },
            }
            if self.system_prompt:
                session_update_msg["session"]["instructions"] = self.system_prompt

            await self._ws.send(json.dumps(session_update_msg))
        except Exception as e:
            logging.error("Error connecting to OpenAI Realtime: %s", e)
            logging.error(f"Traceback:\n{traceback.format_exc()}")
            raise asyncio.CancelledError()

    async def start_session(self):
        """
        Main method for speech synthesis. Connects to the OpenAI Realtime API and manages
        the send_text and receive_audio coroutines.
        """

        async def send_task():
            """Send text chunks to the OpenAI Realtime API for synthesis."""
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
                logging.error("Error sending text to OpenAI Realtime: %s", e)
                logging.error(f"Traceback:\n{traceback.format_exc()}")
                self._generating = False
                await self.output_queue.put(None)
                raise asyncio.CancelledError()

        async def receive_task():
            """Receive synthesized audio from the OpenAI Realtime API and put it in the output queue."""
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
                logging.error("Error receiving audio from OpenAI Realtime: %s", e)
                logging.error(f"Traceback:\n{traceback.format_exc()}")
                self._generating = False
                await self.output_queue.put(None)
                raise asyncio.CancelledError()

        try:
            await asyncio.gather(send_task(), receive_task())
        except asyncio.CancelledError:
            logging.info("OpenAI Realtime cancelled")
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
        while not self.output_queue.empty():
            self.output_queue.get_nowait()
        logging.info("Done cancelling TTS")

    def _initialize_handlers(self):
        """Initialize the dispatch table for handling different message types."""
        self._handlers = {
            ServerEvent.SESSION_CREATED: self._handle_session_created,
            ServerEvent.SESSION_UPDATED: self._handle_session_updated,
            ServerEvent.CONVERSATION_CREATED: self._handle_conversation_created,
            ServerEvent.INPUT_AUDIO_BUFFER_SPEECH_STARTED: self._handle_speech_started,
            ServerEvent.INPUT_AUDIO_BUFFER_SPEECH_STOPPED: self._handle_speech_stopped,
            ServerEvent.INPUT_AUDIO_BUFFER_COMMITTED: self._handle_speech_committed,
            ServerEvent.INPUT_AUDIO_BUFFER_CLEARED: self._handle_speech_cleared,
            ServerEvent.CONVERSATION_ITEM_CREATED: self._handle_conversation_item_created,
            ServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED: self._handle_transcription_completed,
            ServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED: self._handle_transcription_failed,
            ServerEvent.RESPONSE_CREATED: self._handle_response_created,
            ServerEvent.RESPONSE_OUTPUT_ITEM_ADDED: self._handle_output_item_added,
            ServerEvent.RESPONSE_CONTENT_PART_ADDED: self._handle_content_part_added,
            ServerEvent.RESPONSE_AUDIO_TRANSCRIPT_DELTA: self._handle_audio_transcript_delta,
            ServerEvent.RESPONSE_TEXT_DELTA: self._handle_text_delta,
            ServerEvent.RESPONSE_AUDIO_DELTA: self._handle_audio_delta,
            ServerEvent.RESPONSE_AUDIO_DONE: self._handle_audio_done,
            ServerEvent.RESPONSE_AUDIO_TRANSCRIPT_DONE: self._handle_audio_transcript_done,
            ServerEvent.RESPONSE_CONTENT_PART_DONE: self._handle_content_part_done,
            ServerEvent.RESPONSE_OUTPUT_ITEM_DONE: self._handle_output_item_done,
            ServerEvent.RESPONSE_DONE: self._handle_response_done,
            ServerEvent.RATE_LIMITS_UPDATED: self._handle_rate_limits_updated,
            ServerEvent.ERROR: self._handle_error,
        }

    async def _handle_session_created(self, msg: SessionCreated):
        print(msg)
        self._session = RealtimeSession.from_dict(msg)

    async def _handle_session_updated(self, msg: SessionUpdated):
        pass  # Implement your logic here

    async def _handle_conversation_created(self, msg: ConversationCreated):
        self._session.add_conversation(msg)

    async def _handle_speech_started(self, msg: InputAudioBufferSpeechStarted):
        await self._interrupt()

    async def _handle_speech_stopped(self, msg):
        pass  # Implement your logic here

    async def _handle_speech_committed(self, msg):
        pass  # Implement your logic here

    async def _handle_speech_cleared(self, msg):
        pass  # Implement your logic here

    async def _handle_conversation_item_created(self, msg: ConversationItemCreated):
        self._session.add_item(msg)

    async def _handle_response_created(self, msg):
        logging.debug(f"Received response created: {msg}")

    async def _handle_transcription_completed(self, msg: ConversationItemInputAudioTranscriptionCompleted):
        print(msg)
        self._session.add_input_audio_transcription(msg)
        print(self._session.get_items())

    async def _handle_transcription_failed(self, msg):
        logging.error(f"Transcription failed: {msg}")

    async def _handle_output_item_added(self, msg):
        pass  # Implement your logic here

    async def _handle_content_part_added(self, msg: ResponseContentPartAdded):
        pass  # Implement your logic here

    async def _handle_audio_transcript_delta(self, msg: ResponseAudioTranscriptDelta):
        return
        self._session.update_transcript_delta(msg)

    async def _handle_text_delta(self, msg: ResponseTextDeltaAdded):
        return
        self._session.update_text_delta(msg)

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
        pass

    async def _handle_content_part_done(self, msg):
        pass  # Implement your logic here

    async def _handle_output_item_done(self, msg):
        pass  # Implement your logic here

    async def _handle_response_done(self, msg: ResponseDone):
        self._session.add_response(msg)
        print(self._session.get_items())

    async def _handle_rate_limits_updated(self, msg):
        pass  # Implement your logic here

    async def _handle_error(self, msg):
        raise Exception(f"Error: {msg}")

    async def _handle_unknown(self, msg):
        logging.error("Unknown response type in OpenAI Realtime: %s", msg)
