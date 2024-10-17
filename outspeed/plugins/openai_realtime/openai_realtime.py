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
from outspeed.plugins.openai_realtime.events import ClientEvent, ServerEvent
from outspeed.plugins.openai_realtime.session import RealtimeSession
from outspeed.plugins.openai_realtime.types import (
    ConversationCreated,
    ConversationItemCreated,
    ConversationItemInputAudioTranscriptionCompleted,
    InputAudioBufferSpeechStarted,
    ResponseAudioTranscriptDelta,
    ResponseContentPartAdded,
    ResponseDone,
    ResponseFunctionCallArgumentsDone,
    ResponseTextDeltaAdded,
    SessionCreated,
    SessionUpdated,
)
from outspeed.streams import AudioStream, ByteStream, TextStream
from outspeed.tool import Tool


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
        silence_duration_ms: int = 200,
        vad_threshold: float = 0.5,
        initiate_conversation_with_greeting: Optional[str] = None,
        tools: Optional[list[Tool]] = None,
        tool_choice: str = "auto",
        respond_to_tool_calls: bool = True,
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
        self.api_key: str = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        # Set up TTS parameters
        self.voice_id: str = voice_id
        self.model: str = model
        self.output_encoding: str = output_encoding
        self.output_sample_rate: int = output_sample_rate
        self.audio_output_queue: AudioStream = AudioStream()
        self.text_output_queue: TextStream = TextStream()
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
        self.silence_duration_ms = silence_duration_ms
        self.vad_threshold = vad_threshold
        self.initiate_conversation_with_greeting = initiate_conversation_with_greeting
        self.tools = tools
        self.tool_choice = tool_choice
        self.respond_to_tool_calls = respond_to_tool_calls
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
        return self.audio_output_queue, self.text_output_queue

    async def connect_websocket(self):
        headers = {"Authorization": f"Bearer {self.api_key}", "OpenAI-Beta": "realtime=v1"}
        query_params = {
            "model": self.model,
        }
        try:
            self._ws = await websockets.connect(f"{self.base_url}?{urlencode(query_params)}", extra_headers=headers)
            logging.info("Connected to OpenAI Realtime")
            session_update_msg = {
                "type": ClientEvent.SESSION_UPDATE,
                "session": {
                    "modalities": ["text", "audio"],
                    "voice": self.voice_id,
                    "input_audio_format": self.input_encoding,
                    "output_audio_format": self.output_encoding,
                    "input_audio_transcription": {"model": "whisper-1"},
                    "temperature": self.temperature,
                    "max_response_output_tokens": "inf" if self.max_output_tokens is None else self.max_output_tokens,
                    "tool_choice": self.tool_choice,
                },
            }
            if self.turn_detection:
                session_update_msg["session"]["turn_detection"] = {
                    "type": "server_vad",
                    "threshold": self.vad_threshold,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": self.silence_duration_ms,
                }
            else:
                session_update_msg["session"]["turn_detection"] = None

            if self.system_prompt:
                session_update_msg["session"]["instructions"] = self.system_prompt

            if self.tools:
                session_update_msg["session"]["tools"] = []
                for tool in self.tools:
                    tool_json = tool.to_openai_tool_json()["function"]
                    tool_json["type"] = "function"
                    tool_json.pop("strict")
                    session_update_msg["session"]["tools"].append(tool_json)

            await self._ws.send(json.dumps(session_update_msg))

            if self.initiate_conversation_with_greeting:
                await self._ws.send(
                    json.dumps(
                        {
                            "type": ClientEvent.RESPONSE_CREATE,
                            "response": {
                                "modalities": ["text", "audio"],
                                "instructions": "Say the following greeting to the user: "
                                + self.initiate_conversation_with_greeting,
                            },
                        }
                    )
                )
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
                        await self.audio_output_queue.put(text_chunk)
                        continue

                    if isinstance(text_chunk, AudioData):
                        await self._ws.send(
                            json.dumps(
                                {
                                    "type": ClientEvent.INPUT_AUDIO_BUFFER_APPEND,
                                    "audio": text_chunk.resample(self.output_sample_rate).get_base64(),
                                }
                            )
                        )

                    if isinstance(text_chunk, str):
                        await self._ws.send(
                            json.dumps(
                                {
                                    "type": ClientEvent.CONVERSATION_ITEM_CREATE,
                                    "item": {
                                        "type": "message",
                                        "role": "user",
                                        "content": [{"type": "input_text", "text": text_chunk}],
                                    },
                                }
                            )
                        )
                        await self._ws.send(json.dumps({"type": ClientEvent.RESPONSE_CREATE}))

                    self._generating = True
            except Exception as e:
                logging.error("Error sending text to OpenAI Realtime: %s", e)
                logging.error(f"Traceback:\n{traceback.format_exc()}")
                self._generating = False
                await self.audio_output_queue.put(None)
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

                    if msg_type in self._events_to_ignore:
                        continue

                    handler = self._handlers.get(msg_type, self._handle_unknown)
                    await handler(msg)
            except Exception as e:
                logging.error("Error receiving audio from OpenAI Realtime: %s", e)
                logging.error(f"Traceback:\n{traceback.format_exc()}")
                self._generating = False
                await self.audio_output_queue.put(None)
                raise asyncio.CancelledError()

        try:
            await asyncio.gather(send_task(), receive_task())
        except asyncio.CancelledError:
            logging.info("OpenAI Realtime cancelled")
            self._generating = False

    async def close(self):
        """Close the websocket connection and cancel the main task."""
        if self._ws:
            asyncio.create_task(self._ws.close())
        if self._task:
            self._task.cancel()

    async def _interrupt(self):
        """
        Handle interruptions (e.g., when the user starts speaking).
        Cancels ongoing TTS generation and clears the output queue.
        """
        while not self.audio_output_queue.empty():
            self.audio_output_queue.get_nowait()
        logging.info("Done cancelling TTS generation \n")

    def _initialize_handlers(self):
        """Initialize the dispatch table for handling different message types."""
        self._handlers = {
            ServerEvent.SESSION_CREATED: self._handle_session_created,
            ServerEvent.SESSION_UPDATED: self._handle_session_updated,
            ServerEvent.CONVERSATION_CREATED: self._handle_conversation_created,
            ServerEvent.INPUT_AUDIO_BUFFER_SPEECH_STARTED: self._handle_speech_started,
            ServerEvent.CONVERSATION_ITEM_CREATED: self._handle_conversation_item_created,
            ServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED: self._handle_transcription_completed,
            ServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED: self._handle_transcription_failed,
            ServerEvent.RESPONSE_CREATED: self._handle_response_created,
            ServerEvent.RESPONSE_OUTPUT_ITEM_ADDED: self._handle_output_item_added,
            ServerEvent.RESPONSE_CONTENT_PART_ADDED: self._handle_content_part_added,
            ServerEvent.RESPONSE_AUDIO_TRANSCRIPT_DELTA: self._handle_audio_transcript_delta,
            ServerEvent.RESPONSE_TEXT_DELTA: self._handle_text_delta,
            ServerEvent.RESPONSE_AUDIO_DELTA: self._handle_audio_delta,
            ServerEvent.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE: self._handle_function_call_arguments_done,
            ServerEvent.RESPONSE_DONE: self._handle_response_done,
            ServerEvent.ERROR: self._handle_error,
            ServerEvent.RATE_LIMITS_UPDATED: self._handle_rate_limits_updated,
        }

        self._events_to_ignore = [
            ServerEvent.RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA,
            ServerEvent.INPUT_AUDIO_BUFFER_SPEECH_STOPPED,
            ServerEvent.INPUT_AUDIO_BUFFER_COMMITTED,
            ServerEvent.INPUT_AUDIO_BUFFER_CLEARED,
            ServerEvent.RESPONSE_TEXT_DONE,
            ServerEvent.RESPONSE_AUDIO_DONE,
            ServerEvent.RESPONSE_AUDIO_TRANSCRIPT_DONE,
            ServerEvent.RESPONSE_CONTENT_PART_DONE,
            ServerEvent.RESPONSE_OUTPUT_ITEM_DONE,
        ]

    async def _handle_session_created(self, msg: SessionCreated):
        self._session = RealtimeSession.from_dict(msg)

    async def _handle_session_updated(self, msg: SessionUpdated):
        pass  # Implement your logic here

    async def _handle_conversation_created(self, msg: ConversationCreated):
        self._session.add_conversation(msg)

    async def _handle_speech_started(self, msg: InputAudioBufferSpeechStarted):
        await self._interrupt()

    async def _handle_conversation_item_created(self, msg: ConversationItemCreated):
        self._session.add_item(msg)

    async def _handle_response_created(self, msg):
        logging.debug(f"Received response created: {msg} \n")

    async def _handle_transcription_completed(self, msg: ConversationItemInputAudioTranscriptionCompleted):
        chat_msg = self._session.add_input_audio_transcription(msg)
        logging.info(f"Transcription completed: {chat_msg} \n")
        await self.text_output_queue.put(json.dumps(chat_msg))

    async def _handle_transcription_failed(self, msg):
        logging.error(f"Transcription failed: {msg} \n")

    async def _handle_output_item_added(self, msg):
        pass  # Implement your logic here

    async def _handle_content_part_added(self, msg: ResponseContentPartAdded):
        pass  # Implement your logic here

    async def _handle_audio_transcript_delta(self, msg: ResponseAudioTranscriptDelta):
        pass

    async def _handle_text_delta(self, msg: ResponseTextDeltaAdded):
        pass

    async def _handle_audio_delta(self, msg):
        audio = AudioData(
            data=base64.b64decode(msg["delta"]),
            sample_rate=self.output_sample_rate,
            channels=1,
        )
        await self.audio_output_queue.put(audio)

    async def _handle_function_call_arguments_done(self, msg: ResponseFunctionCallArgumentsDone):
        if not self.tools:
            return

        for tool in self.tools:
            if tool.name == msg["name"]:
                try:
                    logging.info(f"Calling tool {tool.name} with arguments: {msg['arguments']} \n")
                    result = await tool._run_tool(
                        {
                            "id": msg["item_id"],
                            "function": {"arguments": json.loads(msg["arguments"]), "name": msg["name"]},
                        }
                    )
                    logging.info(f"Tool {tool.name} returned: {result} \n")
                    break
                except Exception as e:
                    logging.error(f"Error calling tool {tool.name}: {e} \n")
                    return

        await self._ws.send(
            json.dumps(
                {
                    "type": ClientEvent.CONVERSATION_ITEM_CREATE,
                    "item": {
                        "type": "function_call_output",
                        "call_id": msg["call_id"],
                        "output": result["content"],
                    },
                }
            )
        )
        if self.respond_to_tool_calls:
            await self._ws.send(json.dumps({"type": ClientEvent.RESPONSE_CREATE, "response": {"tool_choice": "none"}}))

    async def _handle_response_done(self, msg: ResponseDone):
        logging.info(f"Received response done: {msg} \n")
        chat_msgs = self._session.add_response(msg)
        logging.info(f"Response done: {chat_msgs} \n")
        for chat_msg in chat_msgs:
            await self.text_output_queue.put(json.dumps(chat_msg))

    async def _handle_rate_limits_updated(self, msg):
        logging.debug(f"Rate limits updated: {msg}\n")

    async def _handle_error(self, msg):
        raise Exception(f"Error: {msg}")

    async def _handle_unknown(self, msg):
        logging.error("Unknown response type in OpenAI Realtime: %s \n", msg)
