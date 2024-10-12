from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Dict, List, Literal, Optional

from outspeed.plugins.openai_realtime.types import (
    ConversationCreated,
    ConversationItem,
    ConversationItemCreated,
    ConversationItemInputAudioTranscriptionCompleted,
    InputAudioTranscription,
    ResponseDone,
    ServerVad,
    SessionCreated,
    SessionUpdated,
)


class RealtimeSession:
    def __init__(
        self,
        session_id: str,
        modalities: List[Literal["text", "audio"]] = ["text", "audio"],
        instructions: Optional[str] = None,
        voice: Optional[str] = None,
        input_audio_format: Optional[str] = None,
        output_audio_format: Optional[str] = None,
        input_audio_transcription: Optional[InputAudioTranscription] = None,
        turn_detection: Optional[ServerVad] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_response_output_tokens: Optional[int] = None,
    ) -> None:
        self.session_id = session_id
        self.modalities = modalities
        self.instructions = instructions
        self.voice = voice
        self.input_audio_format = input_audio_format
        self.output_audio_format = output_audio_format
        self.input_audio_transcription = input_audio_transcription
        self.turn_detection = turn_detection
        self.tool_choice = tool_choice
        self.temperature = temperature
        self.max_response_output_tokens = max_response_output_tokens
        self.items: OrderedDict[str, ConversationItem] = OrderedDict()
        self.conversations: List[Conversation] = []
        self._chat_history: List[Dict[str, str]] = []

    def session_update(
        self,
    ) -> None:
        server_vad_opts: ServerVad = {
            "type": "server_vad",
            "threshold": self.turn_detection.threshold,
            "prefix_padding_ms": self.turn_detection.prefix_padding_ms,
            "silence_duration_ms": self.turn_detection.silence_duration_ms,
        }

        self._queue_msg(
            {
                "type": "session.update",
                "session": {
                    "modalities": self._opts.modalities,
                    "instructions": self._opts.instructions,
                    "voice": self._opts.voice,
                    "input_audio_format": self._opts.input_audio_format,
                    "output_audio_format": self._opts.output_audio_format,
                    "input_audio_transcription": {
                        "model": self._opts.input_audio_transcription.model,
                    },
                    "turn_detection": server_vad_opts,
                    "tools": tools,
                    "tool_choice": self._opts.tool_choice,
                    "temperature": self._opts.temperature,
                    "max_response_output_tokens": self._opts.max_response_output_tokens,
                },
            }
        )

    @classmethod
    def from_dict(cls, data: SessionCreated) -> RealtimeSession:
        session = data["session"]
        return cls(
            session_id=session["id"],
            modalities=session["modalities"],
            instructions=session["instructions"],
            voice=session["voice"],
            input_audio_format=session["input_audio_format"],
            output_audio_format=session["output_audio_format"],
            input_audio_transcription=session["input_audio_transcription"],
            turn_detection=session["turn_detection"],
            tool_choice=session["tool_choice"],
            temperature=session["temperature"],
            max_response_output_tokens=session["max_response_output_tokens"],
        )

    def update_from_dict(self, data: SessionUpdated):
        pass

    def add_item(self, data: ConversationItemCreated):
        self.items[data["item"]["id"]] = data["item"]

    def add_input_audio_transcription(self, data: ConversationItemInputAudioTranscriptionCompleted):
        self._chat_history.append({"role": "user", "content": data["transcript"]})
        item = self.items.get(data["item_id"])
        if not item:
            self.items[data["item_id"]] = {
                "id": data["item_id"],
                "object": "realtime.item",
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "audio": data["transcript"],
                    }
                ],
            }
            logging.error(f"Item {data['item_id']} not found")
            return self._chat_history[-1]

        item["content"][data["content_index"]]["transcript"] = data["transcript"]
        return self._chat_history[-1]

    def add_response(self, data: ResponseDone):
        responses = []
        for item in data["response"].get("output", []):
            if item["type"] == "message":
                content = item["content"][0]
                if content["type"] == "text":
                    self._chat_history.append({"role": "assistant", "content": content["text"]})
                    responses.append({"role": "assistant", "content": content["text"]})
                elif content["type"] == "audio":
                    self._chat_history.append({"role": "assistant", "content": content["transcript"]})
                    responses.append({"role": "assistant", "content": content["transcript"]})
            self.items[item["id"]] = item
        return responses

    def add_conversation(self, data: ConversationCreated):
        self.conversations.append(data["conversation"])

    def get_items(self) -> List[ConversationItem]:
        return list(self.items.values())

    def get_chat_history(self) -> List[Dict[str, str]]:
        return self._chat_history
