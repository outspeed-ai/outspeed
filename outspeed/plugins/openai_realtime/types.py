from enum import Enum
from typing import Literal, Optional, TypedDict, Union

from outspeed.plugins.openai_realtime.events import ServerEvent


class AudioFormatType(str, Enum):
    PCM16 = "pcm16"
    G711_ULAW = "g711-ulaw"
    G711_ALAW = "g711-alaw"


class FunctionCallType(str, Enum):
    FUNCTION = "function"


class ItemStatusType(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"


class RoleType(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class SessionVoiceType(str, Enum):
    ALLY = "alloy"
    SHIMMER = "shimmer"
    ECHO = "echo"


class ToolChoiceType(str, Enum):
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


class InputAudioTranscription(TypedDict):
    model: str


class ServerVad(TypedDict):
    type: Literal["server_vad"]
    threshold: Optional[float]
    prefix_padding_ms: Optional[int]
    silence_duration_ms: Optional[int]


class FunctionTool(TypedDict):
    type: Literal["function"]
    name: str
    description: Optional[str]
    parameters: dict


class Session(TypedDict):
    id: str
    object: Literal[ServerEvent.SESSION_CREATED]
    expires_at: int
    model: str
    modalities: list[Literal["text", "audio"]]
    instructions: str
    voice: SessionVoiceType
    input_audio_format: AudioFormatType
    output_audio_format: AudioFormatType
    input_audio_transcription: Optional[InputAudioTranscription]
    turn_detection: Optional[ServerVad]
    tools: list[FunctionTool]
    tool_choice: ToolChoiceType
    temperature: float
    max_response_output_tokens: Optional[int]


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class InputTextContent(TypedDict):
    type: Literal["input_text"]
    text: str


class AudioContent(TypedDict):
    type: Literal["audio"]
    audio: str  # b64


class InputAudioContent(TypedDict):
    type: Literal["input_audio"]
    audio: str  # b64


class FunctionTool(TypedDict):
    type: Literal["function"]
    name: str
    description: Optional[str]
    parameters: dict


class SystemItem(TypedDict):
    id: str
    object: Literal["realtime.item"]
    type: Literal["message"]
    role: Literal["system"]
    content: list[InputTextContent]


class UserItem(TypedDict):
    id: str
    object: Literal["realtime.item"]
    type: Literal["message"]
    role: Literal["user"]
    content: list[Union[InputTextContent, InputAudioContent]]


class AssistantItem(TypedDict):
    id: str
    object: Literal["realtime.item"]
    type: Literal["message"]
    role: Literal["assistant"]
    content: list[Union[TextContent, AudioContent]]


class FunctionCallItem(TypedDict):
    id: str
    object: Literal["realtime.item"]
    type: Literal["function_call"]
    call_id: str
    name: str
    arguments: str


class FunctionCallOutputItem(TypedDict):
    id: str
    object: Literal["realtime.item"]
    type: Literal["function_call_output"]
    call_id: str
    output: str


class SessionCreated(TypedDict):
    event_id: str
    type: Literal[ServerEvent.SESSION_CREATED]
    session: Session


class SessionUpdated(TypedDict):
    event_id: str
    type: Literal[ServerEvent.SESSION_UPDATED]
    session: Session


ConversationItem = Union[SystemItem, UserItem, FunctionCallItem, FunctionCallOutputItem]


class ConversationItemCreated(TypedDict):
    event_id: str
    type: Literal[ServerEvent.CONVERSATION_ITEM_CREATED]
    item: ConversationItem


class ResponseAudioDelta(TypedDict):
    event_id: str
    type: Literal[ServerEvent.RESPONSE_AUDIO_DELTA]
    response_id: str
    output_index: int
    content_index: int
    delta: str  # b64


class InputAudioBufferSpeechStarted(TypedDict):
    event_id: str
    type: Literal[ServerEvent.INPUT_AUDIO_BUFFER_SPEECH_STARTED]
    item_id: str
    audio_start_ms: int


class ResponseAudioTranscriptDelta(TypedDict):
    event_id: str
    type: Literal[ServerEvent.RESPONSE_AUDIO_TRANSCRIPT_DELTA]
    response_id: str
    output_index: int
    content_index: int
    delta: str
    item_id: str


class CancelledStatusDetails(TypedDict):
    type: Literal["cancelled"]
    reason: Literal["turn_detected", "client_cancelled"]


class IncompleteStatusDetails(TypedDict):
    type: Literal["incomplete"]
    reason: Literal["max_output_tokens", "content_filter"]


class Error(TypedDict):
    code: str
    message: str


class FailedStatusDetails(TypedDict):
    type: Literal["failed"]
    error: Optional[Error]


ResponseStatusDetails = Union[CancelledStatusDetails, IncompleteStatusDetails, FailedStatusDetails]


class Usage(TypedDict):
    total_tokens: int
    input_tokens: int
    output_tokens: int


class Response(TypedDict):
    id: str
    object: Literal["realtime.response"]
    status: Literal["in_progress", "completed", "incomplete", "cancelled", "failed"]
    status_details: Optional[ResponseStatusDetails]
    output: list[ConversationItem]
    usage: Optional[Usage]


class ResponseTextDeltaAdded(TypedDict):
    event_id: str
    type: Literal[ServerEvent.RESPONSE_TEXT_DELTA]
    response_id: str
    output_index: int
    content_index: int
    delta: str
    item_id: str


class Conversation(TypedDict):
    id: str
    object: Literal["realtime.conversation"]


class ConversationCreated(TypedDict):
    event_id: str
    type: Literal[ServerEvent.CONVERSATION_CREATED]
    conversation: Conversation


class ConversationItemInputAudioTranscriptionCompleted(TypedDict):
    event_id: str
    type: Literal[ServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED]
    item_id: str
    content_index: int
    transcript: str


class ContentPart(TypedDict):
    type: Literal["text", "audio"]
    audio: Optional[str]  # b64
    transcript: Optional[str]


class ResponseContentPartAdded(TypedDict):
    event_id: str
    type: Literal[ServerEvent.RESPONSE_CONTENT_PART_ADDED]
    item_id: str
    response_id: str
    output_index: int
    content_index: int
    part: ContentPart


class ResponseDone(TypedDict):
    event_id: str
    type: Literal[ServerEvent.RESPONSE_DONE]
    response: Response


class ResponseFunctionCallArgumentsDone(TypedDict):
    event_id: str
    type: Literal[ServerEvent.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE]
    response_id: str
    output_index: int
    arguments: str
    name: str
    item_id: str
    call_id: str
