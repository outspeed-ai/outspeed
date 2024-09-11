import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from realtime.plugins.base_plugin import Plugin
from realtime.streams import TextStream
from realtime.utils import tracing
from realtime.data import SessionData


class GroqLLM(Plugin):
    """
    A plugin for interacting with Groq's LLM API.

    This class handles streaming chat completions, manages chat history,
    and provides methods for running the LLM and handling interruptions.
    """

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stream: bool = True,
        temperature: float = 1.0,
        response_format: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the GroqLLM plugin.

        Args:
            model (str): The name of the Groq model to use.
            api_key (Optional[str]): The API key for Groq. If None, it will be read from the environment.
            base_url (Optional[str]): The base URL for the Groq API. Not used in the current implementation.
            system_prompt (Optional[str]): The system prompt to be used in the conversation.
            stream (bool): Whether to stream the response or not.
            temperature (float): The temperature parameter for the LLM.
            response_format (Optional[Dict[str, Any]]): The desired response format.
        """
        super().__init__()
        self._model: str = model
        self._api_key: str = api_key or os.environ.get("GROQ_API_KEY")
        if self._api_key is None:
            raise ValueError("Groq API key is required")
        self._client: AsyncOpenAI = AsyncOpenAI(api_key=self._api_key, base_url="https://api.groq.com/openai/v1")
        self._history: List[Dict[str, str]] = []
        self.output_queue: TextStream = TextStream()
        self.chat_history_queue: TextStream = TextStream()
        self._generating: bool = False
        self._stream: bool = stream
        self._response_format: Optional[Dict[str, Any]] = response_format
        self._system_prompt: Optional[str] = system_prompt
        if self._system_prompt is not None:
            self._history.append({"role": "system", "content": self._system_prompt})
        self._temperature: float = temperature

        # These will be set later
        self.input_queue: Optional[TextStream] = None
        self.interrupt_queue: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None
        self._interrupt_task: Optional[asyncio.Task] = None

    async def _stream_chat_completions(self) -> None:
        """
        Asynchronously stream chat completions from the Groq API.

        This method continuously reads from the input queue, sends requests to the Groq API,
        and writes the responses to the output queue.
        """
        while True:
            text_chunk: Optional[str] = await self.input_queue.get()
            if text_chunk is None:
                continue
            if isinstance(text_chunk, SessionData):
                await self.output_queue.put(text_chunk)
                continue
            self._generating = True
            tracing.register_event(tracing.Event.LLM_START)

            # Add user message to history and chat history queue
            self._history.append({"role": "user", "content": text_chunk})
            self.chat_history_queue.put_nowait(json.dumps(self._history[-1]))

            # Create chat completion request
            completion_kwargs: Dict[str, Any] = {
                "model": self._model,
                "stream": self._stream,
                "messages": self._history,
                "temperature": self._temperature,
            }
            if self._response_format:
                completion_kwargs["response_format"] = self._response_format

            chunk_stream = await self._client.chat.completions.create(**completion_kwargs)

            # Prepare for assistant's response
            self._history.append({"role": "assistant", "content": ""})
            first_chunk = True

            if self._stream:
                async for chunk in chunk_stream:
                    if first_chunk:
                        tracing.register_event(tracing.Event.LLM_TTFB)
                        first_chunk = False
                    if len(chunk.choices) == 0:
                        continue
                    elif chunk.choices[0].delta.content:
                        self._history[-1]["content"] += chunk.choices[0].delta.content
                        await self.output_queue.put(chunk.choices[0].delta.content)
            else:
                self._history[-1]["content"] = chunk_stream.choices[0].message.content
                await self.output_queue.put(chunk_stream.choices[0].message.content)
                tracing.register_event(tracing.Event.LLM_TTFB)

            tracing.register_event(tracing.Event.LLM_END)
            tracing.register_metric(tracing.Metric.LLM_TOTAL_BYTES, len(self._history[-1]["content"]))
            print("llm", self._history[-1]["content"])
            self.chat_history_queue.put_nowait(json.dumps(self._history[-1]))
            self._generating = False
            await self.output_queue.put(None)

    def run(self, input_queue: TextStream) -> Tuple[TextStream, TextStream]:
        """
        Start the LLM plugin.

        Args:
            input_queue (TextStream): The input queue to read from.

        Returns:
            Tuple[TextStream, TextStream]: The output queue and chat history queue.
        """
        self.input_queue = input_queue
        self._task = asyncio.create_task(self._stream_chat_completions())
        return self.output_queue, self.chat_history_queue

    async def close(self) -> None:
        """
        Close the LLM plugin by cancelling the main task.
        """
        if self._task:
            self._task.cancel()

    async def _interrupt(self) -> None:
        """
        Handle interruptions to the LLM generation.

        This method listens for interrupt signals and cancels the current generation if necessary.
        """
        while True:
            user_speaking: bool = await self.interrupt_queue.get()
            if self._generating and user_speaking:
                if self._task:
                    self._task.cancel()
                while not self.output_queue.empty():
                    self.output_queue.get_nowait()
                print("Done cancelling LLM")
                self._generating = False
                self._task = asyncio.create_task(self._stream_chat_completions())

    async def set_interrupt(self, interrupt_queue: asyncio.Queue) -> None:
        """
        Set up the interrupt mechanism for the LLM plugin.

        Args:
            interrupt_queue (asyncio.Queue): The queue to listen for interrupt signals.
        """
        self.interrupt_queue = interrupt_queue
        self._interrupt_task = asyncio.create_task(self._interrupt())
