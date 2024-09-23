import asyncio
import json
import logging
import os
from typing import Tuple

from openai import AsyncOpenAI

from realtime.plugins.base_plugin import Plugin
from realtime.streams import TextStream, VADStream
from realtime.utils import tracing
from realtime.utils.vad import VADState
from realtime.data import SessionData


class FireworksLLM(Plugin):
    def __init__(
        self,
        model: str = "accounts/fireworks/models/llama-v3p1-8b-instruct",
        api_key=None,
        base_url=None,
        system_prompt=None,
        stream: bool = True,
        temperature: float = 1.0,
        response_format: dict = None,
    ):
        super().__init__()
        self._model: str = model
        api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if api_key is None:
            raise ValueError("Fireworks API key is required")
        self._api_key = api_key
        self._client = AsyncOpenAI(api_key=api_key, base_url="https://api.fireworks.ai/inference/v1")
        self._history = []
        self.output_queue = TextStream()
        self.chat_history_queue = TextStream()
        self._generating = False
        self._stream = stream
        self._response_format = response_format
        self._system_prompt = system_prompt
        if self._system_prompt is not None:
            self._history.append({"role": "system", "content": self._system_prompt})
        self._temperature = temperature

    async def _stream_chat_completions(self):
        try:
            while True:
                text_chunk = await self.input_queue.get()
                if text_chunk is None:
                    continue

                if isinstance(text_chunk, SessionData):
                    await self.output_queue.put(text_chunk)
                    continue
                self._generating = True
                self._history.append({"role": "user", "content": text_chunk})
                self.chat_history_queue.put_nowait(json.dumps(self._history[-1]))
                tracing.register_event(tracing.Event.LLM_START)
                if self._response_format:
                    chunk_stream = await self._client.chat.completions.create(
                        model=self._model,
                        stream=self._stream,
                        messages=self._history,
                        response_format=self._response_format,
                        temperature=self._temperature,
                    )
                else:
                    chunk_stream = await self._client.chat.completions.create(
                        model=self._model,
                        stream=self._stream,
                        messages=self._history,
                        temperature=self._temperature,
                    )
                self._history.append({"role": "assistant", "content": ""})
                tracing.register_event(tracing.Event.LLM_TTFB)
                if self._stream:
                    async for chunk in chunk_stream:
                        if len(chunk.choices) == 0:
                            continue

                        elif chunk.choices[0].delta.content:
                            self._history[-1]["content"] += chunk.choices[0].delta.content
                            await self.output_queue.put(chunk.choices[0].delta.content)
                else:
                    self._history[-1]["content"] = chunk_stream.choices[0].message.content
                    await self.output_queue.put(chunk_stream.choices[0].message.content)
                tracing.register_event(tracing.Event.LLM_END)
                tracing.register_metric(tracing.Metric.LLM_TOTAL_BYTES, len(self._history[-1]["content"]))
                logging.info("llm: %s", self._history[-1]["content"])
                self.chat_history_queue.put_nowait(json.dumps(self._history[-1]))
                self._generating = False
                await self.output_queue.put(None)
        except Exception as e:
            logging.error("Error streaming chat completions", e)
            self._generating = False
            raise asyncio.CancelledError()

    def run(self, input_queue: TextStream) -> Tuple[TextStream, TextStream]:
        self.input_queue = input_queue
        self._task = asyncio.create_task(self._stream_chat_completions())
        return self.output_queue, self.chat_history_queue

    async def close(self):
        self._task.cancel()

    async def _interrupt(self):
        while True:
            vad_state: VADState = await self.interrupt_queue.get()
            if vad_state == VADState.SPEAKING and (not self.input_queue.empty() or not self.output_queue.empty()):
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                while not self.output_queue.empty():
                    self.output_queue.get_nowait()
                while not self.input_queue.empty():
                    self.input_queue.get_nowait()
                logging.info("Done cancelling LLM")
                self._generating = False
                self._task = asyncio.create_task(self._stream_chat_completions())

    def set_interrupt_stream(self, interrupt_stream: VADStream):
        if isinstance(interrupt_stream, VADStream):
            self.interrupt_queue = interrupt_stream
        else:
            raise ValueError("Interrupt stream must be a VADStream")
        self._interrupt_task = asyncio.create_task(self._interrupt())
