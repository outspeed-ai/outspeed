import asyncio
import json
import os
import time
from typing import Tuple

from openai import AsyncOpenAI

from realtime.plugins.base_plugin import Plugin

from realtime.streams import AudioStream, VideoStream, Stream, TextStream, ByteStream


class FireworksLLM(Plugin):
    def __init__(
        self,
        model: str = "accounts/fireworks/models/llama-v2-7b-chat",
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
        while True:
            text_chunk = await self.input_queue.get()
            if text_chunk is None:
                continue
            self._generating = True
            self._history.append({"role": "user", "content": text_chunk})
            self.chat_history_queue.put_nowait(json.dumps(self._history[-1]))
            start_time = time.time()
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
            print(f"=== OpenAI LLM TTFB: {time.time() - start_time}")
            self._history.append({"role": "assistant", "content": ""})
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
            print("llm", self._history[-1]["content"])
            self.chat_history_queue.put_nowait(json.dumps(self._history[-1]))
            self._generating = False

    async def run(self, input_queue: TextStream) -> Tuple[TextStream, TextStream]:
        self.input_queue = input_queue
        self._task = asyncio.create_task(self._stream_chat_completions())
        return self.output_queue, self.chat_history_queue

    async def close(self):
        self._task.cancel()

    async def _interrupt(self):
        while True:
            user_speaking = await self.interrupt_queue.get()
            if self._generating and user_speaking:
                self._task.cancel()
                while not self.output_queue.empty():
                    self.output_queue.get_nowait()
                print("Done cancelling LLM")
                self._generating = False
                self._task = asyncio.create_task(self._stream_chat_completions())

    async def set_interrupt(self, interrupt_queue: asyncio.Queue):
        self.interrupt_queue = interrupt_queue
        self._interrupt_task = asyncio.create_task(self._interrupt())
