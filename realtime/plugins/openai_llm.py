import asyncio
import json
import logging
import os
import time

from openai import AsyncOpenAI

from realtime.plugins.base_plugin import Plugin

logger = logging.getLogger(__name__)


class OpenAILLM(Plugin):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key=None,
        base_url=None,
        system_prompt=None,
    ):
        super().__init__()
        self._model: str = model
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key is required")
        self._api_key = api_key
        self._client = AsyncOpenAI(api_key=self._api_key, base_url=base_url)
        self._history = []
        self.output_queue = asyncio.Queue()
        self.chat_history_queue = asyncio.Queue()
        self._generating = False
        self._system_prompt = system_prompt
        if self._system_prompt is not None:
            self._history.append({"role": "system", "content": self._system_prompt})

    async def _stream_chat_completions(self):
        while True:
            text_chunk = await self.input_queue.get()
            if text_chunk is None:
                continue
            self._generating = True
            self._history.append({"role": "user", "content": text_chunk})
            self.chat_history_queue.put_nowait(json.dumps(self._history[-1]))
            start_time = time.time()
            chunk_stream = await self._client.chat.completions.create(
                model=self._model,
                stream=True,
                messages=self._history,
            )
            logger.info("OpenAI LLM TTFB: %s", time.time() - start_time)
            self._history.append({"role": "assistant", "content": ""})
            async for chunk in chunk_stream:
                if len(chunk.choices) == 0:
                    continue

                elif chunk.choices[0].delta.content:
                    self._history[-1]["content"] += chunk.choices[0].delta.content
                    await self.output_queue.put(chunk.choices[0].delta.content)
            self._generating = False
            self.chat_history_queue.put_nowait(json.dumps(self._history[-1]))
            await self.output_queue.put(None)

    async def run(self, input_queue: asyncio.Queue) -> asyncio.Queue:
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
                logger.info("Done cancelling LLM")
                self._generating = False
                self._task = asyncio.create_task(self._stream_chat_completions())

    async def set_interrupt(self, interrupt_queue: asyncio.Queue):
        self.interrupt_queue = interrupt_queue
        self._interrupt_task = asyncio.create_task(self._interrupt())
