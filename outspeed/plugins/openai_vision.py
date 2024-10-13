import asyncio
import json
import logging
import os
import time
import traceback  # Add this import at the top of your file  # noqa: I001
from typing import Optional

from openai import AsyncOpenAI

from outspeed.data import ImageData, SessionData
from outspeed.plugins.vision_plugin import VisionPlugin
from outspeed.streams import TextStream, VADStream, VideoStream
from outspeed.utils.vad import VADState

logger = logging.getLogger(__name__)


class OpenAIVision(VisionPlugin):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        store_image_history: bool = True,
        stream: bool = True,
        max_output_tokens: int = 75,
    ):
        super().__init__()
        self._model: str = model

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI API key is required")

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._history = []
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._store_image_history = store_image_history
        self._stream = stream
        self._max_output_tokens = max_output_tokens
        if self._system_prompt is not None:
            self._history.append({"role": "system", "content": self._system_prompt})
        self.chat_history_queue = TextStream()
        self.output_queue = TextStream()

    async def _stream_chat_completions(self):
        while True:
            prompt = await self.input_queue.get()
            if prompt is None:
                continue

            if isinstance(prompt, SessionData):
                await self.output_queue.put(prompt)
                continue

            self._generating = True
            start_time = time.time()
            self._history.append(
                {
                    "role": "user",
                    "content": [],
                }
            )

            if isinstance(prompt, ImageData):
                logger.info("OpenAIVision prompt: %s", "image")
                content = prompt.get_pil_image()
                self._history[-1]["content"].append(
                    {"type": "image_url", "image_url": {"url": prompt.get_base64_url(quality=95)}}
                )
                self.chat_history_queue.put_nowait(
                    json.dumps(
                        {
                            "role": "user",
                            "content": "Image",
                        }
                    )
                )
            else:
                logger.info("OpenAIVision prompt: %s", prompt)
                content = prompt
                self._history[-1]["content"].append({"type": "text", "text": prompt})
                self.chat_history_queue.put_nowait(
                    json.dumps(
                        {
                            "role": "user",
                            "content": content,
                        }
                    )
                )

            try:
                chunk_stream = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=self._model,
                        stream=self._stream,
                        messages=self._history,
                        max_tokens=self._max_output_tokens,
                        temperature=self._temperature,
                    ),
                    timeout=5,
                )
            except:
                traceback.print_exc()
                logger.error("OpenAI vision timeout")
                self._history.pop()
                continue

            if self._history[-1]["content"][0].get("type") == "image_url":
                self._history.pop()
            self._history.append({"role": "assistant", "content": ""})
            logger.info("OpenAI LLM TTFB: %s", time.time() - start_time)
            if self._stream:
                async for chunk in chunk_stream:
                    if len(chunk.choices) == 0:
                        continue

                    elif chunk.choices[0].delta.content:
                        self._history[-1]["content"] += chunk.choices[0].delta.content
                        await self.output_queue.put(chunk.choices[0].delta.content)
            else:
                text = chunk_stream.choices[0].delta.content
                self._history[-1]["content"] += text
                await self.output_queue.put(text)
            logger.info("llm %s", self._history[-1]["content"])
            self.chat_history_queue.put_nowait(json.dumps(self._history[-1]))
            self._time_last_response = time.time()
            self._generating = False
            await self.output_queue.put(None)

    def run(self, input_queue: VideoStream) -> TextStream:
        self.input_queue = input_queue
        self._task = asyncio.create_task(self._stream_chat_completions())
        return self.output_queue, self.chat_history_queue

    async def close(self):
        self._task.cancel()

    async def _interrupt(self):
        while True:
            vad_state: VADState = await self.interrupt_queue.get()
            if vad_state == VADState.SPEAKING and (
                not self.input_queue.empty() or not self.output_queue.empty() or self._generating
            ):
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
