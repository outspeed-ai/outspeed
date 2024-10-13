import asyncio
import json
import logging
import os
import time
from typing import Optional

import google.generativeai as genai
import PIL.PngImagePlugin  # Not used but needed to make Gemini API work with PIL  # noqa: F401

from outspeed.data import SessionData
from outspeed.plugins.base_plugin import Plugin
from outspeed.streams import TextStream, VideoStream, VADStream
from outspeed.data import ImageData
from outspeed.utils.vad import VADState

logger = logging.getLogger(__name__)


class GeminiVision(Plugin):
    def __init__(
        self,
        model: str = "gemini-1.5-flash-latest",
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

        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE API key is required")

        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(model)
        self._history = []
        self._system_prompt = system_prompt
        self._temperature = temperature
        self.chat_history_queue = TextStream()
        if self._system_prompt is not None:
            self._history.append({"role": "user", "parts": [self._system_prompt]})
        self.output_queue = TextStream()
        self._store_image_history = store_image_history
        self._stream = stream
        self._max_output_tokens = max_output_tokens
        self._safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]

    async def _stream_chat_completions(self):
        try:
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
                        "parts": [],
                    }
                )
                if isinstance(prompt, ImageData):
                    logger.info("GeminiVision prompt: %s", "image")
                    content = prompt.get_pil_image()
                    self._history[-1]["parts"].append(content)
                    self.chat_history_queue.put_nowait(
                        json.dumps(
                            {
                                "role": "user",
                                "content": "Image",
                            }
                        )
                    )
                else:
                    logger.info("GeminiVision prompt: %s", prompt)
                    content = prompt
                    self._history[-1]["parts"].append(content)
                    self.chat_history_queue.put_nowait(
                        json.dumps(
                            {
                                "role": "user",
                                "content": content,
                            }
                        )
                    )

                try:
                    response = await self._client.generate_content_async(
                        self._history,
                        stream=self._stream,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=self._max_output_tokens, temperature=self._temperature
                        ),
                        safety_settings=self._safety_settings,
                    )
                except Exception as e:
                    logger.error("Google AI vision timeout %s", e)
                    self._history.pop()
                    continue

                # if prompt != "":
                #     self._history[-1]["parts"] = self._history[-1]["parts"][:1]
                # else:
                if not self._store_image_history:
                    self._history.pop()
                self._history.append({"role": "model", "parts": [""]})
                logger.info("Google AI LLM TTFB: %s", time.time() - start_time)
                if self._stream:
                    async for chunk in response:
                        if chunk:
                            try:
                                text = chunk.text
                            except:
                                continue
                            self._history[-1]["parts"][0] += text
                            await self.output_queue.put(text)
                else:
                    text = response.text
                    self._history[-1]["parts"][0] += text
                    await self.output_queue.put(text)
                logger.info("llm %s", self._history[-1]["parts"][0])
                self.chat_history_queue.put_nowait(
                    json.dumps(
                        {
                            "role": "assistant",
                            "content": self._history[-1]["parts"][0],
                        }
                    )
                )
                if not self._store_image_history:
                    self._history.pop()
                self._generating = False
                await self.output_queue.put(None)
        except Exception as e:
            logger.error("GeminiVision error %s", e)
            raise asyncio.CancelledError()

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
