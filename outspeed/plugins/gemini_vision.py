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
from outspeed.streams import TextStream, VideoStream

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
    ):
        super().__init__()
        self._model: str = model

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if api_key is None:
            raise ValueError("Gemini API key is required")

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
            if prompt != "":
                self._history[-1]["parts"].append(prompt)
                self.chat_history_queue.put_nowait(
                    json.dumps(
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    )
                )
            try:
                response = await self._client.generate_content_async(
                    self._history,
                    stream=self._stream,
                    generation_config=genai.types.GenerationConfig(max_output_tokens=75, temperature=self._temperature),
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

    def run(self, input_queue: VideoStream) -> TextStream:
        self.input_queue = input_queue
        self._tasks = [asyncio.create_task(self._stream_chat_completions())]
        return self.output_queue, self.chat_history_queue
