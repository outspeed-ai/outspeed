import asyncio
import json
import logging
import time
from typing import Optional

import google.generativeai as genai
import PIL.PngImagePlugin  # Not used but needed to make Gemini API work with PIL  # noqa: F401

from realtime.plugins.vision_plugin import VisionPlugin
from realtime.streams import TextStream, VideoStream

logger = logging.getLogger(__name__)


class GeminiVision(VisionPlugin):
    def __init__(
        self,
        model: str = "gemini-1.5-flash-latest",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        auto_respond: Optional[int] = None,
        temperature: float = 1.0,
        wait_for_first_user_response: bool = False,
        chat_history: bool = True,
    ):
        super().__init__()
        self._model: str = model
        self._client = genai.GenerativeModel(model)
        self._history = []
        self._system_prompt = system_prompt
        self._temperature = temperature
        if self._system_prompt is not None:
            self._history.append({"role": "user", "parts": [self._system_prompt]})
        self._auto_respond = auto_respond
        self.wait_for_first_user_response = wait_for_first_user_response
        self._time_last_response = None
        self.chat_history_queue = TextStream()
        self._chat_history = chat_history
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
            if self._auto_respond:
                try:
                    prompt = await asyncio.wait_for(self.text_input_queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    if (
                        self._time_last_response is not None
                        and time.time() - self._time_last_response < self._auto_respond
                    ):
                        continue
                    prompt = ""
            else:
                prompt = await self.text_input_queue.get()
            if prompt is None:
                continue
            if prompt == "" and self.image_input_queue.qsize() == 0:
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
            if self.image_input_queue.qsize() > 0:
                while self.image_input_queue.qsize() > 0:
                    image = self.image_input_queue.get_nowait()
                self._history[-1]["parts"].append(image[0])
                logger.info("Google AI image %s", image[1])

            try:
                response = await self._client.generate_content_async(
                    self._history,
                    stream=True,
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
            if not self._chat_history:
                self._history.pop()
            self._history.append({"role": "model", "parts": [""]})
            logger.info("Google AI LLM TTFB: %s", time.time() - start_time)
            async for chunk in response:
                if chunk:
                    try:
                        text = chunk.text
                    except:
                        continue
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
            if not self._chat_history:
                self._history.pop()
            self._time_last_response = time.time()
            self._generating = False
            await self.output_queue.put(None)

    async def run(self, text_input_queue: TextStream, image_input_queue: VideoStream) -> TextStream:
        self.text_input_queue = text_input_queue
        self.image_input_queue = image_input_queue
        self._tasks = [asyncio.create_task(self._stream_chat_completions())]
        return self.output_queue, self.chat_history_queue
