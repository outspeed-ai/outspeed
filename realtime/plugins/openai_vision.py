import traceback  # Add this import at the top of your file  # noqa: I001

import asyncio
import json
import logging
import time

from openai import AsyncOpenAI

from realtime.plugins.vision_plugin import VisionPlugin
from realtime.streams import TextStream, VideoStream

logger = logging.getLogger(__name__)


class OpenAIVision(VisionPlugin):
    def __init__(
        self,
        model: str = "gpt-4-turbo",
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
        auto_respond: int | None = None,
        temperature: float = 1.0,
        wait_for_first_user_response: bool = False,
        key_frame_threshold: float = 0.4,
    ):
        super().__init__()
        self._model: str = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._history = []
        self._system_prompt = system_prompt
        self._temperature = temperature
        if self._system_prompt is not None:
            self._history.append({"role": "system", "content": self._system_prompt})
        self._auto_respond = auto_respond
        self.wait_for_first_user_response = wait_for_first_user_response
        self._time_last_response = None
        self.chat_history_queue = TextStream()
        self._key_frame_threshold = key_frame_threshold

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
            if prompt == "" and len(self.video_frames_stack) == 0:
                continue
            self._generating = True
            start_time = time.time()
            self._history.append(
                {
                    "role": "user",
                    "content": [],
                }
            )
            if prompt != "":
                self._history[-1]["content"].append({"type": "text", "text": prompt})
                self.chat_history_queue.put_nowait(
                    json.dumps(
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    )
                )
            if len(self.video_frames_stack) > 0:
                image = self.video_frames_stack.pop()
                self._history[-1]["content"].append({"type": "image_url", "image_url": {"url": image[0]}})
                logger.info("open ai image %s", image[1])

            try:
                chunk_stream = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=self._model,
                        stream=True,
                        messages=self._history,
                        max_tokens=75,
                        temperature=self._temperature,
                    ),
                    timeout=4,
                )
            except:
                traceback.print_exc()
                logger.error("OpenAI vision timeout")
                self._history.pop()
                continue

            if prompt != "":
                self._history[-1]["content"] = self._history[-1]["content"][:1]
            else:
                self._history[-1]["content"] = []
            self._history.append({"role": "assistant", "content": ""})
            logger.info("OpenAI LLM TTFB: %s", time.time() - start_time)
            async for chunk in chunk_stream:
                if len(chunk.choices) == 0:
                    continue

                elif chunk.choices[0].delta.content:
                    self._history[-1]["content"] += chunk.choices[0].delta.content
                    await self.output_queue.put(chunk.choices[0].delta.content)
            logger.info("llm %s", self._history[-1]["content"])
            self.chat_history_queue.put_nowait(json.dumps(self._history[-1]))
            self._time_last_response = time.time()
            self._generating = False
            await self.output_queue.put(None)

    async def run(self, text_input_queue: TextStream, image_input_queue: VideoStream) -> TextStream:
        self.text_input_queue = text_input_queue
        self.image_input_queue = image_input_queue
        self._tasks = [asyncio.create_task(self._stream_chat_completions()), asyncio.create_task(self.process_video())]
        return self.output_queue, self.chat_history_queue
