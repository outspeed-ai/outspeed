import traceback  # Add this import at the top of your file  # noqa: I001
import asyncio
import json
import logging
import time

from openai import AsyncAssistantEventHandler, AsyncOpenAI
from typing_extensions import override

from realtime.plugins.vision_plugin import VisionPlugin
from realtime.utils.images import convert_yuv420_to_pil

logger = logging.getLogger(__name__)


class EventHandler(AsyncAssistantEventHandler):
    def __init__(self, output_queue: asyncio.Queue, history: list):
        super().__init__()
        self.queue = output_queue
        self.history = history

    @override
    async def on_text_delta(self, delta, snapshot):
        logger.info("delta time %s", time.time())
        self.history[-1]["content"] += delta.value
        self.queue.put_nowait(delta.value)


class OpenAIVisionAssistant(VisionPlugin):
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
        self.chat_history_queue = asyncio.Queue()
        self._key_frame_threshold = key_frame_threshold

    async def _stream_chat_completions(self):
        self.assistant = await self._client.beta.assistants.create(
            name="Assistant",
            instructions=self._system_prompt,
            tools=[],
            model=self._model,
        )
        self._thread = await self._client.beta.threads.create()
        while True:
            if self._auto_respond:
                try:
                    if (
                        self._time_last_response is not None
                        and time.time() - self._time_last_response < self._auto_respond
                    ):
                        await asyncio.sleep(self._auto_respond - time.time() + self._time_last_response)
                    prompt = await asyncio.wait_for(self.text_input_queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
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
                print("got image", image)
                self._history[-1]["content"].append({"type": "image_file", "image_file": {"file_id": image[0]}})
                logger.info("open ai image %s", image[1])

            try:
                chunk_stream = await self._client.beta.threads.messages.create(
                    thread_id=self._thread.id, role="user", content=self._history[-1]["content"]
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
            logger.info("OpenAI LLM TTFB: %s %s", time.time() - start_time, time.time())
            async with self._client.beta.threads.runs.stream(
                thread_id=self._thread.id,
                assistant_id=self.assistant.id,
                event_handler=EventHandler(self.output_queue, self._history),
            ) as stream:
                await stream.until_done()
            logger.info("llm %s", self._history[-1]["content"])
            self.chat_history_queue.put_nowait(json.dumps(self._history[-1]))
            self._time_last_response = time.time()
            self._generating = False
            await self.output_queue.put(None)

    async def run(self, text_input_queue: asyncio.Queue, image_input_queue: asyncio.Queue) -> asyncio.Queue:
        self.text_input_queue = text_input_queue
        self.image_input_queue = image_input_queue
        self._tasks = [asyncio.create_task(self._stream_chat_completions()), asyncio.create_task(self.process_video())]
        return self.output_queue, self.chat_history_queue

    async def process_video(self):
        i = 1
        while True:
            image = await self.image_input_queue.get()
            if self.image_input_queue.qsize() > 0:
                continue
            if image is None:
                continue
            t = time.time()
            pil_image = convert_yuv420_to_pil(image)
            width, height = pil_image.size

            # Setting the points for cropped image
            left = 0
            top = height / 2
            right = width / 2
            bottom = height

            # Cropped image of above dimension
            # (It will not change original image)
            im1 = pil_image.crop((left, top, right, bottom))
            if not self._is_key_frame(im1):
                continue

            logger.info("open ai image processing: %s", time.time() - t)
            im1.save(f"data/{i}.jpeg", quality=95)
            logger.info("open ai image processing: %s", time.time() - t)
            file = await self._client.files.create(file=open(f"data/{i}.jpeg", "rb"), purpose="vision")
            self.video_frames_stack.append((file.id, i))
            logger.info("open ai image processing: %s", time.time() - t)
            i += 1
