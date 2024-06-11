import asyncio
import time

from openai import AsyncOpenAI

from realtime.plugins.vision_plugin import VisionPlugin


class FireworksVision(VisionPlugin):
    def __init__(
        self,
        model: str = "accounts/fireworks/models/firellava-13b",
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
        auto_respond: int | None = None,
        temperature: float = 1.0,
        wait_for_first_user_response: bool = False,
    ):
        super().__init__()
        self._model: str = model
        self._client = AsyncOpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=api_key)
        self._history = []
        self.output_queue = asyncio.Queue()
        self._generating = False
        self.current_video_frame = None
        self._system_prompt = system_prompt
        self._temperature = temperature
        if self._system_prompt is not None:
            self._history.append({"role": "system", "content": self._system_prompt})
        self._auto_respond = auto_respond
        self.wait_for_first_user_response = wait_for_first_user_response

    async def _stream_chat_completions(self):
        while True:
            try:
                if self.wait_for_first_user_response:
                    prompt = await self.text_input_queue.get()
                    self.wait_for_first_user_response = False
                else:
                    prompt = await asyncio.wait_for(self.text_input_queue.get(), timeout=self._auto_respond)
                if prompt is None:
                    continue
            except asyncio.TimeoutError:
                prompt = self._system_prompt
            if len(self.video_frames_stack) == 0:
                continue
            self._generating = True
            start_time = time.time()
            self._history.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            )
            if len(self.video_frames_stack) > 0:
                image = self.video_frames_stack.pop()
                self._history[-1]["content"].append({"type": "image_url", "image_url": {"url": image[0]}})
            chunk_stream = await self._client.chat.completions.create(
                model=self._model,
                stream=True,
                messages=self._history,
                max_tokens=50,
            )
            self._history[-1]["content"] = self._history[-1]["content"][:1]
            print(f"=== OpenAI LLM TTFB: {time.time() - start_time}")
            self._history.append({"role": "assistant", "content": [{"type": "text", "text": ""}]})
            async for chunk in chunk_stream:
                if len(chunk.choices) == 0:
                    continue

                elif chunk.choices[0].delta.content:
                    self._history[-1]["content"][0]["text"] += chunk.choices[0].delta.content
                    await self.output_queue.put(chunk.choices[0].delta.content)
            print("llm", self._history[-1]["content"][0]["text"])
            self._generating = False
            await self.output_queue.put(None)

    async def run(self, text_input_queue: asyncio.Queue, image_input_queue: asyncio.Queue) -> asyncio.Queue:
        self.text_input_queue = text_input_queue
        self.image_input_queue = image_input_queue
        self._tasks = [asyncio.create_task(self._stream_chat_completions()), asyncio.create_task(self.process_video())]
        return self.output_queue
