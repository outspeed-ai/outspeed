import asyncio
import time

import aiohttp
import fal_client


class FalVision:
    def __init__(
        self,
        *,
        api_key: str,
        model="fal-ai/llavav15-13b",
    ):
        super().__init__()
        self._model = model

        self._aiohttp_session = aiohttp.ClientSession()
        self._api_key = api_key
        self.output_queue = asyncio.Queue()

    async def run(self, text_input_queue: asyncio.Queue, image_input_queue: asyncio.Queue) -> asyncio.Queue:
        self.text_input_queue = text_input_queue
        self.image_input_queue = image_input_queue
        self._task = asyncio.create_task(self.astream())
        return self.output_queue

    async def astream(self):
        async def process_video():
            while True:
                self.current_video_frame = await self.image_input_queue.get()

        async def vision():
            while True:
                prompt = await self.text_input_queue.get()
                image = self.current_video_frame
                start_time = time.time()
                image_pil = image.to_image()
                image_url = fal_client.encode_image(image_pil)
                print(f"Processing image took {time.time() - start_time} seconds")
                stream = fal_client.stream_async(self._model, arguments={"prompt": prompt, "image_url": image_url})
                first = True
                result = ""
                async for event in stream:
                    if first:
                        first = False
                        print(f"First event took {time.time() - start_time} seconds")
                    result = event["output"]

                await self.output_queue.put(result)

        await asyncio.gather(process_video(), vision())

    async def aclose(self):
        await self._aiohttp_session.close()
