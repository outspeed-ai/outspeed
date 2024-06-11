import asyncio
import os
import time
from collections import deque

from realtime.plugins.base_plugin import Plugin
from realtime.utils.images import (
    convert_yuv420_to_pil,
    image_hamming_distance,
)
from realtime.streams import VideoStream


class KeyFrameDetector(Plugin):
    def __init__(self, key_frame_threshold=0.4, key_frame_max_time=10):
        super().__init__()
        self.video_frames_stack = deque(maxlen=1)
        self.prev_frame1 = None
        self.time_since_last_key_frame = None
        self.output_queue = VideoStream()
        self._generating = False
        self._key_frame_threshold = key_frame_threshold
        self._key_frame_max_time = key_frame_max_time

    async def process_video(self):
        i = 1
        while True:
            image = await self.image_input_queue.get()
            while self.image_input_queue.qsize() > 0:
                image = self.image_input_queue.get_nowait()

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

            await self.output_queue.put((im1, i))
            # if not os.path.exists("data"):
            #     os.makedirs("data")
            # im1.save(f"data/{i}.jpeg")
            i += 1

    async def close(self):
        for task in self._tasks:
            task.cancel()

    async def _interrupt(self):
        while True:
            user_speaking = await self.interrupt_queue.get()
            if self._generating and user_speaking:
                self._task.cancel()
                while not self.output_queue.empty():
                    self.output_queue.get_nowait()
                print("Done cancelling LLM")
                self._generating = False
                self._tasks = [asyncio.create_task(self.process_video())]

    async def set_interrupt(self, interrupt_queue: asyncio.Queue):
        self.interrupt_queue = interrupt_queue
        self._interrupt_task = asyncio.create_task(self._interrupt())

    def _is_key_frame(self, frame):
        if self.prev_frame1 is None:
            self.prev_frame1 = frame
            self.time_since_last_key_frame = time.time()
            return True
        if time.time() - self.time_since_last_key_frame < 1.0:
            return False
        d3 = image_hamming_distance(self.prev_frame1, frame)
        if d3 >= self._key_frame_threshold:
            self.prev_frame1 = frame
            self.time_since_last_key_frame = time.time()
            return True
        elif (
            self._key_frame_max_time
            and time.time() - self.time_since_last_key_frame > self._key_frame_max_time
        ):
            self.prev_frame1 = frame
            self.time_since_last_key_frame = time.time()
            return True
        return False

    async def run(self, image_input_queue: asyncio.Queue) -> asyncio.Queue:
        self.image_input_queue = image_input_queue
        self._tasks = [asyncio.create_task(self.process_video())]
        return self.output_queue
