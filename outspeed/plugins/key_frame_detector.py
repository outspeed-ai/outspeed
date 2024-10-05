import asyncio
import logging
import time
from collections import deque

from outspeed.data import ImageData, SessionData
from outspeed.plugins.base_plugin import Plugin
from outspeed.streams import VideoStream
from outspeed.utils.images import (
    image_hamming_distance,
)


class KeyFrameDetector(Plugin):
    def __init__(self, key_frame_threshold=0.8, key_frame_max_time=10):
        super().__init__()
        self.video_frames_stack = deque(maxlen=1)
        self.prev_frame1 = None
        self.time_since_last_key_frame = None
        self.output_queue = VideoStream()
        self._generating = False
        self._key_frame_threshold = key_frame_threshold
        self._key_frame_max_time = key_frame_max_time

    async def process_video(self):
        try:
            while True:
                image_data: ImageData = await self.image_input_queue.get()

                if image_data is None:
                    continue

                if isinstance(image_data, SessionData):
                    self.output_queue.put(image_data)
                    continue

                while self.image_input_queue.qsize() > 0:
                    image_data: ImageData = self.image_input_queue.get_nowait()

                pil_image = image_data.get_pil_image()
                width, height = pil_image.size

                if not self._is_key_frame(pil_image):
                    continue

                await self.output_queue.put(ImageData(pil_image))
        except Exception as e:
            logging.error(f"Error in KeyFrameDetector: {e}")
            raise asyncio.CancelledError()

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
        elif self._key_frame_max_time and time.time() - self.time_since_last_key_frame > self._key_frame_max_time:
            self.prev_frame1 = frame
            self.time_since_last_key_frame = time.time()
            return True
        return False

    def run(self, image_input_queue: asyncio.Queue) -> asyncio.Queue:
        self.image_input_queue = image_input_queue
        self._tasks = [asyncio.create_task(self.process_video())]
        return self.output_queue
