import asyncio
import logging
import time
from collections import deque

from outspeed.data import ImageData, SessionData
from outspeed.plugins.base_plugin import Plugin
from outspeed.streams import VADStream, VideoStream
from outspeed.utils.images import (
    image_euclidean_distance,
)
from outspeed.utils.vad import VADState


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
                image_data: ImageData = await self.input_queue.get()

                if image_data is None:
                    continue

                if isinstance(image_data, SessionData):
                    self.output_queue.put(image_data)
                    continue

                self._generating = True

                while self.input_queue.qsize() > 0:
                    image_data: ImageData = self.input_queue.get_nowait()

                pil_image = image_data.get_pil_image()
                width, height = pil_image.size

                if not self._is_key_frame(pil_image):
                    continue

                logging.info("Key frame detected")
                await self.output_queue.put(ImageData(pil_image))

                self._generating = False
        except Exception as e:
            logging.error(f"Error in KeyFrameDetector: {e}")
            raise asyncio.CancelledError()

    def _is_key_frame(self, frame):
        if self.prev_frame1 is None:
            self.prev_frame1 = frame
            self.time_since_last_key_frame = time.time()
            return True
        if time.time() - self.time_since_last_key_frame < 1.0:
            return False
        d3 = image_euclidean_distance(self.prev_frame1, frame)
        if d3 >= self._key_frame_threshold:
            self.prev_frame1 = frame
            self.time_since_last_key_frame = time.time()
            return True
        elif self._key_frame_max_time and time.time() - self.time_since_last_key_frame > self._key_frame_max_time:
            self.prev_frame1 = frame
            self.time_since_last_key_frame = time.time()
            return True
        return False

    def run(self, input_queue: asyncio.Queue) -> asyncio.Queue:
        self.input_queue = input_queue
        self._task = asyncio.create_task(self.process_video())
        return self.output_queue

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
                logging.info("Done cancelling KeyFrameDetector")
                self._generating = False
                self._task = asyncio.create_task(self.process_video())

    def set_interrupt_stream(self, interrupt_stream: VADStream):
        if isinstance(interrupt_stream, VADStream):
            self.interrupt_queue = interrupt_stream
        else:
            raise ValueError("Interrupt stream must be a VADStream")
        self._interrupt_task = asyncio.create_task(self._interrupt())
