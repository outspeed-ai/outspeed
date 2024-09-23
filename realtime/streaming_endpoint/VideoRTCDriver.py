import asyncio
import time

from aiortc import MediaStreamTrack

from realtime.data import ImageData


class VideoRTCDriver(MediaStreamTrack):
    kind = "video"

    def __init__(self, video_input_q, video_output_q):
        super().__init__()
        self.video_output_q = video_output_q
        self.video_input_q = video_input_q
        self._video_samples = 0
        self._track = None
        self._start = None

    async def recv(self):
        video_data = await self.video_output_q.get()
        if video_data is None or not isinstance(video_data, ImageData):
            return None
        video_frame = video_data.get_frame()
        self._video_samples = max(self._video_samples, video_frame.pts)
        video_frame.pts = self._video_samples
        self._video_samples += 1.0
        data_time = video_data.get_duration_seconds()

        if self._start is None:
            self._start = time.time() + data_time
        else:
            wait = self._start - time.time() - data_time
            if wait > 0:
                await asyncio.sleep(wait)
            self._start = max(self._start, time.time()) + data_time
        return video_frame

    def add_track(self, track):
        self._track = track

    async def run_input(self):
        try:
            if not self.video_input_q:
                return
            while not self._track:
                await asyncio.sleep(0.2)
            while True:
                frame = await self._track.recv()
                await self.video_input_q.put(ImageData(frame))
        except Exception as e:
            print("Error in video_frame_callback: ", e)
            raise asyncio.CancelledError
