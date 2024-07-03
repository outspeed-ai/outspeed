import asyncio
import time
from av import VideoFrame
from aiortc import MediaStreamTrack


class VideoRTCDriver(MediaStreamTrack):
    kind = "video"

    def __init__(self, video_input_q, video_output_q):
        super().__init__()
        self.video_output_q = video_output_q
        self.video_input_q = video_input_q
        self._track = None
        self._start = None

    async def recv(self):
        data_time = 0.010
        if self._start is None:
            self._start = time.time() + data_time
        else:
            wait = self._start - time.time() - 0.005
            self._start = time.time() + data_time
            if wait > 0:
                await asyncio.sleep(wait)
        video_frame = await self.video_output_q.get()
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
                await self.video_input_q.put(frame)
        except Exception as e:
            print("Error in video_frame_callback: ", e)
            raise asyncio.CancelledError
