import asyncio
import time
from av import AudioFrame
from aiortc import MediaStreamTrack


class AudioRTCDriver(MediaStreamTrack):
    kind = "audio"

    def __init__(self, audio_input_q, audio_output_q):
        super().__init__()
        self.audio_input_q = audio_input_q
        self.audio_output_q = audio_output_q
        self._start = None
        self._track = None

    async def recv(self):
        data_time = 0.020
        if self._start is None:
            self._start = time.time() + data_time
        else:
            wait = self._start - time.time() - 0.005
            self._start = time.time() + data_time
            if wait > 0:
                await asyncio.sleep(wait)
        frame = await self.audio_output_q.get()
        return frame

    async def run_input(self):
        try:
            if not self.audio_input_q:
                return
            while not self._track:
                await asyncio.sleep(0.2)
            while True:
                frame = await self._track.recv()
                await self.audio_input_q.put(frame)
        except Exception as e:
            print("Error in audio_frame_callback: ", e)
            raise asyncio.CancelledError

    def add_track(self, track):
        self._track = track
