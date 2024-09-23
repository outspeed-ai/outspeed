import asyncio
import fractions
import logging
import time

from aiortc import MediaStreamTrack
from av import AudioResampler

from realtime.data import AudioData, SessionData


class AudioRTCDriver(MediaStreamTrack):
    kind = "audio"

    def __init__(
        self,
        audio_input_q,
        audio_output_q,
        output_audio_sample_rate=48000,
        output_audio_layout="stereo",
        output_audio_format="s16",
    ):
        super().__init__()
        self.audio_input_q = audio_input_q
        self.audio_output_q = audio_output_q
        self.audio_data_q = asyncio.Queue()
        self.audio_samples = 0
        self._start = None
        self._track = None
        self.output_audio_sample_rate = output_audio_sample_rate
        self.output_audio_layout = output_audio_layout
        self.output_audio_format = output_audio_format
        self.output_audio_time_base = fractions.Fraction(1, self.output_audio_sample_rate)
        self.output_audio_chunk_size_seconds = 0.020
        self.output_audio_resampler = AudioResampler(
            format=self.output_audio_format,
            layout=self.output_audio_layout,
            rate=self.output_audio_sample_rate,
            frame_size=int(self.output_audio_sample_rate * self.output_audio_chunk_size_seconds),
        )

    async def recv(self):
        frame = await self.audio_data_q.get()
        data_time = frame.samples / frame.sample_rate
        if self._start is None:
            self._start = time.time() + data_time
        else:
            wait = self._start - time.time() - data_time
            if wait > 0:
                await asyncio.sleep(wait)
            self._start = max(self._start, time.time()) + data_time
        return frame

    async def run_input(self):
        try:
            if not self.audio_input_q:
                return
            while not self._track:
                await asyncio.sleep(0.2)
            await self.audio_input_q.put(SessionData())
            while True:
                frame = await self._track.recv()
                await self.audio_input_q.put(AudioData(frame))
        except Exception as e:
            logging.error(f"Error in audio_frame_callback: {str(e)}")
            raise asyncio.CancelledError

    async def run_output(self):
        try:
            while True:
                while self.audio_data_q.qsize() > 2:
                    await asyncio.sleep(0.01)
                audio_data: AudioData = await self.audio_output_q.get()
                if audio_data is None or not isinstance(audio_data, AudioData):
                    continue
                self.audio_samples = max(self.audio_samples, audio_data.get_pts())
                for nframe in self.output_audio_resampler.resample(audio_data.get_frame()):
                    # fix timestamps
                    nframe.pts = self.audio_samples
                    nframe.time_base = self.output_audio_time_base
                    self.audio_samples += nframe.samples
                    self.audio_data_q.put_nowait(nframe)
        except Exception as e:
            logging.error(f"Error in audio_frame_callback: {str(e)}")
            raise asyncio.CancelledError

    async def run(self):
        await asyncio.gather(self.run_input(), self.run_output())

    def add_track(self, track):
        self._track = track
