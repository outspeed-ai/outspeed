import asyncio
import fractions
import logging

import av
import numpy as np

from realtime.plugins.base_plugin import Plugin
from realtime.streams import AudioStream, ByteStream

logger = logging.getLogger(__name__)


class AudioConverter(Plugin):
    def __init__(self, sample_rate=16000, channel_layout="mono", format="s16"):
        self.input_sample_rate = sample_rate
        self.input_channel_layout = channel_layout
        self.input_format = format
        self.output_queue = AudioStream()
        self.audio_samples = 0
        self.output_audio_sample_rate = 48000
        # audio_samples = 0
        self.output_audio_time_base = fractions.Fraction(1, self.output_audio_sample_rate)
        self.output_audio_resampler = av.AudioResampler(
            format="s16",
            layout="stereo",
            rate=self.output_audio_sample_rate,
            frame_size=int(self.output_audio_sample_rate * 0.020),
        )

    async def run(self, input_queue: ByteStream) -> AudioStream:
        self.input_queue = input_queue
        self._task = asyncio.create_task(self.convert_bytes_to_frame())
        return self.output_queue

    async def convert_bytes_to_frame(self):
        audio_data = b""
        while True:
            chunk = await self.input_queue.get()
            audio_data += chunk
            if len(audio_data) < 2:
                continue
            if len(audio_data) % 2 != 0:
                array = np.frombuffer(audio_data[:-1], dtype=np.int16).reshape(1, -1)  # mono has 1 channel
                audio_data = audio_data[-1:]
            else:
                array = np.frombuffer(audio_data, dtype=np.int16).reshape(1, -1)  # mono has 1 channel
                audio_data = b""

            # Create a new AudioFrame from the NumPy array
            frame = av.AudioFrame.from_ndarray(array, format=self.input_format, layout=self.input_channel_layout)
            frame.sample_rate = self.input_sample_rate

            for nframe in self.output_audio_resampler.resample(frame):
                # fix timestamps
                nframe.pts = self.audio_samples
                nframe.time_base = self.output_audio_time_base
                self.audio_samples += nframe.samples
                self.output_queue.put_nowait(nframe)
