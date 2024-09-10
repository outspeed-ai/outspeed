import asyncio
import fractions
import logging

import av
import numpy as np

from realtime.plugins.base_plugin import Plugin
from realtime.streams import AudioStream, ByteStream, VADStream
from realtime.data import AudioData
from realtime.utils.vad import VADState

logger = logging.getLogger(__name__)


class AudioConverter(Plugin):
    def __init__(self, sample_rate=16000, channel_layout="mono", format="s16"):
        self.input_sample_rate = sample_rate
        self.input_channel_layout = channel_layout
        self.input_format = format
        self.output_queue = AudioStream()
        self.audio_samples = 0
        self.output_audio_sample_rate = 48000
        self.output_audio_chunk_size_seconds = 0.020
        self.output_audio_time_base = fractions.Fraction(1, self.output_audio_sample_rate)
        self.output_audio_resampler = av.AudioResampler(
            format="s16",
            layout="stereo",
            rate=self.output_audio_sample_rate,
            frame_size=int(self.output_audio_sample_rate * self.output_audio_chunk_size_seconds),
        )

    def run(self, input_queue: ByteStream) -> AudioStream:
        self.input_queue = input_queue
        self._task = asyncio.create_task(self.convert_bytes_to_frame())
        return self.output_queue

    async def convert_bytes_to_frame(self):
        audio_data = b""
        try:
            while True:
                chunk: AudioData = await self.input_queue.get()
                if chunk:
                    audio_data += chunk.get_bytes()
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
                    await self.output_queue.put(AudioData(nframe, sample_rate=self.output_audio_sample_rate))
        except Exception as e:
            logger.error(f"Error in AudioConverter: {e}")

    async def _interrupt(self):
        """
        Handle interruptions (e.g., when the user starts speaking).
        Cancels ongoing TTS generation and clears the output queue.
        """
        while True:
            vad_state: VADState = await self.interrupt_queue.get()
            if vad_state == VADState.SPEAKING and (not self.input_queue.empty() or not self.output_queue.empty()):
                if self._task:
                    logging.info("Cancelling AudioConverter task")
                    self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    logging.info("Done cancelling AudioConverter task")
                while not self.output_queue.empty():
                    self.output_queue.get_nowait()
                while not self.input_queue.empty():
                    self.input_queue.get_nowait()
                logging.info("Done cancelling AudioConverter")
                self._task = asyncio.create_task(self.convert_bytes_to_frame())

    def set_interrupt_stream(self, interrupt_stream: VADStream):
        """
        Set up the interrupt queue and start the interrupt handling task.

        Args:
            interrupt_stream (VADStream): The stream for receiving interrupt signals.
        """
        if isinstance(interrupt_stream, VADStream):
            self.interrupt_queue = interrupt_stream
        else:
            raise ValueError("Interrupt stream must be a VADStream")
        self._interrupt_task = asyncio.create_task(self._interrupt())
