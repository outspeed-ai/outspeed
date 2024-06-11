import asyncio
import os
import threading

import numpy as np
import torch

# Brute force torchaudio to use ffmpeg 7, otherwise it will use ffmpeg 6 which causes issues on mac
os.environ["TORIO_USE_FFMPEG_VERSION"] = "7"
# Although torchaudio is not utilized directly in this context, it is necessary to attempt its import as it is a dependency for Silero.
import torchaudio  # noqa: F401

from realtime.plugins.base_plugin import Plugin
from realtime.utils.cloneable_queue import CloneableQueue


class SileroVAD(Plugin):
    def __init__(
        self,
        buffer_duration: float = 0.1,
        audio_sample_rate: int = 8000,
        sensitivity_threshold: float = 0.91,
    ):
        if audio_sample_rate not in [8000, 16000]:
            raise ValueError("Silero VAD only supports 8KHz and 16KHz sample rates")

        torch.set_num_threads(1)

        (self.model, self.utils) = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
        )

        self.buffer_duration = buffer_duration
        self.audio_sample_rate = audio_sample_rate
        self.buffer_samples = int(self.audio_sample_rate * self.buffer_duration)

        self.sensitivity_threshold = sensitivity_threshold

        self._loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self.output_queue = CloneableQueue()
        self.user_speaking = False

    async def run(self, input_queue: CloneableQueue) -> CloneableQueue:
        self.input_queue = input_queue
        self._vad_thread = threading.Thread(target=self.execute_vad, daemon=True)
        self._vad_thread.start()
        return self.output_queue

    def execute_vad(self):
        try:
            buffer = None
            while True:
                future = asyncio.run_coroutine_threadsafe(self.input_queue.get(), self._loop)
                audio_frame = future.result()
                audio_data_int16 = audio_frame.to_ndarray()
                audio_data_float32 = self.convert_int_to_float(audio_data_int16)
                if buffer is None:
                    buffer = audio_data_float32
                else:
                    buffer = np.concatenate((buffer, audio_data_float32))

                if buffer.shape[0] > self.buffer_samples:
                    confidence_level = self.model(
                        torch.from_numpy(buffer[: self.buffer_samples]), self.audio_sample_rate
                    ).item()
                    buffer = buffer[-self.buffer_samples :]

                    is_speaking = confidence_level > self.sensitivity_threshold
                    if not is_speaking:
                        self.user_speaking = False
                    elif self._loop and is_speaking and not self.user_speaking:
                        print("silero", is_speaking, confidence_level)
                        self.user_speaking = True
                        asyncio.run_coroutine_threadsafe(self.output_queue.put(is_speaking), self._loop)
        except BaseException:
            # This is triggered by an empty audio buffer
            return False

    def convert_int_to_float(self, sound):
        try:
            maximum_absolute_value = np.abs(sound).max()
            sound = sound.astype("float32")
            if maximum_absolute_value > 0:
                sound *= 1 / 32768
            sound = sound.squeeze()  # Adjust based on specific needs
            return sound
        except ValueError:
            return sound
