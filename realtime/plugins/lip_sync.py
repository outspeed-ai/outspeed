import asyncio
import json
import logging
import os
import subprocess
import time
import wave

import numpy as np

from realtime.plugins.base_plugin import Plugin
from realtime.streams import ByteStream, TextStream
from elevenlabs import save

logger = logging.getLogger(__name__)


class LipSync(Plugin):
    def __init__(
        self,
        channels: int = 1,
        sample_width: int = 2,
        sample_rate: int = 16000,
        rhubarb_path: str | None = None,
    ):
        self._channels = channels
        self._sample_width = sample_width
        self._sample_rate = sample_rate
        self.output_queue = TextStream()
        self.rhubarb_path = rhubarb_path

    async def run(self, input_queue: ByteStream) -> TextStream:
        self.input_queue = input_queue
        self._task = asyncio.create_task(self.lip_sync())
        return self.output_queue

    async def lip_sync(self):
        try:
            message = 1
            if not os.path.exists("audios"):
                os.makedirs("audios")
            while True:
                audio_frame = await self.input_queue.get()
                start_time = time.time()
                self.write_to_wav(audio_frame, f"audios/message_{message}.wav")
                logger.debug(f"Wrote audio frame to file in {time.time() - start_time} seconds")
                start_time = time.time()
                cmd = " ".join(
                    [
                        "/Users/janakagrawal/Documents/GitHub/realtime-examples/3d_avatar_chatbot/backend/rhubarb/rhubarb",
                        "-f",
                        "json",
                        "-o",
                        f"audios/message_{message}.json",
                        f"audios/message_{message}.wav",
                        "-r phonetic",
                    ],
                )
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await proc.communicate()  # Read the output and error streams
                logger.debug(stdout.decode())  # Decode and print the stdout
                logger.debug(stderr.decode())  # Decode and print the stderr
                await proc.wait()
                logger.info(f"Rhubarb took {time.time() - start_time} seconds to process the file")
                start_time = time.time()
                with open(f"audios/message_{message}.json", "r") as file:
                    data = json.load(file)
                logger.debug(f"Wrote data to queue in {time.time() - start_time} seconds")
                await self.output_queue.put(data)
                message += 1

        except BaseException:
            # This is triggered by an empty audio buffer
            return False

    def write_to_wav(self, byte_data, file_name="output.wav"):
        print("Writing to wav", file_name)
        with wave.open(file_name, "wb") as wav_file:
            wav_file.setnchannels(self._channels)  # Mono
            wav_file.setsampwidth(self._sample_width)  # Sample width in bytes
            wav_file.setframerate(self._sample_rate)  # Sample rate in Hz
            wav_file.writeframes(byte_data)
