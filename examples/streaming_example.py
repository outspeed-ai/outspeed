import asyncio
import logging
from typing import Tuple

import outspeed as sp

logging.basicConfig(level=logging.INFO)

"""
Wrapping your class with @realtime.App() will tell the realtime server which functions to run.
"""


@sp.App()
class ReplayBot:
    async def setup(self):
        pass

    @sp.streaming_endpoint()
    async def run(self, audio_input_queue: sp.AudioStream, video_input_queue: sp.VideoStream):
        return (audio_input_queue, video_input_queue, None)

    async def teardown(self):
        pass


if __name__ == "__main__":
    ReplayBot().start()
