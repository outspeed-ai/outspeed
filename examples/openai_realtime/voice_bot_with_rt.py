import logging
import os

from pydantic import BaseModel

import outspeed as sp

import aiohttp


def check_outspeed_version():
    import importlib.metadata

    from packaging import version

    required_version = "0.1.147"

    try:
        current_version = importlib.metadata.version("outspeed")
        if version.parse(current_version) < version.parse(required_version):
            raise ValueError(f"Outspeed version {current_version} is not greater than {required_version}.")
        else:
            print(f"Outspeed version {current_version} meets the requirement.")
    except importlib.metadata.PackageNotFoundError:
        raise ValueError("Outspeed package is not installed.")


check_outspeed_version()

"""
The @outspeed.App() decorator is used to wrap the VoiceBot class.
This tells the outspeed server which functions to run.
"""


@sp.App()
class VoiceBot:
    async def setup(self) -> None:
        # Initialize the AI services
        self.rt_node = sp.OpenAIRealtime()

    @sp.streaming_endpoint()
    async def run(self, audio_input_queue: sp.AudioStream, text_input_queue: sp.TextStream):
        # Set up the AI service pipeline
        audio_output_stream: sp.AudioStream
        audio_output_stream, text_output_stream = self.rt_node.run(text_input_queue, audio_input_queue)

        return audio_output_stream, text_output_stream

    async def teardown(self) -> None:
        # Close the OpenAI Realtime node when the application is shutting down
        await self.rt_node.close()


if __name__ == "__main__":
    # Start the VoiceBot when the script is run directly
    VoiceBot().start()
