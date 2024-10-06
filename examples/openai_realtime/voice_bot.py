import json
import logging

import outspeed as sp
from outspeed.plugins.openai_realtime.openai_realtime_basic import OpenAIRealtimeBasic

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)

"""
The @outspeed.App() decorator is used to wrap the VoiceBot class.
This tells the outspeed server which functions to run.
"""


@sp.App()
class VoiceBot:
    """
    VoiceBot class represents a voice-based AI assistant.

    This class handles the setup, running, and teardown of various AI services
    used to process audio input, generate responses, and convert text to speech.
    """

    async def setup(self) -> None:
        """
        Initialize the VoiceBot.

        This method is called when the app starts. It should be used to set up
        services, load models, and perform any necessary initialization.
        """
        pass

    @sp.streaming_endpoint()
    async def run(self, audio_input_queue: sp.AudioStream, text_input_queue: sp.TextStream) -> sp.AudioStream:
        """
        Handle the main processing loop for the VoiceBot.

        This method is called for each new connection request. It sets up and
        runs the various AI services in a pipeline to process audio input and
        generate audio output.

        Args:
            audio_input_queue (sp.AudioStream): The input stream of audio data.

        Returns:
            sp.AudioStream: The output stream of generated audio responses.
        """
        # Initialize the AI services
        self.llm_node = OpenAIRealtimeBasic()
        self.vad_node = sp.SileroVAD(sample_rate=8000, min_volume=0)

        # vad_stream = self.vad_node.run(audio_input_queue.clone())
        # Set up the AI service pipeline
        audio_output_stream: sp.AudioStream
        audio_output_stream = self.llm_node.run(text_input_queue, audio_input_queue)

        # self.llm_node.set_interrupt_stream(vad_stream)
        # self.token_aggregator_node.set_interrupt_stream(vad_stream.clone())
        # self.tts_node.set_interrupt_stream(vad_stream.clone())

        return audio_output_stream

    async def teardown(self) -> None:
        """
        Clean up resources when the VoiceBot is shutting down.

        This method is called when the app stops or is shut down unexpectedly.
        It should be used to release resources and perform any necessary cleanup.
        """
        await self.llm_node.close()
        await self.vad_node.close()


if __name__ == "__main__":
    # Start the VoiceBot when the script is run directly
    VoiceBot().start()
