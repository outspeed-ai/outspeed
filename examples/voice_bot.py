import json
import logging

import realtime as rt

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)

"""
The @realtime.App() decorator is used to wrap the VoiceBot class.
This tells the realtime server which functions to run.
"""


@rt.App()
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

    @rt.streaming_endpoint()
    async def run(self, audio_input_queue: rt.AudioStream, text_input_queue: rt.TextStream) -> rt.AudioStream:
        """
        Handle the main processing loop for the VoiceBot.

        This method is called for each new connection request. It sets up and
        runs the various AI services in a pipeline to process audio input and
        generate audio output.

        Args:
            audio_input_queue (rt.AudioStream): The input stream of audio data.

        Returns:
            rt.AudioStream: The output stream of generated audio responses.
        """
        # Initialize the AI services
        self.deepgram_node = rt.DeepgramSTT(sample_rate=8000)
        self.llm_node = rt.FireworksLLM(
            system_prompt="You are a helpful assistant. Keep your answers very short. No special characters in responses.",
        )
        self.token_aggregator_node = rt.TokenAggregator()
        self.tts_node = rt.CartesiaTTS(
            voice_id="95856005-0332-41b0-935f-352e296aa0df",
        )

        # Set up the AI service pipeline
        deepgram_stream: rt.TextStream = self.deepgram_node.run(audio_input_queue)

        text_input_queue = rt.map(text_input_queue, lambda x: json.loads(x).get("content"))

        llm_input_queue: rt.TextStream = rt.merge(
            [deepgram_stream, text_input_queue],
        )

        llm_token_stream: rt.TextStream
        chat_history_stream: rt.TextStream
        llm_token_stream, chat_history_stream = self.llm_node.run(llm_input_queue)

        token_aggregator_stream: rt.TextStream = self.token_aggregator_node.run(llm_token_stream)
        tts_stream: rt.AudioStream = self.tts_node.run(token_aggregator_stream)

        return tts_stream

    async def teardown(self) -> None:
        """
        Clean up resources when the VoiceBot is shutting down.

        This method is called when the app stops or is shut down unexpectedly.
        It should be used to release resources and perform any necessary cleanup.
        """
        await self.deepgram_node.close()
        await self.llm_node.close()
        await self.token_aggregator_node.close()
        await self.tts_node.close()


if __name__ == "__main__":
    # Start the VoiceBot when the script is run directly
    VoiceBot().start()
