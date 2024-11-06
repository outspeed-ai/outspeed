import json

import outspeed as sp


@sp.App()
class VoiceBot:
    """
    This class handles the setup, running, and teardown of various AI services
    used to process audio input, generate responses, and convert text to speech.
    """

    async def setup(self) -> None:
        """
        This method is called when the app starts. It should be used to set up services, load models, and perform any necessary initialization.
        """
        self.deepgram_node = sp.DeepgramSTT()
        self.llm_node = sp.GroqLLM(
            system_prompt="You are a helpful assistant. Keep your answers very short. No special characters in responses.",
        )
        self.token_aggregator_node = sp.TokenAggregator()
        self.tts_node = sp.CartesiaTTS(
            voice_id="95856005-0332-41b0-935f-352e296aa0df",
            volume=0.7,
        )

    @sp.streaming_endpoint()
    async def run(self, audio_input_stream: sp.AudioStream, text_input_stream: sp.TextStream):
        """
        It sets up and runs the various AI services in a pipeline to process audio input and generate audio output.
        """
        deepgram_stream: sp.TextStream = self.deepgram_node.run(audio_input_stream)

        text_input_stream = sp.map(text_input_stream, lambda x: json.loads(x).get("content"))

        llm_input_stream: sp.TextStream = sp.merge(
            [deepgram_stream, text_input_stream],
        )

        llm_token_stream: sp.TextStream
        chat_history_stream: sp.TextStream
        llm_token_stream, chat_history_stream = self.llm_node.run(llm_input_stream)

        token_aggregator_stream: sp.TextStream = self.token_aggregator_node.run(llm_token_stream)
        tts_stream: sp.AudioStream = self.tts_node.run(token_aggregator_stream)

        chat_history_stream = sp.filter(chat_history_stream, lambda x: json.loads(x).get("role") != "user")

        return tts_stream, chat_history_stream

    async def teardown(self) -> None:
        """
        Clean up resources when the VoiceBot is shutting down.
        """
        await self.deepgram_node.close()
        await self.llm_node.close()
        await self.token_aggregator_node.close()
        await self.tts_node.close()


if __name__ == "__main__":
    # Start the VoiceBot when the script is run directly
    VoiceBot().start()
