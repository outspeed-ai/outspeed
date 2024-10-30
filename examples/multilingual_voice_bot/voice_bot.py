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
        self.transcriber_node = sp.AzureTranscriber(languages=["en-US", "fr-FR"])
        self.llm_node = sp.OpenAILLM(
            system_prompt="You are a helpful assistant. Keep your answers very short. No special characters in responses. You can speak English and French. Reply in the same language as the user's response.",
            model="gpt-4o-mini",
        )
        self.token_aggregator_node = sp.TokenAggregator()
        self.tts_node = sp.ElevenLabsTTS(
            voice_id="4DHVFVPkvJPP4FNkikun",
            model="eleven_turbo_v2_5",
            volume=0.7,
        )
        self.vad_node = sp.SileroVAD(min_volume=0)

    @sp.streaming_endpoint()
    async def run(self, audio_input_queue: sp.AudioStream, text_input_queue: sp.TextStream) -> sp.AudioStream:
        """
        It sets up and runs the various AI services in a pipeline to process audio input and generate audio output.
        """
        transcriber_stream: sp.TextStream = self.transcriber_node.run(audio_input_queue)

        vad_stream: sp.VADStream = self.vad_node.run(audio_input_queue.clone())

        text_input_queue = sp.map(text_input_queue, lambda x: json.loads(x).get("content"))

        llm_input_queue = sp.merge(
            [transcriber_stream, text_input_queue],
        )

        llm_token_stream: sp.TextStream
        chat_history_stream: sp.TextStream
        llm_token_stream, chat_history_stream = self.llm_node.run(llm_input_queue)

        token_aggregator_stream: sp.TextStream = self.token_aggregator_node.run(llm_token_stream)
        tts_stream: sp.AudioStream = self.tts_node.run(token_aggregator_stream)

        self.llm_node.set_interrupt_stream(vad_stream)
        self.token_aggregator_node.set_interrupt_stream(vad_stream.clone())
        self.tts_node.set_interrupt_stream(vad_stream.clone())

        return tts_stream

    async def teardown(self) -> None:
        """
        Clean up resources when the VoiceBot is shutting down.
        """
        await self.transcriber_node.close()
        await self.llm_node.close()
        await self.token_aggregator_node.close()
        await self.tts_node.close()


if __name__ == "__main__":
    # Start the VoiceBot when the script is run directly
    VoiceBot().start()
