import json
import logging

import realtime as rt

logging.basicConfig(level=logging.INFO)


@rt.App()
class WebsocketVoiceBot:
    """
    A bot that uses WebSocket to interact with clients, processing audio and text data.

    Methods:
        setup(): Prepares any necessary configurations.
        run(ws: WebSocket): Handles the WebSocket connection and processes data.
        teardown(): Cleans up any resources or configurations.
    """

    async def setup(self):
        self.test_str = "hello"
        self.deepgram_node = rt.DeepgramSTT(sample_rate=48000)
        self.llm_node = rt.FireworksLLM(
            system_prompt="You are a helpful assistant who answers questions about Outspeed. Outspeed builds tooling and infrastructure for Realtime AI applications.",
            temperature=0.9
        )
        self.token_aggregator_node = rt.TokenAggregator()
        self.tts_node = rt.ElevenLabsTTS(stream=True)

    @rt.websocket()
    async def run(self, audio_input_stream: rt.AudioStream, text_input_stream: rt.TextStream) -> rt.AudioStream:

        deepgram_stream = self.deepgram_node.run(audio_input_stream)

        llm_input_queue: rt.TextStream = rt.merge(
            [deepgram_stream, text_input_stream],
        )

        llm_token_stream, _ = self.llm_node.run(llm_input_queue)
        token_aggregator_stream: rt.TextStream = self.token_aggregator_node.run(llm_token_stream)

        tts_stream = self.tts_node.run(token_aggregator_stream)
        return tts_stream

    async def teardown(self):
        pass


if __name__ == "__main__":
    v = WebsocketVoiceBot()
    v.start()
