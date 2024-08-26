import asyncio
import json
import logging

import realtime
from realtime.ops.map import map
from realtime.ops.merge import merge
from realtime.plugins.azure_tts import AzureTTS
from realtime.plugins.deepgram_stt import DeepgramSTT
from realtime.plugins.fireworks_llm import FireworksLLM
from realtime.server import RealtimeServer
from realtime.streams import AudioStream, TextStream

logging.basicConfig(level=logging.INFO)


@realtime.App()
class ReplayBot:
    """
    A bot that uses WebSocket to interact with clients, processing audio and text data.

    Methods:
        setup(): Prepares any necessary configurations.
        run(ws: WebSocket): Handles the WebSocket connection and processes data.
        teardown(): Cleans up any resources or configurations.
    """

    async def setup(self):
        pass

    @realtime.websocket()
    async def run(audio_input_stream: AudioStream, message_stream: TextStream):
        deepgram_node = DeepgramSTT(sample_rate=audio_input_stream.sample_rate)
        llm_node = FireworksLLM(
            system_prompt="You are a virtual girlfriend.\
            You will always reply with a JSON object.\
            Each message has a text, facialExpression, and animation property.\
            The text property is a short response to the user (no emoji).\
            The different facial expressions are: smile, sad, angry, and default.\
            The different animations are: Talking_0, Talking_1, Talking_2, Crying, Laughing, Rumba, Idle, Terrified, and Angry.",
            temperature=0.9,
            response_format={"type": "json_object"},
            stream=False,
            model="accounts/fireworks/models/llama-v3-70b-instruct",
        )
        tts_node = AzureTTS(stream=False)

        deepgram_stream = await deepgram_node.run(audio_input_stream)
        deepgram_stream = merge([deepgram_stream, message_stream])

        llm_token_stream, chat_history_stream = await llm_node.run(deepgram_stream)

        json_text_stream = map(await llm_token_stream.clone(), lambda x: json.loads(x).get("text"))

        tts_stream, viseme_stream = await tts_node.run(json_text_stream)

        llm_with_viseme_stream = merge([llm_token_stream, viseme_stream])

        return tts_stream, llm_with_viseme_stream

    async def teardown(self):
        pass


if __name__ == "__main__":
    v = ReplayBot()
    asyncio.run(RealtimeServer().start())
