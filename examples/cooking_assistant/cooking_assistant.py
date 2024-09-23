import asyncio
import logging
import os
from typing import Tuple

logging.basicConfig(level=logging.ERROR)

import realtime
from realtime.plugins.deepgram_stt import DeepgramSTT
from realtime.plugins.eleven_labs_tts import ElevenLabsTTS
from realtime.plugins.openai_vision import OpenAIVision
from realtime.plugins.token_aggregator import TokenAggregator
from realtime.streams import AudioStream, VideoStream, Stream, TextStream, ByteStream
from realtime.plugins.silero_vad import SileroVAD
from realtime.plugins.audio_convertor import AudioConverter


@realtime.App()
class CookingAssistant:
    async def setup(self):
        self.deepgram_node = DeepgramSTT(
            api_key=os.environ.get("DEEPGRAM_API_KEY"), sample_rate=8000
        )
        self.openai_node = OpenAIVision(
            api_key=os.environ.get("OPENAI_API_KEY"),
            system_prompt="You are a chef. Your job is to provide feedback and guide the user on how to cook a dish. First tell them the recipe and then guide them on how to cook it. Take into account previous responses for a smooth flow. Do not repeat yourself. Keep the output within 15 words.",
            auto_respond=10,
            wait_for_first_user_response=True,
        )
        self.token_aggregator_node = TokenAggregator()
        self.elevenlabs_node = ElevenLabsTTS(
            api_key=os.environ.get("ELEVEN_LABS_API_KEY")
        )
        self.silero_vad_node = SileroVAD()
        self.audio_convertor_node = AudioConverter()

    @realtime.streaming_endpoint()
    async def run(
        self, audio_input_stream: AudioStream, video_input_stream: VideoStream
    ) -> Tuple[Stream, ...]:
        audio_input_stream_copy = await audio_input_stream.clone()

        deepgram_stream: TextStream = await self.deepgram_node.run(audio_input_stream)
        silero_vad_stream: TextStream = await self.silero_vad_node.run(
            audio_input_stream_copy
        )
        openai_stream: TextStream
        openai_stream, chat_history = await self.openai_node.run(
            deepgram_stream, video_input_stream
        )
        token_aggregator_stream: TextStream = await self.token_aggregator_node.run(
            openai_stream
        )
        elevenlabs_stream: ByteStream = await self.elevenlabs_node.run(
            token_aggregator_stream
        )
        audio_stream: AudioStream = await self.audio_convertor_node.run(
            elevenlabs_stream
        )

        await self.openai_node.set_interrupt(silero_vad_stream)
        await self.elevenlabs_node.set_interrupt(await silero_vad_stream.clone())
        await self.token_aggregator_node.set_interrupt(await silero_vad_stream.clone())

        return audio_stream, await video_input_stream.clone(), chat_history

    async def teardown(self):
        await self.deepgram_node.close()
        await self.openai_node.close()
        await self.token_aggregator_node.close()
        await self.elevenlabs_node.close()
        await self.silero_vad_node.close()
        await self.audio_convertor_node.close()


if __name__ == "__main__":
    asyncio.run(CookingAssistant().run())
