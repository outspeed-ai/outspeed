import asyncio
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)

import realtime
from realtime.plugins.eleven_labs_tts import ElevenLabsTTS
from realtime.plugins.token_aggregator import TokenAggregator
from realtime.plugins.deepgram_stt import DeepgramSTT
from realtime.plugins.gemini_vision import GeminiVision
from realtime.plugins.key_frame_detector import KeyFrameDetector
from realtime.streams import AudioStream, VideoStream, Stream, TextStream, ByteStream
from realtime.plugins.audio_convertor import AudioConverter


@realtime.App()
class PokerCommentator:
    async def setup(self):
        self.deepgram_node = DeepgramSTT(sample_rate=8000)
        self.keyframe_node = KeyFrameDetector(
            key_frame_threshold=0.2, key_frame_max_time=15
        )
        self.llm_node = GeminiVision(
            system_prompt="You are a poker commentator. Your job is to provide useful and deep insights on the strategy. Use exclamation points to show excitement. Do not mention the pot size. Make sure to read the cards correctly. If no cards are shown then just say something to pass time. Comment on the latest move. Do not state the obvious. Do not assume anything if no action is visible. Mention player names. Keep the response within 1-2 sentences. Do not mention all-in. Do not say tough spot or great spot. For example:\n\n 1. The table is silent, waiting for the next move as the dealer flips the turn card.\n2. Sara folds, deciding not to risk more chips on a weak hand.\n3. The dealer flips the turn card, and Alex bets 100 chips.",
            auto_respond=8,
            temperature=0.9,
            chat_history=True,
        )
        self.token_aggregator_node = TokenAggregator()
        self.tts_node = ElevenLabsTTS(
            optimize_streaming_latency=3,
            voice_id="iF9Lv1Pii7eCFPy5XZlZ",
        )
        self.audio_convertor_node = AudioConverter()

    @realtime.streaming_endpoint()
    async def run(
        self, audio_input_stream: AudioStream, video_input_stream: VideoStream
    ) -> Tuple[Stream, ...]:
        deepgram_stream: TextStream = await self.deepgram_node.run(audio_input_stream)

        key_frame_stream: VideoStream = await self.keyframe_node.run(video_input_stream)

        llm_token_stream: TextStream
        chat_history_stream: TextStream
        llm_token_stream, chat_history_stream = await self.llm_node.run(
            deepgram_stream, key_frame_stream
        )

        token_aggregator_stream: TextStream = await self.token_aggregator_node.run(
            llm_token_stream
        )

        tts_stream: ByteStream = await self.tts_node.run(token_aggregator_stream)

        output_video_stream = await video_input_stream.clone()
        audio_stream: AudioStream = await self.audio_convertor_node.run(tts_stream)

        return audio_stream, chat_history_stream, output_video_stream

    async def teardown(self):
        await self.deepgram_node.close()
        await self.keyframe_node.close()
        await self.llm_node.close()
        await self.token_aggregator_node.close()
        await self.tts_node.close()


if __name__ == "__main__":
    asyncio.run(PokerCommentator().run())
