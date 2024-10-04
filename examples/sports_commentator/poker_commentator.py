import asyncio
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)

import outspeed as sp


@sp.App()
class PokerCommentator:
    async def setup(self):
        self.deepgram_node = sp.DeepgramSTT(sample_rate=8000)
        self.keyframe_node = sp.KeyFrameDetector(key_frame_threshold=0.2, key_frame_max_time=15)
        self.llm_node = sp.GeminiVision(
            system_prompt="You are a poker commentator. Your job is to provide useful and deep insights on the strategy. Use exclamation points to show excitement. Do not mention the pot size. Make sure to read the cards correctly. If no cards are shown then just say something to pass time. Comment on the latest move. Do not state the obvious. Do not assume anything if no action is visible. Mention player names. Keep the response within 1-2 sentences. Do not mention all-in. Do not say tough spot or great spot. For example:\n\n 1. The table is silent, waiting for the next move as the dealer flips the turn card.\n2. Sara folds, deciding not to risk more chips on a weak hand.\n3. The dealer flips the turn card, and Alex bets 100 chips.",
            auto_respond=8,
            temperature=0.9,
            chat_history=True,
        )
        self.token_aggregator_node = sp.TokenAggregator()
        self.tts_node = sp.ElevenLabsTTS(
            optimize_streaming_latency=3,
            voice_id="iF9Lv1Pii7eCFPy5XZlZ",
        )

    @sp.streaming_endpoint()
    async def run(
        self, audio_input_stream: sp.AudioStream, video_input_stream: sp.VideoStream
    ) -> Tuple[sp.Stream, ...]:
        deepgram_stream: sp.TextStream = self.deepgram_node.run(audio_input_stream)

        key_frame_stream: sp.VideoStream = self.keyframe_node.run(video_input_stream)

        llm_token_stream: sp.TextStream
        chat_history_stream: sp.TextStream
        llm_token_stream, chat_history_stream = self.llm_node.run(deepgram_stream, key_frame_stream)

        token_aggregator_stream: sp.TextStream = self.token_aggregator_node.run(llm_token_stream)

        tts_stream: sp.ByteStream = self.tts_node.run(token_aggregator_stream)

        audio_stream: sp.AudioStream = self.audio_convertor_node.run(tts_stream)

        return audio_stream, chat_history_stream

    async def teardown(self):
        self.deepgram_node.close()
        self.keyframe_node.close()
        self.llm_node.close()
        self.token_aggregator_node.close()
        self.tts_node.close()


if __name__ == "__main__":
    PokerCommentator().start()
