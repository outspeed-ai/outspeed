import outspeed as sp


@sp.App()
class CookingAssistant:
    async def setup(self):
        self.deepgram_node = sp.DeepgramSTT(sample_rate=8000)
        self.keyframe_node = sp.KeyFrameDetector(key_frame_threshold=0.8, key_frame_max_time=20)
        self.llm_node = sp.GeminiVision(
            system_prompt="You are a chef. Your job is to provide feedback and guide the user on how to cook a dish. First tell them the recipe and then guide them on how to cook it. Take into account previous responses for a smooth flow. Do not repeat yourself. Keep the output within 15 words.",
            temperature=0.9,
            store_image_history=False,
            model="gemini-1.5-flash",
            max_output_tokens=30,
        )
        self.token_aggregator_node = sp.TokenAggregator()
        self.tts_node = sp.CartesiaTTS(voice_id="5619d38c-cf51-4d8e-9575-48f61a280413")
        self.vad_node = sp.SileroVAD(sample_rate=8000, min_volume=0)

    @sp.streaming_endpoint()
    async def run(self, audio_input_stream: sp.AudioStream, video_input_stream: sp.VideoStream):
        deepgram_stream: sp.TextStream = self.deepgram_node.run(audio_input_stream)
        vad_stream: sp.VADStream = self.vad_node.run(audio_input_stream.clone())

        key_frame_stream: sp.VideoStream = self.keyframe_node.run(video_input_stream)

        llm_input_stream = sp.merge([deepgram_stream, key_frame_stream])

        llm_token_stream: sp.TextStream
        chat_history_stream: sp.TextStream
        llm_token_stream, chat_history_stream = self.llm_node.run(llm_input_stream)

        token_aggregator_stream: sp.TextStream = self.token_aggregator_node.run(llm_token_stream)

        tts_stream: sp.AudioStream = self.tts_node.run(token_aggregator_stream)

        self.llm_node.set_interrupt_stream(vad_stream)
        self.token_aggregator_node.set_interrupt_stream(vad_stream.clone())
        self.tts_node.set_interrupt_stream(vad_stream.clone())
        self.keyframe_node.set_interrupt_stream(vad_stream.clone())

        return tts_stream, chat_history_stream

    async def teardown(self):
        await self.deepgram_node.close()
        await self.openai_node.close()
        await self.token_aggregator_node.close()
        await self.tts_node.close()


if __name__ == "__main__":
    CookingAssistant().start()
