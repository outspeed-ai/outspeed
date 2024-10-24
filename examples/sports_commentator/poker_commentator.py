import outspeed as sp


def check_outspeed_version():
    import importlib.metadata

    from packaging import version

    required_version = "0.2.0"

    try:
        current_version = importlib.metadata.version("outspeed")
        if version.parse(current_version) < version.parse(required_version):
            raise ValueError(f"Outspeed version {current_version} is not greater than {required_version}.")
        else:
            print(f"Outspeed version {current_version} meets the requirement.")
    except importlib.metadata.PackageNotFoundError:
        raise ValueError("Outspeed package is not installed.")


check_outspeed_version()


@sp.App()
class PokerCommentator:
    async def setup(self):
        self.deepgram_node = sp.DeepgramSTT()
        self.keyframe_node = sp.KeyFrameDetector(key_frame_threshold=0.8, key_frame_max_time=15)
        self.llm_node = sp.OpenAIVision(
            system_prompt="You are a poker commentator. Use exclamation points to show excitement. Keep it high level. Keep the response short and concise. No special characters.",
            temperature=0.9,
            store_image_history=False,
            model="gpt-4o-mini",
            max_output_tokens=50,
        )
        self.token_aggregator_node = sp.TokenAggregator()
        self.tts_node = sp.CartesiaTTS(voice_id="5619d38c-cf51-4d8e-9575-48f61a280413")
        self.vad_node = sp.SileroVAD(min_volume=0, min_speech_duration_seconds=0.2)

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

        video_output_stream: sp.VideoStream = video_input_stream.clone()

        self.llm_node.set_interrupt_stream(vad_stream)
        self.token_aggregator_node.set_interrupt_stream(vad_stream.clone())
        self.tts_node.set_interrupt_stream(vad_stream.clone())
        self.keyframe_node.set_interrupt_stream(vad_stream.clone())

        return tts_stream, chat_history_stream, video_output_stream

    async def teardown(self):
        await self.deepgram_node.close()
        await self.keyframe_node.close()
        await self.llm_node.close()
        await self.token_aggregator_node.close()
        await self.tts_node.close()


if __name__ == "__main__":
    PokerCommentator().start()
