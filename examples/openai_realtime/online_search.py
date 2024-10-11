import logging

import outspeed as sp
from pydantic import BaseModel


def check_outspeed_version():
    import importlib.metadata

    from packaging import version

    required_version = "0.1.146"

    try:
        current_version = importlib.metadata.version("outspeed")
        if version.parse(current_version) < version.parse(required_version):
            raise ValueError(f"Outspeed version {current_version} is not greater than {required_version}.")
        else:
            print(f"Outspeed version {current_version} meets the requirement.")
    except importlib.metadata.PackageNotFoundError:
        raise ValueError("Outspeed package is not installed.")


check_outspeed_version()
# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)


class Query(BaseModel):
    query: str


class SearchResult(BaseModel):
    result: str


class SearchTool(sp.Tool):
    async def run(self, query: Query) -> SearchResult:
        return SearchResult(result="This is a test result")


@sp.App()
class VoiceBot:
    async def setup(self) -> None:
        # Initialize the AI services
        self.llm_node = sp.OpenAIRealtime(
            tools=[
                SearchTool(
                    name="search",
                    description="Search the web for information",
                    parameters_type=Query,
                    response_type=SearchResult,
                )
            ]
        )

    @sp.streaming_endpoint()
    async def run(self, audio_input_queue: sp.AudioStream, text_input_queue: sp.TextStream) -> sp.AudioStream:
        # vad_stream = self.vad_node.run(audio_input_queue.clone())
        # Set up the AI service pipeline
        audio_output_stream: sp.AudioStream
        audio_output_stream = self.llm_node.run(text_input_queue, audio_input_queue)

        # self.llm_node.set_interrupt_stream(vad_stream)
        # self.token_aggregator_node.set_interrupt_stream(vad_stream.clone())
        # self.tts_node.set_interrupt_stream(vad_stream.clone())

        return audio_output_stream

    async def teardown(self) -> None:
        """
        Clean up resources when the VoiceBot is shutting down.

        This method is called when the app stops or is shut down unexpectedly.
        It should be used to release resources and perform any necessary cleanup.
        """
        await self.llm_node.close()


if __name__ == "__main__":
    # Start the VoiceBot when the script is run directly
    VoiceBot().start()
