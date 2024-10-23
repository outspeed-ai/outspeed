import logging
import os

from pydantic import BaseModel

import outspeed as sp

import aiohttp


def check_outspeed_version():
    import importlib.metadata

    from packaging import version

    required_version = "0.1.147"

    try:
        current_version = importlib.metadata.version("outspeed")
        if version.parse(current_version) < version.parse(required_version):
            raise ValueError(f"Outspeed version {current_version} is not greater than {required_version}.")
        else:
            print(f"Outspeed version {current_version} meets the requirement.")
    except importlib.metadata.PackageNotFoundError:
        raise ValueError("Outspeed package is not installed.")


check_outspeed_version()

"""
The @outspeed.App() decorator is used to wrap the VoiceBot class.
This tells the outspeed server which functions to run.
"""


class Query(BaseModel):
    query: str


class SearchResult(BaseModel):
    result: str


class SearchTool(sp.Tool):
    name = "search"
    description = "Search the web for information"
    parameters_type = Query
    response_type = SearchResult

    async def run(self, query: Query) -> SearchResult:
        url = "https://api.exa.ai/search"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": os.getenv("EXA_API_KEY"),  # Ensure EXA_API_KEY is set in your environment
        }
        payload = {
            "query": query.query,
            "type": "neural",
            "useAutoprompt": True,
            "numResults": 1,
            "contents": {"text": True},
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    data = await response.json()
                    # Process the response data as needed
                    return SearchResult(result=str(data.get("results", [{}])[0].get("text", "")))
            except aiohttp.ClientError as e:
                logging.error(f"HTTP request failed: {e}")
                return SearchResult(result="An error occurred while processing the search request.")


@sp.App()
class VoiceBot:
    async def setup(self) -> None:
        # Initialize the AI services
        self.llm_node = sp.OpenAIRealtime(
            tools=[
                SearchTool(),
            ]
        )

    @sp.streaming_endpoint()
    async def run(self, audio_input_queue: sp.AudioStream, text_input_queue: sp.TextStream):
        # Set up the AI service pipeline
        audio_output_stream: sp.AudioStream
        audio_output_stream, text_output_stream = self.llm_node.run(text_input_queue, audio_input_queue)

        return audio_output_stream, text_output_stream

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
