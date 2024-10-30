import logging
import os

from pydantic import BaseModel

import outspeed as sp

import aiohttp


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
        if not os.getenv("EXA_API_KEY"):
            raise ValueError("EXA_API_KEY is not set in the environment variables.")

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
        self.deepgram_node = sp.DeepgramSTT()
        self.llm_node = sp.OpenAILLM(
            tool_choice="required",
            tools=[
                SearchTool(),
            ],
        )
        self.token_aggregator_node = sp.TokenAggregator()
        self.tts_node = sp.CartesiaTTS(
            voice_id="95856005-0332-41b0-935f-352e296aa0df",
        )
        self.vad_node = sp.SileroVAD(min_volume=0)

    @sp.streaming_endpoint()
    async def run(self, audio_input_queue: sp.AudioStream, text_input_queue: sp.TextStream):
        # Set up the AI service pipeline
        deepgram_stream: sp.TextStream = self.deepgram_node.run(audio_input_queue)

        vad_stream: sp.VADStream = self.vad_node.run(audio_input_queue.clone())

        text_input_queue = sp.map(text_input_queue, lambda x: json.loads(x).get("content"))

        llm_input_queue: sp.TextStream = sp.merge(
            [deepgram_stream, text_input_queue],
        )

        llm_token_stream: sp.TextStream
        chat_history_stream: sp.TextStream
        llm_token_stream, chat_history_stream = self.llm_node.run(llm_input_queue)

        token_aggregator_stream: sp.TextStream = self.token_aggregator_node.run(llm_token_stream)
        tts_stream: sp.AudioStream = self.tts_node.run(token_aggregator_stream)

        self.llm_node.set_interrupt_stream(vad_stream)
        self.token_aggregator_node.set_interrupt_stream(vad_stream.clone())
        self.tts_node.set_interrupt_stream(vad_stream.clone())

        return tts_stream, chat_history_stream

    async def teardown(self) -> None:
        """
        Clean up resources when the VoiceBot is shutting down.
        """
        await self.deepgram_node.close()
        await self.llm_node.close()
        await self.token_aggregator_node.close()
        await self.tts_node.close()


if __name__ == "__main__":
    # Start the VoiceBot when the script is run directly
    VoiceBot().start()
