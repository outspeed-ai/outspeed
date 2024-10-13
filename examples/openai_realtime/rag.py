import logging
import os

import nest_asyncio
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from pydantic import BaseModel

import outspeed as sp

nest_asyncio.apply()

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))


def check_outspeed_version():
    import importlib.metadata

    from packaging import version

    required_version = "0.1.149"

    try:
        current_version = importlib.metadata.version("outspeed")
        if version.parse(current_version) < version.parse(required_version):
            raise ValueError(f"Outspeed version {current_version} is not greater than {required_version}.")
        else:
            print(f"Outspeed version {current_version} meets the requirement.")
    except importlib.metadata.PackageNotFoundError:
        raise ValueError("Outspeed package is not installed.")


check_outspeed_version()


class Query(BaseModel):
    query_for_neural_search: str


class SearchResult(BaseModel):
    result: str


class SearchTool(sp.Tool):
    def __init__(
        self,
        name: str,
        description: str,
        parameters_type: type[Query],
        response_type: type[SearchResult],
        query_engine,
    ):
        super().__init__(name, description, parameters_type, response_type)
        self.query_engine = query_engine

    async def run(self, query: Query) -> SearchResult:
        logging.info(f"Searching for: {query.query_for_neural_search}")
        response = self.query_engine.query(query.query_for_neural_search)
        logging.info(f"RAG Response: {response}")
        return SearchResult(result=str(response))


@sp.App()
class VoiceBot:
    async def setup(self) -> None:
        # download and install dependencies
        documents = SimpleDirectoryReader(f"{PARENT_DIR}/data/").load_data()
        node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
        nodes = node_parser.get_nodes_from_documents(documents=documents)
        vector_index = VectorStoreIndex(nodes)
        self.query_engine = vector_index.as_query_engine(similarity_top_k=2)
        # Initialize the AI services
        self.llm_node = sp.OpenAIRealtime(
            tools=[
                SearchTool(
                    name="search",
                    description="Search the web for information",
                    parameters_type=Query,
                    response_type=SearchResult,
                    query_engine=self.query_engine,
                )
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
