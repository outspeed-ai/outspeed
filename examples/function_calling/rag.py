import logging
import os

import nest_asyncio
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from pydantic import BaseModel

import outspeed as sp

nest_asyncio.apply()

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))


class Query(BaseModel):
    query_for_neural_search: str


class RAGResult(BaseModel):
    result: str


class RAGTool(sp.Tool):
    name = "rag"
    description = "Search the knowledge base for information"
    parameters_type = Query
    response_type = RAGResult

    def __init__(self):
        super().__init__()
        documents = SimpleDirectoryReader(f"{PARENT_DIR}/data/").load_data()
        node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
        nodes = node_parser.get_nodes_from_documents(documents=documents)
        vector_index = VectorStoreIndex(nodes)
        self.query_engine = vector_index.as_query_engine(similarity_top_k=2)

    async def run(self, query: Query) -> RAGResult:
        logging.info(f"Searching for: {query.query_for_neural_search}")
        response = self.query_engine.query(query.query_for_neural_search)
        logging.info(f"RAG Response: {response}")
        return RAGResult(result=str(response))


@sp.App()
class VoiceBot:
    async def setup(self) -> None:
        # Initialize the AI services
        self.deepgram_node = sp.DeepgramSTT()
        self.llm_node = sp.OpenAILLM(
            tool_choice="required",
            tools=[
                RAGTool(),
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
