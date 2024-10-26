import json
import pytest

from outspeed.plugins.openai_llm import OpenAILLM
from outspeed.streams import TextStream


@pytest.mark.asyncio
async def test_openai_llm_response(load_env_file):
    llm = OpenAILLM(stream=False)
    llm_input_queue = TextStream()
    output_queue, chat_history_queue = llm.run(input_queue=llm_input_queue)

    # Send a test message to the input queue
    await llm_input_queue.put("Test message")

    # Retrieve the response from the output queue
    response = await output_queue.get()

    assert isinstance(response, str)

    await llm.close()
