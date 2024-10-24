import pytest

from outspeed.plugins.openai_llm import OpenAILLM
from outspeed.streams import TextStream


@pytest.mark.asyncio
async def test_openai_llm_response(mock_openai_client):
    llm = OpenAILLM(stream=False)
    output_queue, chat_history_queue = llm.run(input_queue=TextStream())

    # Send a test message to the input queue
    await llm.input_queue.put("Test message")

    # Retrieve the response from the output queue
    response = await output_queue.get()

    assert response == "Hello, this is a mocked response."

    await llm.close()
