import json
import pytest

from outspeed.plugins.openai_llm import OpenAILLM
from outspeed.streams import TextStream


@pytest.mark.asyncio
async def test_openai_llm_response(mock_openai_client):
    llm = OpenAILLM(stream=False, api_key="test")
    output_queue, chat_history_queue = llm.run(input_queue=TextStream())

    # Send a test message to the input queue
    await llm.input_queue.put("Test message")

    # Retrieve the response from the output queue
    response = await output_queue.get()

    assert response == "Hello, this is a mocked response."

    await llm.close()


@pytest.mark.asyncio
async def test_openai_llm_with_json_response(mock_openai_client):
    llm = OpenAILLM(stream=False, response_format={"type": "json"}, api_key="test")
    output_queue, _ = llm.run(input_queue=TextStream())

    await llm.input_queue.put("Test JSON response")
    response = await output_queue.get()

    response = json.loads(response)
    assert isinstance(response, dict)
    assert "message" in response
    assert response["message"] == "Hello, this is a mocked response."

    await llm.close()


@pytest.mark.asyncio
async def test_openai_llm_with_tool_choice_none(mock_openai_client):
    llm = OpenAILLM(stream=False, tool_choice="none", api_key="test")
    output_queue, _ = llm.run(input_queue=TextStream())

    await llm.input_queue.put("Test tool_choice none")
    response = await output_queue.get()

    assert response == "Hello, this is a mocked response."

    await llm.close()
