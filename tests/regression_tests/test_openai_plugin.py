import json
from pydantic import BaseModel
import pytest

from outspeed.plugins.openai_llm import OpenAILLM
from outspeed.streams import TextStream
from outspeed.tool import Tool


@pytest.fixture
def mock_tool():
    class MockToolParameters(BaseModel):
        message: str

    class MockToolResponse(BaseModel):
        response: str

    class MockTool(Tool):
        name = "mock_tool"
        description = "Mock tool"
        parameters_type = MockToolParameters
        response_type = MockToolResponse

        async def run(self, input_parameters: MockToolParameters):
            return MockToolResponse(response=input_parameters.message)

    return MockTool()


@pytest.mark.asyncio
async def test_openai_llm_response_with_streaming_disabled(load_env_file):
    llm = OpenAILLM(stream=False)
    llm_input_queue = TextStream()
    output_queue, chat_history_queue = llm.run(input_queue=llm_input_queue)

    # Send a test message to the input queue
    await llm_input_queue.put("Test message")

    # Retrieve the response from the output queue
    response = await output_queue.get()

    assert isinstance(response, str)

    chat_history_response = await chat_history_queue.get()

    assert isinstance(chat_history_response, str)

    chat_history_response_json = json.loads(chat_history_response)
    assert chat_history_response_json["role"] == "user"
    assert chat_history_response_json["content"] == "Test message"

    chat_history_response = await chat_history_queue.get()

    assert isinstance(chat_history_response, str)

    chat_history_response_json = json.loads(chat_history_response)
    assert chat_history_response_json["role"] == "assistant"
    assert chat_history_response_json["content"] == response

    await llm.close()


@pytest.mark.asyncio
async def test_openai_llm_with_custom_model(load_env_file):
    llm = OpenAILLM(model="gpt-3.5-turbo", stream=False)
    llm_input_queue = TextStream()
    output_queue, chat_history_queue = llm.run(input_queue=llm_input_queue)

    await llm_input_queue.put("Test message with custom model")
    response = await output_queue.get()

    assert isinstance(response, str)

    chat_history_response = await chat_history_queue.get()

    assert isinstance(chat_history_response, str)

    chat_history_response_json = json.loads(chat_history_response)
    assert chat_history_response_json["role"] == "user"
    assert chat_history_response_json["content"] == "Test message with custom model"

    chat_history_response = await chat_history_queue.get()

    assert isinstance(chat_history_response, str)

    chat_history_response_json = json.loads(chat_history_response)
    assert chat_history_response_json["role"] == "assistant"
    assert chat_history_response_json["content"] == response
    await llm.close()


@pytest.mark.asyncio
async def test_openai_llm_with_system_prompt(load_env_file):
    system_prompt = "You are a helpful assistant."
    llm = OpenAILLM(system_prompt=system_prompt, stream=False)
    llm_input_queue = TextStream()
    output_queue, chat_history_queue = llm.run(input_queue=llm_input_queue)

    await llm_input_queue.put("Test message with system prompt")
    response = await output_queue.get()

    assert isinstance(response, str)

    chat_history_response = await chat_history_queue.get()

    assert isinstance(chat_history_response, str)

    chat_history_response_json = json.loads(chat_history_response)
    assert chat_history_response_json["role"] == "user"
    assert chat_history_response_json["content"] == "Test message with system prompt"

    chat_history_response = await chat_history_queue.get()

    assert isinstance(chat_history_response, str)

    chat_history_response_json = json.loads(chat_history_response)
    assert chat_history_response_json["role"] == "assistant"
    assert chat_history_response_json["content"] == response

    await llm.close()


@pytest.mark.asyncio
async def test_openai_llm_with_different_temperature(load_env_file):
    temperature = 0.7
    llm = OpenAILLM(temperature=temperature, stream=False)
    llm_input_queue = TextStream()
    output_queue, chat_history_queue = llm.run(input_queue=llm_input_queue)

    await llm_input_queue.put("Test message with different temperature")
    response = await output_queue.get()

    assert isinstance(response, str)
    chat_history_response = await chat_history_queue.get()

    assert isinstance(chat_history_response, str)

    chat_history_response_json = json.loads(chat_history_response)
    assert chat_history_response_json["role"] == "user"
    assert chat_history_response_json["content"] == "Test message with different temperature"

    chat_history_response = await chat_history_queue.get()

    assert isinstance(chat_history_response, str)

    chat_history_response_json = json.loads(chat_history_response)
    assert chat_history_response_json["role"] == "assistant"
    assert chat_history_response_json["content"] == response
    await llm.close()


@pytest.mark.asyncio
async def test_openai_llm_with_custom_response_format(load_env_file):
    custom_response_format = {"type": "json_object"}
    llm = OpenAILLM(
        response_format=custom_response_format,
        stream=False,
        system_prompt="You are a helpful assistant. Reply with JSON.",
    )
    llm_input_queue = TextStream()
    output_queue, chat_history_queue = llm.run(input_queue=llm_input_queue)

    await llm_input_queue.put("Test message with custom response format")
    response = await output_queue.get()

    assert isinstance(response, str)

    response_json = json.loads(response)

    chat_history_response = await chat_history_queue.get()

    assert isinstance(chat_history_response, str)

    chat_history_response_json = json.loads(chat_history_response)
    assert chat_history_response_json["role"] == "user"
    assert chat_history_response_json["content"] == "Test message with custom response format"

    chat_history_response = await chat_history_queue.get()

    assert isinstance(chat_history_response, str)

    chat_history_response_json = json.loads(chat_history_response)
    assert chat_history_response_json["role"] == "assistant"
    assert chat_history_response_json["content"] == response

    await llm.close()


@pytest.mark.asyncio
async def test_openai_llm_with_tools(load_env_file, mock_tool):
    llm = OpenAILLM(tools=[mock_tool], tool_choice="required", stream=False)
    llm_input_queue = TextStream()
    output_queue, chat_history_queue = llm.run(input_queue=llm_input_queue)

    await llm_input_queue.put("Test message with tools")
    response = await output_queue.get()
    response = await output_queue.get()

    assert isinstance(response, str)

    chat_history_response = await chat_history_queue.get()

    assert isinstance(chat_history_response, str)

    chat_history_response_json = json.loads(chat_history_response)
    assert chat_history_response_json["role"] == "user"
    assert chat_history_response_json["content"] == "Test message with tools"

    chat_history_response = await chat_history_queue.get()

    assert isinstance(chat_history_response, str)

    chat_history_response_json = json.loads(chat_history_response)
    assert chat_history_response_json["role"] == "assistant"
    assert isinstance(chat_history_response_json.get("tool_calls"), list)
    assert len(chat_history_response_json.get("tool_calls")) == 1
    assert chat_history_response_json.get("tool_calls")[0].get("type") == "function"
    assert chat_history_response_json.get("tool_calls")[0].get("function").get("name") == "mock_tool"
    await llm.close()


@pytest.mark.asyncio
async def test_openai_llm_with_tool_choice_none(load_env_file, mock_tool):
    llm = OpenAILLM(tools=[mock_tool], tool_choice="none", stream=False)
    llm_input_queue = TextStream()
    output_queue, chat_history_queue = llm.run(input_queue=llm_input_queue)

    await llm_input_queue.put("Test message with tool choice none")
    response = await output_queue.get()

    assert isinstance(response, str)
    chat_history_response = await chat_history_queue.get()

    assert isinstance(chat_history_response, str)

    chat_history_response_json = json.loads(chat_history_response)
    assert chat_history_response_json["role"] == "user"
    assert chat_history_response_json["content"] == "Test message with tool choice none"

    chat_history_response = await chat_history_queue.get()

    assert isinstance(chat_history_response, str)

    chat_history_response_json = json.loads(chat_history_response)
    assert chat_history_response_json["role"] == "assistant"
    assert chat_history_response_json["content"] == response
    await llm.close()
