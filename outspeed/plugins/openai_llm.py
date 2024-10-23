import asyncio
import json
import logging
import os
import traceback
from typing import Any, Dict, Literal, Optional, Tuple, Union

from openai import AsyncOpenAI

from outspeed.data import SessionData, TextData
from outspeed.ops.merge import merge
from outspeed.plugins.base_plugin import Plugin
from outspeed.streams import TextStream, VADStream
from outspeed.tool import Tool, ToolCallResponseData
from outspeed.utils import tracing
from outspeed.utils.vad import VADState


class OpenAILLM(Plugin):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key=None,
        base_url=None,
        system_prompt=None,
        stream: bool = True,
        temperature: float = 1.0,
        response_format: Dict[str, Any] = {"type": "text"},
        tools: Optional[list[Tool]] = None,
        tool_choice: Literal["auto", "none", "required"] = "auto",
    ):
        super().__init__()
        self._model: str = model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self._api_key = api_key
        self._client = AsyncOpenAI(api_key=self._api_key, base_url=base_url)
        self._history = []
        self.output_queue = TextStream()
        self.chat_history_queue = TextStream()
        self._generating = False
        self._stream = stream
        self._response_format = response_format
        self._system_prompt = system_prompt
        if self._system_prompt is not None:
            self._history.append({"role": "system", "content": self._system_prompt})
        self._temperature = temperature
        self._tools = tools
        self._tool_choice = tool_choice
        self._tool_output_queue = TextStream()
        self._tool_call_tasks = []
        self._removed_tool_calls = set()

    @property
    def chat_history(self) -> list[dict]:
        return self._history

    async def _stream_chat_completions(self):
        try:
            while True:
                data: Union[SessionData, TextData, str] = await self.input_queue.get()
                if data is None:
                    continue

                if isinstance(data, SessionData):
                    await self.output_queue.put(data)
                    continue

                self._generating = True

                if isinstance(data, str) or isinstance(data, TextData):
                    if self._history and self._history[-1].get("tool_calls"):
                        tool_calls = self._history.pop().get("tool_calls")
                        self._removed_tool_calls.update([tool_call["id"] for tool_call in tool_calls])
                    self._history.append({"role": "user", "content": data})
                elif isinstance(data, list) and all(isinstance(item, ToolCallResponseData) for item in data):
                    tool_call_response_json = [
                        item.get_json() for item in data if item.tool_call_id not in self._removed_tool_calls
                    ]
                    if not tool_call_response_json:
                        continue
                    self._history.extend(tool_call_response_json)
                else:
                    raise ValueError(f"Unknown type in input queue: {data}")

                self.chat_history_queue.put_nowait(json.dumps(self._history[-1]))
                tracing.register_event(tracing.Event.LLM_START)

                chunk_stream = await self._client.chat.completions.create(
                    model=self._model,
                    stream=self._stream,
                    messages=self._history,
                    response_format=self._response_format,
                    temperature=self._temperature,
                    tools=[tool.to_openai_tool_json() for tool in self._tools],
                    tool_choice="none" if self._history[-1]["role"] == "tool" else self._tool_choice,
                )  # type: ignore

                self._history.append({"role": "assistant"})
                tracing.register_event(tracing.Event.LLM_TTFB)

                if self._stream:
                    async for chunk in chunk_stream:
                        if len(chunk.choices) == 0:
                            continue

                        elif chunk.choices[0].delta.content:
                            if not self._history[-1].get("content"):
                                self._history[-1]["content"] = ""
                            self._history[-1]["content"] += chunk.choices[0].delta.content
                            await self.output_queue.put(chunk.choices[0].delta.content)

                        elif chunk.choices[0].delta.tool_calls:
                            if not self._history[-1].get("tool_calls"):
                                self._history[-1]["tool_calls"] = []
                            for tool in chunk.choices[0].delta.tool_calls:
                                if tool.index == len(self._history[-1]["tool_calls"]):
                                    self._history[-1]["tool_calls"].append(
                                        {
                                            "id": tool.id,
                                            "type": tool.type,
                                            "function": {
                                                "arguments": tool.function.arguments,
                                                "name": tool.function.name,
                                            },
                                        }
                                    )
                                elif tool.index < len(self._history[-1]["tool_calls"]):
                                    self._history[-1]["tool_calls"][tool.index]["function"]["arguments"] += (
                                        tool.function.arguments
                                    )
                                else:
                                    raise ValueError(f"Tool call index out of bounds: {tool.index}")
                else:
                    if chunk_stream.choices[0].message.content:
                        self._history[-1]["content"] = chunk_stream.choices[0].message.content
                        await self.output_queue.put(chunk_stream.choices[0].message.content)
                    elif chunk_stream.choices[0].message.tool_calls:
                        self._history[-1]["tool_calls"] = chunk_stream.choices[0].message.tool_calls

                if self._history[-1].get("tool_calls"):
                    asyncio.create_task(self._handle_function_call_arguments_done(self._history[-1]["tool_calls"]))

                tracing.register_event(tracing.Event.LLM_END)
                tracing.register_metric(tracing.Metric.LLM_TOTAL_BYTES, len(self._history[-1].get("content", "")))
                logging.info(
                    "llm: %s",
                    self._history[-1].get("content", "")
                    if self._history[-1].get("content")
                    else self._history[-1].get("tool_calls", []),
                )

                self.chat_history_queue.put_nowait(json.dumps(self._history[-1]))

                self._generating = False

                await self.output_queue.put(None)
        except Exception as e:
            logging.error("Error streaming chat completions", e)
            self._generating = False
            raise asyncio.CancelledError()

    def run(self, input_queue: TextStream) -> Tuple[TextStream, TextStream]:
        self.input_queue = merge([input_queue, self._tool_output_queue])
        self._task = asyncio.create_task(self._stream_chat_completions())
        return self.output_queue, self.chat_history_queue

    async def close(self):
        self._task.cancel()

    async def _interrupt(self):
        while True:
            vad_state: VADState = await self.interrupt_queue.get()
            if vad_state == VADState.SPEAKING and (
                not self.input_queue.empty() or not self.output_queue.empty() or self._generating
            ):
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                for task in self._tool_call_tasks:
                    task.cancel()
                for task in self._tool_call_tasks:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                while not self.output_queue.empty():
                    self.output_queue.get_nowait()
                while not self.input_queue.empty():
                    self.input_queue.get_nowait()
                logging.info("Done cancelling LLM")
                self._generating = False
                self._task = asyncio.create_task(self._stream_chat_completions())

    def set_interrupt_stream(self, interrupt_stream: VADStream):
        if isinstance(interrupt_stream, VADStream):
            self.interrupt_queue = interrupt_stream
        else:
            raise ValueError("Interrupt stream must be a VADStream")
        self._interrupt_task = asyncio.create_task(self._interrupt())

    async def _handle_function_call_arguments_done(self, tool_calls: list[dict]):
        current_tool_calls_tasks = []
        try:
            for tool_call in tool_calls:
                current_tool_calls_tasks.append(asyncio.create_task(self._run_tool(tool_call)))
            self._tool_call_tasks.extend(current_tool_calls_tasks)
            results = await asyncio.gather(*current_tool_calls_tasks)
            await self._tool_output_queue.put(results)
        except Exception as e:
            logging.error(f"Error handling function call arguments: {e} \n")
            logging.error(traceback.format_exc())

    async def _run_tool(self, tool_call: dict):
        if not self._tools:
            return

        for tool in self._tools:
            if tool.name == tool_call["function"]["name"]:
                logging.info(f"Calling tool {tool.name} with arguments: {tool_call['function']['arguments']} \n")
                result = await tool._run_tool(
                    {
                        "id": tool_call["id"],
                        "function": {
                            "arguments": json.loads(tool_call["function"]["arguments"]),
                            "name": tool_call["function"]["name"],
                        },
                    }
                )
                logging.info(f"Tool {tool.name} returned: {result} \n")
                return ToolCallResponseData.from_json(result)

        return ToolCallResponseData.from_json(
            {"tool_call_id": tool_call["id"], "role": "tool", "content": "Invalid Tool Name"}
        )
