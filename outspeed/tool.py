from typing import Dict, List, Literal, Type, TypedDict

import openai
from pydantic import BaseModel

from outspeed.data import TextData


class ToolType(TypedDict):
    id: str
    type: Literal["function"]
    function: Dict[str, str]


class ToolCallType(TypedDict):
    role: Literal["assistant"]
    tool_calls: List[ToolType]


class ToolCallResponseType(TypedDict):
    role: Literal["tool"]
    tool_call_id: str
    content: str


class Tool:
    name: str
    description: str
    parameters_type: Type[BaseModel]
    response_type: Type[BaseModel]

    def __init__(
        self,
        should_respond_after_tool_call: bool = False,
    ):
        self.should_respond_after_tool_call = should_respond_after_tool_call
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Tool name must be set and must be a string")
        if not self.description or not isinstance(self.description, str):
            raise ValueError("Tool description must be set and must be a string")
        if not self.parameters_type or not issubclass(self.parameters_type, BaseModel):
            raise ValueError("Tool parameters type must be set and must be a Pydantic BaseModel")
        if not self.response_type or not issubclass(self.response_type, BaseModel):
            raise ValueError("Tool response type must be set and must be a Pydantic BaseModel")

    async def run(self, input_parameters: Type[BaseModel]):
        raise NotImplementedError("Tool run method must be implemented by the subclass")

    def to_openai_tool_json(self):
        json = openai.pydantic_function_tool(
            model=self.parameters_type,
            name=self.name,
            description=self.description,
        )
        return json

    def to_openai_tool_response_json(self, response: BaseModel):
        json = openai.pydantic_function_tool(
            model=self.response_type,
            name=self.name,
            description=self.description,
        )
        return json

    async def _run_tool(self, function_json: dict):
        if function_json["function"]["name"] != self.name:
            raise ValueError(f"Tool name mismatch: {function_json['name']} != {self.name}")

        input_parameters = self.parameters_type.model_validate(function_json["function"]["arguments"])
        response = await self.run(input_parameters)

        if not isinstance(response, self.response_type):
            raise ValueError(f"Tool response type mismatch: {type(response)} != {self.response_type}")

        json = {"tool_call_id": function_json["id"], "role": "tool", "content": response.model_dump_json()}

        return json


class ToolCallData(TextData):
    pass


class ToolCallResponseData(TextData):
    @property
    def tool_call_id(self):
        return self.get_json().get("tool_call_id")
