import asyncio
from typing import Type

import openai
from pydantic import BaseModel


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters_type: Type[BaseModel],
        response_type: Type[BaseModel],
        should_respond_after_tool_call: bool = False,
    ):
        self.name = name
        self.description = description

        self.parameters_type = parameters_type
        self.response_type = response_type
        self.should_respond_after_tool_call = should_respond_after_tool_call

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
        print(function_json)
        if function_json["function"]["name"] != self.name:
            raise ValueError(f"Tool name mismatch: {function_json['name']} != {self.name}")

        input_parameters = self.parameters_type.model_validate(function_json["function"]["arguments"])
        response = await self.run(input_parameters)

        if not isinstance(response, self.response_type):
            raise ValueError(f"Tool response type mismatch: {type(response)} != {self.response_type}")

        json = {"tool_call_id": function_json["id"], "role": "tool", "content": response.model_dump_json()}

        return json
