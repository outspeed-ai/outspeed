import os
from typing import Any, Dict, Literal, Optional

from outspeed.plugins.openai_llm import OpenAILLM
from outspeed.tool import Tool


class FireworksLLM(OpenAILLM):
    def __init__(
        self,
        model: str = "accounts/fireworks/models/llama-v3p1-8b-instruct",
        api_key=None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stream: bool = True,
        temperature: float = 1.0,
        response_format: Dict[str, Any] = {"type": "text"},
        tools: Optional[list[Tool]] = None,
        tool_choice: Literal["auto", "none", "required"] = "auto",
    ):
        api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("Fireworks API key is required")

        if tools and model not in [
            "accounts/fireworks/models/firefunction-v2",
            "accounts/fireworks/models/firefunction-v1",
        ]:
            raise ValueError("Tools are only supported for firefunction-v2 and firefunction-v1")

        base_url: str = base_url or "https://api.fireworks.ai/inference/v1"

        super().__init__(
            model=model,
            stream=stream,
            temperature=temperature,
            response_format=response_format,
            api_key=api_key,
            base_url=base_url,
            system_prompt=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
        )
