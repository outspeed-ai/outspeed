import os
from typing import Any, Dict, Literal, Optional

from outspeed.plugins.openai_llm import OpenAILLM
from outspeed.tool import Tool


class GroqLLM(OpenAILLM):
    """
    A plugin for interacting with Groq's LLM API.

    This class handles streaming chat completions, manages chat history,
    and provides methods for running the LLM and handling interruptions.
    """

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stream: bool = True,
        temperature: float = 1.0,
        response_format: Dict[str, Any] = {"type": "text"},
        tools: Optional[list[Tool]] = None,
        tool_choice: Literal["auto", "none", "required"] = "auto",
    ):
        """
        Initialize the GroqLLM plugin.

        Args:
            model (str): The name of the Groq model to use.
            api_key (Optional[str]): The API key for Groq. If None, it will be read from the environment.
            base_url (Optional[str]): The base URL for the Groq API. Not used in the current implementation.
            system_prompt (Optional[str]): The system prompt to be used in the conversation.
            stream (bool): Whether to stream the response or not.
            temperature (float): The temperature parameter for the LLM.
            response_format (Optional[Dict[str, Any]]): The desired response format.
        """
        api_key: str = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key is required")

        base_url: str = base_url or "https://api.groq.com/openai/v1"

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
