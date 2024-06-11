import asyncio
import os
import time

from realtime.plugins.openai_llm import OpenAILLM


async def test():
    client = OpenAILLM(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    input_queue = asyncio.Queue()
    output_queue = await client.arun(input_queue=input_queue)
    await input_queue.put("Hello. How are you?")
    while True:
        c = await output_queue.get()
        if not c:
            break
        print(c, end="")
    await client.aclose()


