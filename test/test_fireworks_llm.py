import asyncio
import os
import time

from realtime.plugins.fireworks_llm import FireworksLLM


async def test():
    client = FireworksLLM(
        api_key=os.environ.get("FIREWORKS_API_KEY"), model="accounts/fireworks/models/llama-v3-8b-instruct"
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


