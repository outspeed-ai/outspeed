import asyncio
import os
import time

from realtime.plugins.eleven_labs_tts import ElevenLabsTTS


async def test():
    client = ElevenLabsTTS(
        api_key=os.environ.get("ELEVENLABS_API_KEY"),
    )
    input_queue = asyncio.Queue()
    output_queue = await client.arun(input_queue=input_queue)
    start_time = time.time()
    await input_queue.put("Hello. How are you?")
    await output_queue.get()
    print("Time to first chunk:", time.time() - start_time)
    await client.aclose()

