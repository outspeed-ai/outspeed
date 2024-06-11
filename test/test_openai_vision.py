import asyncio
import os
import time
import av

from realtime.plugins.openai_vision import OpenAIVision


async def test():
    client = OpenAIVision(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    input_queue = asyncio.Queue()
    container = av.open("https://github.com/vikhyat/moondream/raw/main/assets/demo-1.jpg")
    # Iterate through frames in the container
    for frame in container.decode(video=0):
        # Return the first video frame
        await input_queue.put(frame)
    output_queue = await client.arun(input_queue=input_queue)
    while True:
        c = await output_queue.get()
        if not c:
            break
        print(c, end="")
    await client.aclose()


