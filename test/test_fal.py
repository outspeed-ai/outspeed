import asyncio
import os
import time
import av

from realtime.plugins.fal_vision import FalVision


async def test():
    client = FalVision(api_key=os.environ.get("FAL_API_KEY"), model="fal-ai/llavav15-13b")
    container = av.open("https://github.com/vikhyat/moondream/raw/main/assets/demo-1.jpg")

    # Iterate through frames in the container
    for frame in container.decode(video=0):
        # Return the first video frame
        await client.astream("Describe the image.", image=frame)
    # input_queue = asyncio.Queue()
    # output_queue = await client.arun(input_queue=input_queue)
    # await input_queue.put("Hello. How are you?")
    # while True:
    #     c = await output_queue.get()
    #     if not c:
    #         break
    #     print(c, end="")
    await client.aclose()


