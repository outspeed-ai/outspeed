import asyncio
from typing import Callable, List

from realtime.streams import AudioStream, ByteStream, Stream, TextStream, VideoStream


def map(input_queue: Stream, func: Callable):
    output_queue = None
    if isinstance(input_queue, AudioStream):
        output_queue = AudioStream()
    elif isinstance(input_queue, VideoStream):
        output_queue = VideoStream()
    elif isinstance(input_queue, TextStream):
        output_queue = TextStream()
    elif isinstance(input_queue, ByteStream):
        output_queue = ByteStream()
    else:
        raise ValueError(f"Invalid input queue type: {type(input_queue)}")

    async def run():
        while True:
            item = await input_queue.get()
            try:
                result = func(item)
            except Exception as e:
                print(f"Error in map function: {e}")
                continue
            await output_queue.put(result)

    asyncio.create_task(run())
    return output_queue
