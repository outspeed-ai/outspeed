import asyncio
from typing import List

from realtime.streams import AudioStream, ByteStream, Stream, TextStream, VideoStream


def merge(input_queues: List[Stream]):
    output_queue = None
    if not all(isinstance(x, type(input_queues[0])) for x in input_queues):
        raise ValueError("All input queues must be of the same type")
    if isinstance(input_queues[0], AudioStream):
        output_queue = AudioStream()
    elif isinstance(input_queues[0], VideoStream):
        output_queue = VideoStream()
    elif isinstance(input_queues[0], TextStream):
        output_queue = TextStream()
    elif isinstance(input_queues[0], ByteStream):
        output_queue = ByteStream()
    else:
        raise ValueError(f"Invalid input queue type: {type(input_queues[0])}")

    async def get(queue):
        while True:
            item = await queue.get()
            output_queue.put_nowait(item)

    for q in input_queues:
        asyncio.create_task(get(q))
    return output_queue
