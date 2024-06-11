import asyncio
from typing import Callable, List

from realtime.streams import AudioStream, ByteStream, Stream, TextStream, VideoStream


def join(input_queues: List[Stream], func: Callable):
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

    async def run():
        while True:
            if any([q.empty() for q in input_queues]):
                await asyncio.sleep(0.2)
                continue

            items = [in_q.get_nowait() for in_q in input_queues]
            try:
                await output_queue.put(func(*items))
            except Exception as e:
                print(f"Error in join function: {e}")
                continue

    asyncio.create_task(run())
    return output_queue
