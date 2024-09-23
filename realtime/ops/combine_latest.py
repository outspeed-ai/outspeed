import asyncio
from typing import List

from realtime.streams import AudioStream, ByteStream, Stream, TextStream, VideoStream


def combine_latest(input_queues: List[Stream]):
    output_queues = []
    for q in input_queues:
        if isinstance(q, AudioStream):
            output_queues.append(AudioStream())
        elif isinstance(q, VideoStream):
            output_queues.append(VideoStream())
        elif isinstance(q, TextStream):
            output_queues.append(TextStream())
        elif isinstance(q, ByteStream):
            output_queues.append(ByteStream())

    async def run():
        while True:
            if any([q.empty() for q in input_queues]):
                await asyncio.sleep(0.2)
                continue

            # Dequeue from input and enqueue to output simultaneously
            for in_q, out_q in zip(input_queues, output_queues):
                item = in_q.get_nowait()
                out_q.put_nowait(item)

    asyncio.create_task(run())
    return output_queues
