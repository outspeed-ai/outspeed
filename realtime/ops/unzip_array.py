import asyncio

from realtime.streams import AudioStream, ByteStream, Stream, TextStream, VideoStream


def unzip_array(input_queue: Stream):
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
                if isinstance(item, list):
                    for i in item:
                        await output_queue.put(i)
            except Exception as e:
                print(f"Error in unzip array function: {e}")
                continue

    asyncio.create_task(run())
    return output_queue
