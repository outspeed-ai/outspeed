import asyncio
from typing import List, Union

from realtime.streams import AudioStream, ByteStream, Stream, TextStream, VideoStream


def merge(input_queues: List[Stream]) -> Union[AudioStream, VideoStream, TextStream, ByteStream]:
    """
    Merge multiple input streams of the same type into a single output stream.

    This function takes a list of input streams and combines them into a single output stream
    of the same type. It supports AudioStream, VideoStream, TextStream, and ByteStream types.

    Args:
        input_queues (List[Stream]): A list of input streams to be merged.

    Returns:
        Union[AudioStream, VideoStream, TextStream, ByteStream]: A single output stream
        containing data from all input streams.

    Raises:
        ValueError: If the input queues are not all of the same type or if an unsupported
        stream type is provided.
    """
    # Initialize the output queue
    output_queue: Union[AudioStream, VideoStream, TextStream, ByteStream] = None

    # Check if all input queues are of the same type
    if not all(isinstance(x, type(input_queues[0])) for x in input_queues):
        raise ValueError("All input queues must be of the same type")

    # Determine the type of the output queue based on the input queue type
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

    async def get(queue: Stream) -> None:
        """
        Asynchronous function to continuously get items from an input queue
        and put them into the output queue.

        Args:
            queue (Stream): The input stream to read from.
        """
        while True:
            item = await queue.get()
            output_queue.put_nowait(item)

    # Create a task for each input queue to continuously read and merge data
    for q in input_queues:
        asyncio.create_task(get(q))

    return output_queue
