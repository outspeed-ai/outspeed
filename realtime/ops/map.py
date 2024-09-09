import asyncio
from typing import Callable, TypeVar

from realtime.streams import AudioStream, ByteStream, Stream, TextStream, VideoStream

T = TypeVar("T")
R = TypeVar("R")


def map(input_queue: Stream[T], func: Callable[[T], R]) -> Stream[R]:
    """
    Apply a function to each item in the input stream and return a new stream with the results.

    This function creates a new stream of the same type as the input stream and applies
    the given function to each item in the input stream, putting the results into the
    output stream.

    Args:
        input_queue (Stream[T]): The input stream to map over.
        func (Callable[[T], R]): The function to apply to each item in the input stream.

    Returns:
        Stream[R]: A new stream containing the results of applying the function to each input item.

    Raises:
        ValueError: If the input queue type is not recognized.
    """
    # Determine the type of the output queue based on the input queue type
    if isinstance(input_queue, AudioStream):
        output_queue: Stream[R] = AudioStream()
    elif isinstance(input_queue, VideoStream):
        output_queue: Stream[R] = VideoStream()
    elif isinstance(input_queue, TextStream):
        output_queue: Stream[R] = TextStream()
    elif isinstance(input_queue, ByteStream):
        output_queue: Stream[R] = ByteStream()
    else:
        raise ValueError(f"Invalid input queue type: {type(input_queue)}")

    async def run() -> None:
        """
        Asynchronous task that continuously processes items from the input queue,
        applies the mapping function, and puts the results into the output queue.
        """
        while True:
            # Get the next item from the input queue
            item = await input_queue.get()
            try:
                # Apply the mapping function to the item
                result = func(item)
            except Exception as e:
                # If an error occurs during mapping, log it and continue with the next item
                print(f"Error in map function: {e}")
                continue
            # Put the result into the output queue
            await output_queue.put(result)

    # Create an asynchronous task to run the mapping process
    asyncio.create_task(run())

    return output_queue
