import asyncio
from typing import Callable, TypeVar

from outspeed.streams import AudioStream, ByteStream, Stream, TextStream, VideoStream

T = TypeVar("T")
R = TypeVar("R")


def filter(input_queue: Stream[T], predicate: Callable[[T], bool]) -> Stream[T]:
    """
    Filter items from the input stream based on a predicate function.

    This function creates a new stream of the same type as the input stream and only includes
    items from the input stream where the predicate function returns True.

    Args:
        input_queue (Stream[T]): The input stream to filter.
        predicate (Callable[[T], bool]): The predicate function that determines which items to keep.

    Returns:
        Stream[T]: A new stream containing only the items where predicate(item) is True.

    Raises:
        ValueError: If the input queue type is not recognized.
    """
    # Determine the type of the output queue based on the input queue type
    if isinstance(input_queue, AudioStream):
        output_queue: Stream[T] = AudioStream()
    elif isinstance(input_queue, VideoStream):
        output_queue: Stream[T] = VideoStream()
    elif isinstance(input_queue, TextStream):
        output_queue: Stream[T] = TextStream()
    elif isinstance(input_queue, ByteStream):
        output_queue: Stream[T] = ByteStream()
    else:
        raise ValueError(f"Invalid input queue type: {type(input_queue)}")

    async def run() -> None:
        """
        Asynchronous task that continuously processes items from the input queue,
        applies the predicate function, and puts matching items into the output queue.
        """
        while True:
            # Get the next item from the input queue
            item = await input_queue.get()
            try:
                # Apply the predicate function to the item
                if item is not None and predicate(item):
                    # Only put items in output queue if predicate returns True
                    await output_queue.put(item)
            except Exception as e:
                # If an error occurs during filtering, log it and continue with the next item
                print(f"Error in filter predicate: {e}")
                continue

    # Create an asynchronous task to run the filtering process
    asyncio.create_task(run())

    return output_queue
