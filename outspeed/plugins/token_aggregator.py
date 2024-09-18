import asyncio
from typing import List, Optional

from outspeed.plugins.base_plugin import Plugin
from outspeed.streams import TextStream

# Define sentence endings for token aggregation
SENTENCE_ENDINGS: List[str] = [".", "!", "?", "\n"]


class TokenAggregator(Plugin):
    """
    A plugin that aggregates tokens into sentences or meaningful chunks.

    This plugin receives tokens from an input queue, aggregates them into
    larger chunks (e.g., sentences), and sends the aggregated text to an
    output queue when certain conditions are met.
    """

    def __init__(self):
        """Initialize the TokenAggregator plugin."""
        super().__init__()
        self.output_queue: TextStream = TextStream()
        self.buffer: str = ""
        self.input_queue: Optional[asyncio.Queue] = None
        self.interrupt_queue: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None
        self._interrupt_task: Optional[asyncio.Task] = None

    def run(self, input_queue: asyncio.Queue) -> asyncio.Queue:
        """
        Start the token aggregation process.

        Args:
            input_queue (asyncio.Queue): The input queue to receive tokens from.

        Returns:
            asyncio.Queue: The output queue where aggregated text will be sent.
        """
        self.input_queue = input_queue
        self._task = asyncio.create_task(self._aggregate_tokens())
        return self.output_queue

    async def _aggregate_tokens(self) -> None:
        """
        Aggregate tokens into larger chunks and send them to the output queue.

        This method runs in a loop, continuously reading tokens from the input queue,
        aggregating them in the buffer, and sending completed chunks to the output queue.
        """
        while True:
            token = await self.input_queue.get()
            if token is None:
                if self.buffer:
                    await self.output_queue.put(self.buffer)
                    self.buffer = ""
                await self.output_queue.put(None)
                continue
            if not token:
                continue
            self.buffer += token

            # Find the last occurrence of any sentence ending
            i = max((self.buffer.rfind(ending) for ending in SENTENCE_ENDINGS), default=-1)

            # If a sentence ending is found and the chunk is long enough, send it to the output queue
            if i != -1 and len(self.buffer[: i + 1]) >= 10:
                await self.output_queue.put(self.buffer[: i + 1])
                self.buffer = self.buffer[i + 1 :]

    async def close(self) -> None:
        """Cancel the token aggregation task."""
        if self._task:
            self._task.cancel()

    async def _interrupt(self) -> None:
        """
        Handle interruptions (e.g., when the user starts speaking).

        This method listens for interrupt signals and clears the buffer and output queue
        when an interrupt is received while the output queue is not empty.
        """
        while True:
            user_speaking = await self.interrupt_queue.get()
            if user_speaking and not self.output_queue.empty():
                self.buffer = ""
                if self._task:
                    self._task.cancel()
                while not self.output_queue.empty():
                    self.output_queue.get_nowait()
                print("Done cancelling Token Aggregator")
                self._task = asyncio.create_task(self._aggregate_tokens())

    async def set_interrupt(self, interrupt_queue: asyncio.Queue) -> None:
        """
        Set up the interrupt handling mechanism.

        Args:
            interrupt_queue (asyncio.Queue): The queue to receive interrupt signals from.
        """
        self.interrupt_queue = interrupt_queue
        self._interrupt_task = asyncio.create_task(self._interrupt())
