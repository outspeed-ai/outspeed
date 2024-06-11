import asyncio

from realtime.plugins.base_plugin import Plugin
from realtime.streams import TextStream

SENTENCE_ENDINGS = [".", "!", "?", "\n"]


class TokenAggregator(Plugin):
    def __init__(self):
        super().__init__()
        self.output_queue = TextStream()
        self.buffer = ""

    async def run(self, input_queue: asyncio.Queue) -> asyncio.Queue:
        self.input_queue = input_queue
        self._task = asyncio.create_task(self._aggregate_tokens())
        return self.output_queue

    async def _aggregate_tokens(self):
        while True:
            token = await self.input_queue.get()
            if not token:
                continue
            self.buffer += token
            for ending in SENTENCE_ENDINGS:
                i = self.buffer.rfind(ending)
                if i != -1 and len(self.buffer[: i + 1]) >= 10:
                    await self.output_queue.put(self.buffer[: i + 1])
                    self.buffer = self.buffer[i + 1 :]
                    break

    async def close(self):
        self._task.cancel()

    async def _interrupt(self):
        while True:
            user_speaking = await self.interrupt_queue.get()
            if user_speaking and not self.output_queue.empty():
                self.buffer = ""
                self._task.cancel()
                while not self.output_queue.empty():
                    self.output_queue.get_nowait()
                print("Done cancelling Token Aggregator")
                self._task = asyncio.create_task(self._aggregate_tokens())

    async def set_interrupt(self, interrupt_queue: asyncio.Queue):
        self.interrupt_queue = interrupt_queue
        self._interrupt_task = asyncio.create_task(self._interrupt())
