import asyncio
import logging
import traceback
from typing import Type

from outspeed.data import SessionData
from outspeed.streams import Stream, TextStream, VADStream
from outspeed.utils.vad import VADState


class Node:
    def __init__(self):
        pass

    def run(self, input_stream: Type[Stream]) -> Stream:
        if not issubclass(input_stream, Stream):
            raise ValueError("All arguments must be instances of Stream")

        self._input_queue = input_stream
        self._output_queue = Stream()

        self._task = asyncio.create_task(self._process_stream())

        return self._output_queue

    async def _process_stream(self):
        try:
            while True:
                input_data = await self._input_queue.get()
                await self.process(input_data)
        except Exception as e:
            logging.error(f"Error in node {self.__class__.__name__}: {e}")
            logging.error(traceback.format_exc())
            raise asyncio.CancelledError()

    async def process(self, input_data):
        raise NotImplementedError()


class CustomLLMNode(Node):
    def run(self, input_stream: Type[Stream]) -> Stream:
        if not isinstance(input_stream, Stream):
            raise ValueError("All arguments must be instances of Stream")

        self._input_queue = input_stream
        self._output_queue = TextStream()

        print("Creating task")
        self._task = asyncio.create_task(self._process_stream())

        return self._output_queue

    async def _process_stream(self):
        try:
            while True:
                input_data = await self._input_queue.get()
                print(input_data)
                if isinstance(input_data, SessionData):
                    await self._output_queue.put(input_data)
                else:
                    output = await self.process(input_data)
                    await self._output_queue.put(output)
        except Exception as e:
            logging.error(f"Error in node {self.__class__.__name__}: {e}")
            raise asyncio.CancelledError()

    def set_interrupt_stream(self, interrupt_stream: VADStream):
        if isinstance(interrupt_stream, VADStream):
            self.interrupt_queue = interrupt_stream
        else:
            raise ValueError("Interrupt stream must be a VADStream")
        self._interrupt_task = asyncio.create_task(self._interrupt())

    async def _interrupt(self):
        while True:
            vad_state: VADState = await self.interrupt_queue.get()
            if vad_state == VADState.SPEAKING and (not self._input_queue.empty() or not self._output_queue.empty()):
                while not self._output_queue.empty():
                    self._output_queue.get_nowait()
                while not self._input_queue.empty():
                    self._input_queue.get_nowait()
                logging.info("Done cancelling LLM")

    async def close(self):
        self._task.cancel()
