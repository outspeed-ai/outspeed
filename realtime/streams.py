import asyncio
from typing import List


class Stream(asyncio.Queue):
    """An asynchronous queue where objects added to it are also added to its copies."""

    def __init__(self):
        super().__init__()
        self._clones: List[Stream] = []

    async def put(self, item):
        """Put an item in all queues of all instances."""
        self.put_nowait(item)

    def put_nowait(self, item):
        """Put an item in all queues of all instances."""
        super().put_nowait(item)
        for clone in self._clones:
            clone.put_nowait(item)


class AudioStream(Stream):
    type = "audio"

    async def clone(self):
        """Create a copy of this queue."""
        clone = AudioStream()
        self._clones.append(clone)
        return clone


class VideoStream(Stream):
    type = "video"

    async def clone(self):
        """Create a copy of this queue."""
        clone = VideoStream()
        self._clones.append(clone)
        return clone


class TextStream(Stream):
    type = "text"

    async def clone(self):
        """Create a copy of this queue."""
        clone = TextStream()
        self._clones.append(clone)
        return clone


class ByteStream(Stream):
    type = "bytes"

    async def clone(self):
        """Create a copy of this queue."""
        clone = ByteStream()
        self._clones.append(clone)
        return clone
