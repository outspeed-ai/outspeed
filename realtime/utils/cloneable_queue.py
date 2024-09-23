import asyncio
from typing import List


class CloneableQueue(asyncio.Queue):
    """An asynchronous queue where objects added to it are also added to its copies."""

    def __init__(self):
        super().__init__()
        self._clones: List[CloneableQueue] = []

    async def put(self, item):
        """Put an item in all queues of all instances."""
        self.put_nowait(item)

    def put_nowait(self, item):
        """Put an item in all queues of all instances."""
        super().put_nowait(item)
        for clone in self._clones:
            clone.put_nowait(item)

    async def clone(self):
        """Create a copy of this queue."""
        clone = CloneableQueue()
        self._clones.append(clone)
        return clone
