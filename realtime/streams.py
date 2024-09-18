import asyncio
from typing import Any, List


class Stream(asyncio.Queue):
    """
    An asynchronous queue where objects added to it are also added to its copies.

    This class extends asyncio.Queue to provide a mechanism for creating and managing
    multiple copies (clones) of the stream, where any item added to the original stream
    is automatically added to all of its clones.
    """

    def __init__(self) -> None:
        """Initialize the Stream with an empty list of clones."""
        super().__init__()
        self._clones: List[Stream] = []
        self._cache: List[Any] = []

    def put_nowait(self, item: Any) -> None:
        """
        Put an item in all queues of all instances immediately.

        This method adds the item to the current queue and all of its clones
        without waiting for an available slot.

        Args:
            item (Any): The item to be added to the queue and all its clones.
        """
        super().put_nowait(item)
        for clone in self._clones:
            clone.put_nowait(item)

    def get_nowait(self) -> Any:
        """
        Get the first element from the queue without removing it.
        """
        if self._cache:
            return self._cache.pop(0)
        return super().get_nowait()

    def get_first_element_without_removing(self) -> Any:
        """
        Get the first element from the queue without removing it.
        """
        if self._cache:
            return self._cache[0]
        try:
            element = super().get_nowait()
        except asyncio.QueueEmpty:
            return None
        self._cache.append(element)
        return element

    def get_element_at_index(self, index: int) -> Any:
        """
        Get the element at the given index without removing it.
        """
        while len(self._cache) <= index and super().qsize() > 0:
            self._cache.append(super().get_nowait())
        if index < len(self._cache):
            return self._cache.pop(index)
        return None

    def qsize(self) -> int:
        """
        Get the number of elements in the queue.
        """
        return super().qsize() + len(self._cache)


class AudioStream(Stream):
    """
    A specialized Stream for audio data.

    This class extends the Stream class to handle audio-specific properties
    such as sample rate.
    """

    type: str = "audio"

    def __init__(self, sample_rate: int = 8000) -> None:
        """
        Initialize the AudioStream with a given sample rate.

        Args:
            sample_rate (int, optional): The sample rate of the audio stream. Defaults to 8000.
        """
        super().__init__()
        self.sample_rate: int = sample_rate

    def clone(self) -> "AudioStream":
        """
        Create a copy of this AudioStream.

        Returns:
            AudioStream: A new AudioStream instance that is a clone of the current one.
        """
        clone = AudioStream(sample_rate=self.sample_rate)
        self._clones.append(clone)
        return clone


class VideoStream(Stream):
    """A specialized Stream for video data."""

    type: str = "video"

    def clone(self) -> "VideoStream":
        """
        Create a copy of this VideoStream.

        Returns:
            VideoStream: A new VideoStream instance that is a clone of the current one.
        """
        clone = VideoStream()
        self._clones.append(clone)
        return clone


class TextStream(Stream):
    """A specialized Stream for text data."""

    type: str = "text"

    def clone(self) -> "TextStream":
        """
        Create a copy of this TextStream.

        Returns:
            TextStream: A new TextStream instance that is a clone of the current one.
        """
        clone = TextStream()
        self._clones.append(clone)
        return clone


class ByteStream(Stream):
    """A specialized Stream for byte data."""

    type: str = "bytes"

    def clone(self) -> "ByteStream":
        """
        Create a copy of this ByteStream.

        Returns:
            ByteStream: A new ByteStream instance that is a clone of the current one.
        """
        clone = ByteStream()
        self._clones.append(clone)
        return clone


class VADStream(Stream):
    """
    A specialized Stream for audio data.

    This class extends the Stream class to handle audio-specific properties
    such as sample rate.
    """

    type: str = "vad"

    def __init__(self) -> None:
        """
        Initialize the AudioStream with a given sample rate.

        Args:
            sample_rate (int, optional): The sample rate of the audio stream. Defaults to 8000.
        """
        super().__init__()

    def clone(self) -> "AudioStream":
        """
        Create a copy of this AudioStream.

        Returns:
            AudioStream: A new AudioStream instance that is a clone of the current one.
        """
        clone = VADStream()
        self._clones.append(clone)
        return clone
