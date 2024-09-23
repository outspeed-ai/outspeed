import base64
import fractions
import io
import time
from typing import Any, Dict, Optional, Union
import uuid

import numpy as np
from av import AudioFrame, VideoFrame
from PIL import Image

from realtime.utils.clock import Clock


class AudioData:
    """
    A class to handle audio data with various utilities.

    This class provides methods for working with audio data in different formats,
    including conversion between bytes and AudioFrame objects, duration calculation,
    and base64 encoding.
    """

    def __init__(
        self,
        data: Union[bytes, AudioFrame],
        sample_rate: int = 8000,
        channels: int = 1,
        sample_width: int = 2,
        format: str = "wav",
        relative_start_time: Optional[float] = None,
        extra_tags: Dict[str, Any] = {},
    ):
        """
        Initialize an AudioData object.

        Args:
            data (Union[bytes, AudioFrame]): The audio data as bytes or an AudioFrame.
            sample_rate (int): The sample rate of the audio in Hz. Defaults to 8000.
            channels (int): The number of audio channels. Defaults to 1 (mono).
            sample_width (int): The width of each sample in bytes. Defaults to 2.
            format (str): The audio format ('wav' or 'opus'). Defaults to 'wav'.
            relative_start_time (Optional[float]): The relative start time of the audio.
                                                   If None, uses the current playback time.

        Raises:
            ValueError: If the data is not of type bytes or AudioFrame.
        """
        if not isinstance(data, (bytes, AudioFrame)):
            raise ValueError("AudioData data must be bytes or av.AudioFrame")
        self.data: Union[bytes, AudioFrame] = data
        self.sample_rate: int = sample_rate
        self.channels: int = channels
        self.sample_width: int = sample_width
        self.format: str = format
        self.extra_tags: Dict[str, Any] = extra_tags
        self.relative_start_time: float = relative_start_time or Clock.get_playback_time()

    def get_bytes(self) -> bytes:
        """
        Convert the audio data to bytes.

        Returns:
            bytes: The audio data as bytes.

        Raises:
            ValueError: If the data is not of type bytes or AudioFrame.
        """
        if isinstance(self.data, bytes):
            return self.data
        elif isinstance(self.data, AudioFrame):
            return self.data.to_ndarray().tobytes()
        else:
            raise ValueError("AudioData data must be bytes or av.AudioFrame")

    def get_duration_seconds(self) -> float:
        """
        Calculate the duration of the audio in seconds.

        Returns:
            float: The duration of the audio in seconds.
        """
        audio_bytes = self.get_bytes()
        return len(audio_bytes) / (self.sample_rate * self.channels * self.sample_width)

    def get_base64(self) -> str:
        """
        Encode the audio data to base64.

        Returns:
            str: The base64 encoded audio data as a string.
        """
        return base64.b64encode(self.get_bytes()).decode("utf-8")

    def get_start_seconds(self) -> float:
        """
        Get the start time of the audio in seconds.

        Returns:
            float: The relative start time of the audio in seconds.
        """
        return self.relative_start_time

    def get_pts(self) -> int:
        """
        Get the presentation timestamp (pts) of the audio frame.

        Returns:
            int: The pts value.
        """
        return int(self.relative_start_time * self.sample_rate)

    def get_frame(self) -> AudioFrame:
        """
        Convert the audio data to an AudioFrame.

        Returns:
            AudioFrame: The audio data as an AudioFrame object.

        Raises:
            ValueError: If the data format is invalid or unsupported.
        """
        if isinstance(self.data, AudioFrame):
            return self.data
        elif isinstance(self.data, bytes):
            if len(self.data) < 2:
                raise ValueError("AudioData data must be at least 2 bytes")
            if len(self.data) % 2 != 0:
                audio_data = self.data[:-1]
            else:
                audio_data = self.data

            # Convert bytes to numpy array
            array = np.frombuffer(audio_data, dtype=np.int16).reshape(1, -1)  # mono has 1 channel

            # Set channel layout
            if self.channels == 2:
                channel_layout = "stereo"
            elif self.channels == 1:
                channel_layout = "mono"
            else:
                raise ValueError("AudioData channels must be 1 or 2")

            # Set audio format
            if self.format == "wav":
                format = "s16"
            elif self.format == "opus":
                format = "opus"
            else:
                raise ValueError("AudioData format must be wav or opus")

            # Create AudioFrame from numpy array
            frame = AudioFrame.from_ndarray(array, format=format, layout=channel_layout)
            frame.sample_rate = self.sample_rate
            frame.pts = self.get_pts()
            frame.time_base = fractions.Fraction(1, self.sample_rate)
            return frame
        else:
            raise ValueError("AudioData data must be bytes or av.AudioFrame")


class ImageData:
    """
    A class to handle video/image data with various utilities.

    This class provides methods for working with video/image data in different formats,
    including conversion between various types and VideoFrame objects, and duration calculation.
    """

    def __init__(
        self,
        data: Union[np.ndarray, VideoFrame, Image.Image, bytes],
        width: int = 640,
        height: int = 480,
        frame_rate: int = 30,
        format: str = "jpeg",
        relative_start_time: Optional[float] = None,
        extra_tags: Dict[str, Any] = {},
    ):
        """
        Initialize an ImageData object.

        Args:
            data (Union[np.ndarray, VideoFrame, Image.Image, bytes]): The image data.
            width (int): The width of the image in pixels. Defaults to 640.
            height (int): The height of the image in pixels. Defaults to 480.
            frame_rate (int): The frame rate of the video. Defaults to 30.
            format (str): The image format (e.g., 'jpeg', 'png'). Defaults to 'jpeg'.
            relative_start_time (Optional[float]): The relative start time of the image.
                                                   If None, uses the current playback time.

        Raises:
            ValueError: If the data is not of a supported type.
        """
        if not isinstance(data, (np.ndarray, VideoFrame, Image.Image, bytes)):
            raise ValueError("VideoData data must be np.ndarray, av.VideoFrame, PIL.Image.Image or bytes")
        self.data: Union[np.ndarray, VideoFrame, Image.Image, bytes] = data
        self.width: int = width
        self.height: int = height
        self.frame_rate: int = frame_rate
        self.format: str = format
        self.extra_tags: Dict[str, Any] = extra_tags
        self.relative_start_time: float = relative_start_time or Clock.get_playback_time()

    def get_pts(self) -> int:
        """
        Get the presentation timestamp (pts) of the video frame.

        Returns:
            int: The pts value.
        """
        return int(self.relative_start_time * self.frame_rate)

    def get_frame(self) -> VideoFrame:
        """
        Convert the image data to a VideoFrame.

        Returns:
            VideoFrame: The image data as a VideoFrame object.

        Raises:
            ValueError: If the data format is invalid or unsupported.
        """
        if isinstance(self.data, bytes):
            pil_image = Image.open(io.BytesIO(self.data), formats=[self.format])
            image_frame = VideoFrame.from_image(pil_image)
            image_frame.pts = self.get_pts()
            image_frame.time_base = fractions.Fraction(1, self.frame_rate)
            return image_frame
        elif isinstance(self.data, np.ndarray):
            return VideoFrame.from_ndarray(self.data, format="rgb24")
        elif isinstance(self.data, Image.Image):
            return VideoFrame.from_image(self.data)
        elif isinstance(self.data, VideoFrame):
            return self.data
        else:
            raise ValueError("VideoData data must be bytes, np.ndarray, PIL.Image.Image, or av.VideoFrame")

    def get_duration_seconds(self) -> float:
        """
        Calculate the duration of the video frame in seconds.

        Returns:
            float: The duration of the video frame in seconds.
        """
        return 1.0 / self.frame_rate


class TextData:
    """
    A class to handle text data with timing information.

    This class provides a structure for storing text data along with
    absolute and relative timing information.
    """

    def __init__(
        self,
        data: Optional[str] = None,
        absolute_time: Optional[float] = None,
        relative_time: Optional[float] = None,
    ):
        """
        Initialize a TextData object.

        Args:
            data (Optional[str]): The text data. Defaults to None.
            absolute_time (Optional[float]): The absolute time of the text data.
                                             If None, uses the current time.
            relative_time (Optional[float]): The relative time of the text data.
                                             Defaults to 0.0.

        Raises:
            ValueError: If the data is not of type str.
        """
        if data is not None and not isinstance(data, str):
            raise ValueError("TextData data must be str")
        self.data: Optional[str] = data
        self.absolute_time: float = absolute_time or time.time()
        self.relative_time: float = relative_time or 0.0


class SessionData:
    def __init__(self, session_id: str = None, start_time: float = None):
        self.session_id: str = session_id or str(uuid.uuid4())
        self.start_time: float = start_time or time.time()
