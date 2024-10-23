import base64
import fractions
import io
import json
import time
import uuid
from typing import Any, Dict, Optional, Union

import numpy as np
from av import AudioFrame, VideoFrame
from PIL import Image
from pydub import AudioSegment

from outspeed.utils.clock import Clock
from outspeed.utils.images import convert_yuv420_to_pil


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

        if isinstance(data, AudioFrame) and data.format.name != "s16":
            raise ValueError(f"AudioData only supports s16 format. Received: {data.format}")

        self.data: Union[bytes, AudioFrame] = data
        self._sample_rate: int = sample_rate
        self._channels: int = channels
        self._sample_width: int = sample_width
        self.format: str = format
        self.extra_tags: Dict[str, Any] = extra_tags
        self.relative_start_time: float = relative_start_time or Clock.get_playback_time()

    @property
    def sample_rate(self) -> int:
        if isinstance(self.data, AudioFrame):
            return self.data.sample_rate
        else:
            return self._sample_rate

    @property
    def channels(self) -> int:
        if isinstance(self.data, AudioFrame):
            if self.data.layout.name == "stereo":
                return 2
            else:
                return 1
        else:
            return self._channels

    @property
    def sample_width(self) -> int:
        if isinstance(self.data, AudioFrame):
            if self.data.format.name == "s16":
                return 2
            else:
                raise ValueError("Unsupported audio format")
        else:
            return self._sample_width

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
            else:
                raise ValueError("AudioData format must be wav")

            # Create AudioFrame from numpy array
            frame = AudioFrame.from_ndarray(array, format=format, layout=channel_layout)
            frame.sample_rate = self.sample_rate
            frame.pts = self.get_pts()
            frame.time_base = fractions.Fraction(1, self.sample_rate)
            return frame
        else:
            raise ValueError("AudioData data must be bytes or av.AudioFrame")

    def resample(self, sample_rate: int, channels: int = 1) -> "AudioData":
        """
        Resample the audio data to a new sample rate.

        Args:
            sample_rate (int): The new sample rate in Hz.

        Returns:
            AudioData: The resampled audio data.
        """
        if self.sample_rate == sample_rate and self.channels == channels:
            return self

        audio_segment = AudioSegment(
            data=self.get_bytes(),
            sample_width=self.sample_width,
            frame_rate=self.sample_rate,
            channels=self.channels,
        )

        # Resample the audio to 16000 Hz
        resampled_audio = audio_segment.set_frame_rate(sample_rate)

        # Change the number of channels
        resampled_audio = resampled_audio.set_channels(channels)

        # Convert the resampled audio back to bytes
        data = resampled_audio.raw_data

        return AudioData(
            data, sample_rate, channels, self.sample_width, self.format, self.relative_start_time, self.extra_tags
        )

    def change_volume(self, percentage: float) -> "AudioData":
        """
        Change the volume to a percentage of the original volume.

        Args:
            percentage (float): The desired volume as a percentage (e.g., 50 for 50%).

        Returns:
            AudioData: A new instance with the adjusted volume.

        Raises:
            ValueError: If the percentage is not between 0 and 100.
        """
        if percentage == 1:
            return self

        if not 0 < percentage < 1:
            raise ValueError("Percentage must be between 0 and 100")

        # Create an AudioSegment from the current audio data
        audio_segment = AudioSegment(
            data=self.get_bytes(), sample_width=self.sample_width, frame_rate=self.sample_rate, channels=self.channels
        )

        # Calculate gain in dB
        change_db = 20 * np.log10(percentage)
        adjusted_audio = audio_segment.apply_gain(change_db)

        return AudioData(
            data=adjusted_audio.raw_data,
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_width=self.sample_width,
            format=self.format,
            relative_start_time=self.relative_start_time,
            extra_tags=self.extra_tags,
        )


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
            image_frame = VideoFrame.from_image(self.data)
            image_frame.pts = self.get_pts()
            image_frame.time_base = fractions.Fraction(1, self.frame_rate)
            return image_frame
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

    def get_pil_image(self) -> Image.Image:
        """
        Convert the image data to a PIL Image.

        Returns:
            Image.Image: The image data as a PIL Image object.
        """
        if isinstance(self.data, bytes):
            pil_image = Image.open(io.BytesIO(self.data), formats=[self.format])
            return pil_image
        elif isinstance(self.data, np.ndarray):
            return Image.fromarray(self.data)
        elif isinstance(self.data, Image.Image):
            return self.data
        elif isinstance(self.data, VideoFrame):
            return convert_yuv420_to_pil(self.data)
        else:
            raise ValueError("VideoData data must be bytes, np.ndarray, PIL.Image.Image, or av.VideoFrame")

    def get_bytes(self, quality: int = 100) -> bytes:
        """
        Convert the image data to bytes.

        Returns:
            bytes: The image data as bytes.

        Raises:
            ValueError: If the data is not of type bytes or VideoFrame.
        """
        if isinstance(self.data, bytes):
            return self.data
        elif isinstance(self.data, VideoFrame):
            return self.data.to_ndarray().tobytes()
        elif isinstance(self.data, Image.Image):
            with io.BytesIO() as buffer:
                self.data.save(buffer, format=self.format, quality=quality)
                return buffer.getvalue()
        else:
            raise ValueError("VideoData data must be bytes or av.VideoFrame")

    def get_base64_url(self, quality: int = 100) -> str:
        """
        Encode the image data to base64.

        Returns:
                str: The base64 encoded image data as a string.
        """
        return f"data:image/{self.format};base64,{base64.b64encode(self.get_bytes(quality)).decode()}"


class TextData:
    """
    A class to handle text data with timing information.

    This class provides a structure for storing text data along with
    absolute and relative timing information.
    """

    def __init__(
        self,
        data: str,
        absolute_time: Optional[float] = None,
        relative_time: Optional[float] = None,
        extra_tags: Dict[str, Any] = {},
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
        if not isinstance(data, str):
            raise ValueError("TextData data must be str")
        self.data: str = data
        self.absolute_time: float = absolute_time or time.time()
        self.relative_time: float = relative_time or 0.0
        self.extra_tags: Dict[str, Any] = extra_tags

    def get_text(self) -> str:
        return self.data

    def get_json(self) -> Dict[str, Any]:
        try:
            return json.loads(self.data)
        except json.JSONDecodeError:
            raise ValueError(f"TextData data is not valid JSON: {self.data}")

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "TextData":
        try:
            return cls(json.dumps(data))
        except json.JSONDecodeError:
            raise ValueError(f"TextData data is not valid JSON: {data}")


class SessionData:
    def __init__(self, session_id: str = None, start_time: float = None):
        self.session_id: str = session_id or str(uuid.uuid4())
        self.start_time: float = start_time or time.time()
