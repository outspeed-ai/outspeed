import asyncio
import functools
import inspect
import logging
import traceback
from typing import Callable, List, Optional, Tuple, Union

from outspeed._realtime_function import RealtimeFunction
from outspeed.server import RealtimeServer
from outspeed.streaming_endpoint.AudioRTCDriver import AudioRTCDriver
from outspeed.streaming_endpoint.server import create_webrtc_offer_endpoint
from outspeed.streaming_endpoint.TextRTCDriver import TextRTCDriver
from outspeed.streaming_endpoint.VideoRTCDriver import VideoRTCDriver
from outspeed.streams import AudioStream, TextStream, VideoStream
from outspeed.utils import tracing
from outspeed.utils.audio import AudioCodec
from outspeed.utils.images import VideoCodec

logger = logging.getLogger(__name__)


def streaming_endpoint(audio_codec: str = AudioCodec.OPUS, video_codec: str = VideoCodec.H264) -> Callable:
    """
    Decorator for creating a streaming endpoint.

    This decorator wraps a function to set up and manage audio, video, and text streams
    for real-time communication.

    Returns:
        Callable: A decorator function.
    """

    def decorator(func: Callable) -> Callable:
        # Create and run the server
        create_webrtc_offer_endpoint(
            func,
            audio_codec,
            video_codec,
        )
        rt_func = RealtimeFunction(func)
        return rt_func

    return decorator
