import asyncio
import functools
import inspect
import logging
import traceback
from typing import Callable, List, Optional, Tuple, Union

from outspeed._realtime_function import RealtimeFunction
from outspeed.server import RealtimeServer
from outspeed.streaming_endpoint.AudioRTCDriver import AudioRTCDriver
from outspeed.streaming_endpoint.server import create_and_run_server
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
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> None:
            try:
                # Initialize input queues
                audio_input_q: Optional[AudioStream] = None
                video_input_q: Optional[VideoStream] = None
                text_input_q: Optional[TextStream] = None

                # Inspect the function signature and set up input streams
                signature = inspect.signature(func)
                parameters = signature.parameters
                for name, param in parameters.items():
                    if param.annotation == AudioStream:
                        audio_input_q = AudioStream()
                        kwargs[name] = audio_input_q
                    elif param.annotation == VideoStream:
                        video_input_q = VideoStream()
                        kwargs[name] = video_input_q
                    elif param.annotation == TextStream:
                        text_input_q = TextStream()
                        kwargs[name] = text_input_q

                tracing.start()
                # Call the wrapped function and get output streams
                output_streams = await func(*args, **kwargs)
                # Ensure output_streams is iterable
                if not isinstance(output_streams, (list, tuple)):
                    output_streams = (output_streams,)

                # Initialize output queues
                aq, vq, tq = None, None, None
                for s in output_streams:
                    if isinstance(s, AudioStream):
                        aq = s
                    elif isinstance(s, VideoStream):
                        vq = s
                    elif isinstance(s, TextStream):
                        tq = s

                # Set up RTC drivers for each stream type
                video_output_frame_processor = VideoRTCDriver(video_input_q, vq)
                # TODO: get audio_output_layout, audio_output_format, audio_output_sample_rate from SDP
                audio_output_frame_processor = AudioRTCDriver(audio_input_q, aq)
                text_output_processor = TextRTCDriver(text_input_q, tq)

                # Create and run the server
                create_and_run_server(
                    audio_output_frame_processor,
                    video_output_frame_processor,
                    text_output_processor,
                    audio_codec,
                    video_codec,
                )

                # Create and run tasks for each processor
                tasks = [
                    asyncio.create_task(video_output_frame_processor.run_input()),
                    asyncio.create_task(audio_output_frame_processor.run()),
                    asyncio.create_task(text_output_processor.run_input()),
                ]
                await asyncio.gather(*tasks)

            except asyncio.CancelledError:
                pass
            except Exception as e:
                logging.error(f"Error in streaming_endpoint: {e}")
                logging.error(f"Traceback:\n{traceback.format_exc()}")
            finally:
                RealtimeServer().remove_connection()
                tracing.end()
                logging.error("Streaming endpoint disconnected")

                raise asyncio.CancelledError

        rt_func = RealtimeFunction(wrapper)
        return rt_func

    return decorator
