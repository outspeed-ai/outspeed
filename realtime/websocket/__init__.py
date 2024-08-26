import asyncio
import functools
import logging
from realtime.server import RealtimeServer
from fastapi import WebSocket
from realtime.streams import AudioStream, TextStream, VideoStream, ByteStream
import inspect
from realtime.websocket.processors import WebsocketInputStream, WebsocketOutputStream


logger = logging.getLogger(__name__)


def websocket(path: str = "/"):
    """
    Decorator for handling WebSocket connections.
    TODO: Add video support
    """

    def decorator(func):
        async def websocket_handler(websocket: WebSocket):
            RealtimeServer().add_connection()
            try:
                await websocket.accept()
                audio_metadata = await websocket.receive_json()

                audio_input_q = None
                video_input_q = None
                text_input_q = None
                kwargs = {}

                signature = inspect.signature(func)
                parameters = signature.parameters
                for name, param in parameters.items():
                    if param.annotation == AudioStream:
                        audio_input_q = AudioStream(sample_rate=audio_metadata.get("sampleRate", 48000))
                        kwargs[name] = audio_input_q
                    elif param.annotation == VideoStream:
                        video_input_q = VideoStream()
                        kwargs[name] = video_input_q
                    elif param.annotation == TextStream:
                        text_input_q = TextStream()
                        kwargs[name] = text_input_q

                output_streams = await func(**kwargs)

                if not isinstance(output_streams, (list, tuple)):
                    output_streams = (output_streams,)

                aq, vq, tq = None, None, None
                for s in output_streams:
                    if isinstance(s, AudioStream):
                        aq = s
                    elif isinstance(s, VideoStream):
                        vq = s
                    elif isinstance(s, TextStream):
                        tq = s
                    elif isinstance(s, ByteStream):
                        bq = s

                # TODO: Update the default sample rate to be consistent across all plugins
                tasks = [
                    WebsocketInputStream(websocket, audio_metadata.get("sampleRate", 48000)).run(
                        audio_stream=audio_input_q, message_stream=text_input_q, video_stream=video_input_q
                    ),
                    WebsocketOutputStream(websocket).run(
                        audio_stream=aq, message_stream=tq, video_stream=vq, byte_stream=bq
                    ),
                ]

                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                logging.error("websocket: CancelledError")
            except Exception as e:
                logging.error("websocket: Error in websocket: ", e)
            finally:
                logging.info("websocket: Removing connection")
                RealtimeServer().remove_connection()

        fastapi_app = RealtimeServer().get_app()
        fastapi_app.websocket(path)(websocket_handler)

    return decorator
