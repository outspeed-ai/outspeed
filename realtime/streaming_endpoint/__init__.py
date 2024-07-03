import asyncio
import functools
import inspect
import logging
import os
import ssl
import time

import uvicorn

from realtime.streaming_endpoint.AudioRTCDriver import AudioRTCDriver
from realtime.streaming_endpoint.server import create_and_run_server
from realtime.streaming_endpoint.TextRTCDriver import TextRTCDriver
from realtime.streaming_endpoint.VideoRTCDriver import VideoRTCDriver
from realtime.streams import AudioStream, TextStream, VideoStream

logger = logging.getLogger(__name__)


def streaming_endpoint():
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                audio_input_q = None
                video_input_q = None
                text_input_q = None
                instance = None

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
                instance = args[0]
                await instance.setup()
                output_streams = await func(*args, **kwargs)
                # Ensure output_streams is iterable
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

                video_output_frame_processor = VideoRTCDriver(video_input_q, vq)
                audio_output_frame_processor = AudioRTCDriver(audio_input_q, aq)
                text_output_processor = TextRTCDriver(text_input_q, tq)
                webrtc_app = create_and_run_server(
                    audio_output_frame_processor, video_output_frame_processor, text_output_processor
                )

                HOSTNAME = "0.0.0.0"
                PORT = int(os.getenv("HTTP_PORT", 8080))
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ssl_context.load_cert_chain(os.environ["SSL_CERT_PATH"], keyfile=os.environ["SSL_KEY_PATH"])
                server = uvicorn.Server(
                    config=uvicorn.Config(
                        webrtc_app,
                        host=HOSTNAME,
                        port=PORT,
                        log_level="info",
                        ssl_keyfile=os.environ["SSL_KEY_PATH"],
                        ssl_certfile=os.environ["SSL_CERT_PATH"],
                    ),
                )
                server_task = asyncio.create_task(server.serve())

                tasks = [
                    asyncio.create_task(video_output_frame_processor.run_input()),
                    asyncio.create_task(audio_output_frame_processor.run_input()),
                    asyncio.create_task(text_output_processor.run_input()),
                ]
                await asyncio.gather(*tasks)

            except asyncio.CancelledError:
                print("streaming_endpoint: CancelledError")
            except Exception as e:
                print("Error in streaming_endpoint: ", e)
            finally:
                logging.info("Received exit, stopping bot")
                if instance:
                    await instance.teardown()
                try:
                    await server.shutdown()
                except Exception as e:
                    logging.info("Error in server.shutdown: ", e)
                loop = asyncio.get_event_loop()
                tasks = asyncio.all_tasks(loop)
                for task in tasks:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        logging.info("Task was cancelled")

        return wrapper

    return decorator
