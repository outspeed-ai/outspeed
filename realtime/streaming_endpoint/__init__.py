import asyncio
import functools
import inspect
import logging
import os
import time

import uvicorn

from realtime.streaming_endpoint.audio_frame_output import AudioOutputFrameProcessor
from realtime.streaming_endpoint.server import create_and_run_server
from realtime.streaming_endpoint.text_frame_output import TextFrameOutputProcessor
from realtime.streaming_endpoint.video_frame_output import VideoOutputFrameProcessor
from realtime.streams import AudioStream, TextStream, VideoStream

logger = logging.getLogger(__name__)


def streaming_endpoint():
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Add the extra argument
            video_output_frame_processor = VideoOutputFrameProcessor()
            audio_output_frame_processor = AudioOutputFrameProcessor()
            text_output_processor = TextFrameOutputProcessor()

            webrtc_app = create_and_run_server()

            HOSTNAME = "0.0.0.0"
            PORT = int(os.getenv("HTTP_PORT", 8080))
            server = uvicorn.Server(config=uvicorn.Config(webrtc_app, host=HOSTNAME, port=PORT, log_level="info"))
            asyncio.create_task(server.serve())

            while True:
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

                    async def audio_frame_callback():
                        while not audio_output_frame_processor.track:
                            await asyncio.sleep(0.2)
                        audio_track = audio_output_frame_processor.track
                        while True:
                            try:
                                frame = await audio_track.recv()
                                await audio_input_q.put(frame)
                            except:
                                print("Error in audio_frame_callback")
                                raise asyncio.CancelledError

                    async def video_frame_callback():
                        while not video_output_frame_processor.track:
                            await asyncio.sleep(0.2)
                        video_track = video_output_frame_processor.track
                        while True:
                            try:
                                frame = await video_track.recv()
                                await video_input_q.put(frame)
                            except:
                                print("Error in video_frame_callback")
                                raise asyncio.CancelledError

                    async def text_frame_callback():
                        while not text_output_processor.track:
                            await asyncio.sleep(0.2)
                        text_track = text_output_processor.track
                        while True:
                            try:
                                text = await text_track.recv()
                                await text_input_q.put(text)
                            except:
                                print("Error in text_frame_callback")
                                raise asyncio.CancelledError

                    async def process_audio_output():
                        _start = None
                        data_time = 0.020
                        while True:
                            if not aq:
                                break
                            if _start is None:
                                _start = time.time() + data_time
                            else:
                                wait = _start - time.time() - 0.005
                                _start += data_time
                                if wait > 0:
                                    await asyncio.sleep(wait)
                            try:
                                audio_frame = aq.get_nowait()
                                if audio_frame is None:
                                    continue
                                await audio_output_frame_processor.put_frame(audio_frame)
                            except:
                                pass  # Ignore Empty exception

                            # await audio_output_frame_processor.put_frame(generate_silence_packet())

                    async def process_video_output():
                        _start = None
                        data_time = 0.020
                        while True:
                            if not vq:
                                break
                            if _start is None:
                                _start = time.time() + data_time
                            else:
                                wait = _start - time.time() - 0.005
                                _start += data_time
                                if wait > 0:
                                    await asyncio.sleep(wait)
                            try:
                                video_frame = vq.get_nowait()
                                if video_frame is None:
                                    continue
                                await video_output_frame_processor.put_frame(video_frame)
                            except:
                                pass  # Ignore Empty exception

                    async def process_text_output():
                        while True:
                            if not tq:
                                break
                            try:
                                text_frame = await tq.get()
                                if text_frame is None:
                                    continue
                                await text_output_processor.send(text_frame)
                            except:
                                pass  # Ignore Empty exception

                    tasks = [
                        asyncio.create_task(process_audio_output()),
                        asyncio.create_task(process_video_output()),
                        asyncio.create_task(process_text_output()),
                    ]
                    if audio_input_q:
                        tasks.append(asyncio.create_task(audio_frame_callback()))
                    if video_input_q:
                        tasks.append(asyncio.create_task(video_frame_callback()))
                    if text_input_q:
                        tasks.append(asyncio.create_task(text_frame_callback()))
                    await asyncio.gather(*tasks)
                except asyncio.CancelledError:
                    VideoOutputFrameProcessor.track = None
                    AudioOutputFrameProcessor.track = None
                    TextFrameOutputProcessor.track = None
                    if instance:
                        await instance.teardown()
                    for task in tasks:
                        task.cancel()
                    await asyncio.sleep(1)
                    logger.info("Restarting streaming endpoint")

        return wrapper

    return decorator
