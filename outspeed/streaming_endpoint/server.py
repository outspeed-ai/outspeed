import asyncio
import inspect
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Callable, Dict, Optional

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import HTTPException
from aiortc.rtcrtpsender import RTCRtpSender

from outspeed.server import RealtimeServer
from outspeed.streams import AudioStream, TextStream, VideoStream
from outspeed.utils.audio import AudioCodec
from outspeed.utils.images import VideoCodec

ROOT = os.path.dirname(__file__)

logger = logging.getLogger(__name__)

pcs = set()


def force_codec(pc, sender, forced_codec: str):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences([codec for codec in codecs if codec.mimeType.lower() == forced_codec.lower()])


async def process_func(func: Callable):
    audio_input_q: Optional[AudioStream] = None
    video_input_q: Optional[VideoStream] = None
    text_input_q: Optional[TextStream] = None

    # Inspect the function signature and set up input streams
    signature = inspect.signature(func)
    parameters = signature.parameters
    args = []
    kwargs = {}
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

    # Call the wrapped function and get output streams
    output_streams = await func(*args, **kwargs)
    # Ensure output_streams is iterable
    if not isinstance(output_streams, (list, tuple)):
        output_streams = (output_streams,)

    # Initialize output queues
    audio_output_queue, video_output_queue, text_output_queue = None, None, None
    for s in output_streams:
        if isinstance(s, AudioStream):
            audio_output_queue = s
        elif isinstance(s, VideoStream):
            video_output_queue = s
        elif isinstance(s, TextStream):
            text_output_queue = s

    # Set up RTC drivers for each stream type
    video_output_frame_processor = VideoRTCDriver(video_input_q, video_output_queue)
    # TODO: get audio_output_layout, audio_output_format, audio_output_sample_rate from SDP
    audio_output_frame_processor = AudioRTCDriver(audio_input_q, audio_output_queue)
    text_output_processor = TextRTCDriver(text_input_q, text_output_queue)

    # Create and run tasks for each processor
    tasks = [
        asyncio.create_task(video_output_frame_processor.run_input()),
        asyncio.create_task(audio_output_frame_processor.run()),
        asyncio.create_task(text_output_processor.run_input()),
    ]
    await asyncio.gather(*tasks)

    return audio_output_frame_processor, video_output_frame_processor, text_output_processor


def offer(func: Callable, audio_codec: str, video_codec: str):
    async def handshake(params: Dict[str, str]):
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        pc_id = "PeerConnection(%s)" % uuid.uuid4()
        pcs.add(pc)
        RealtimeServer().add_connection()

        text_driver, audio_driver, video_driver = await process_func(func)

        def log_info(msg, *args):
            logger.debug(pc_id + " " + msg, *args)

        # log_info("Created for %s", params["remote"])

        @pc.on("datachannel")
        def on_datachannel(channel):
            text_driver.add_track(channel)

            @channel.on("message")
            def on_message(message):
                text_driver.put_text(message)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log_info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                RealtimeServer().remove_connection()
                pcs.discard(pc)

        # For some unknown reason, making this funciton async breaks aiortc
        @pc.on("track")
        def on_track(track):
            log_info("Track %s received", track.kind)

            # avoid using relay.subscribe since it doesn't close properly when connection is stopped
            if track.kind == "audio":
                if not audio_driver.audio_input_q:
                    raise HTTPException(
                        status_code=400,
                        detail="Please check that the proper Audio and Video settings are enabled.",
                    )

                audio_driver.add_track(track)
            elif track.kind == "video":
                if not video_driver.video_input_q:
                    raise HTTPException(
                        status_code=400,
                        detail="Please check that the proper Audio and Video settings are enabled.",
                    )

                video_driver.add_track(track)

        try:
            # handle offer
            await pc.setRemoteDescription(offer)
            if video_driver.video_output_q:
                video_sender = pc.addTrack(video_driver)
                force_codec(pc, video_sender, video_codec)
            if audio_driver.audio_output_q:
                audio_sender = pc.addTrack(audio_driver)
                force_codec(pc, audio_sender, audio_codec)

            # send answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
        except Exception as e:
            logger.error(
                "Please check that the proper Audio and Video settings are enabled. Error handling offer: %s", e
            )
            await pc.close()
            pcs.discard(pc)
            raise HTTPException(
                status_code=400,
                detail="Please check that the proper Audio and Video settings are enabled.",
            )
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    return handshake


async def get_active_connection_ids():
    return {"connections": [str(pc) for pc in pcs]}


@asynccontextmanager
async def on_shutdown():
    yield
    # close peer connections
    logger.info("Closing peer connections")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


def create_webrtc_offer_endpoint(
    func: Callable,
    audio_codec: str,
    video_codec: str,
):
    fastapi_app = RealtimeServer().get_app()
    fastapi_app.add_api_route("/offer", offer(func, audio_codec, video_codec), methods=["POST"])
    fastapi_app.add_event_handler("shutdown", on_shutdown)
