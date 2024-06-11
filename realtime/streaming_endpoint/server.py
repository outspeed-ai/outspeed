import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Dict

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from realtime.streaming_endpoint.audio_frame_output import AudioOutputFrameProcessor
from realtime.streaming_endpoint.text_frame_output import TextFrameOutputProcessor
from realtime.streaming_endpoint.video_frame_output import VideoOutputFrameProcessor

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
logger.setLevel(logging.DEBUG)
pcs = asyncio.Queue()
relay = MediaRelay()


async def offer(params: Dict[str, str]):
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    await pcs.put(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    # log_info("Created for %s", params["remote"])

    @pc.on("datachannel")
    def on_datachannel(channel):
        text_output_processor = TextFrameOutputProcessor(channel)

        @channel.on("message")
        def on_message(message):
            text_output_processor.put_text(message)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # For some unknown reason, making this funciton async breaks aiortc
    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        # avoid using relay.subscribe since it doesn't close properly when connection is stopped
        if track.kind == "audio":
            AudioOutputFrameProcessor.track = track
        elif track.kind == "video":
            pc.addTrack(VideoOutputFrameProcessor(track))

    # handle offer
    await pc.setRemoteDescription(offer)
    pc.addTrack(AudioOutputFrameProcessor())

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@asynccontextmanager
async def on_shutdown(app: FastAPI):
    yield
    # close peer connections
    # coros = [pc.close() for pc in pcs]
    # await asyncio.gather(*coros)
    # pcs.clear()


def create_and_run_server():
    webrtc_app = FastAPI(lifespan=on_shutdown)
    webrtc_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    webrtc_app.add_api_route("/offer", offer, methods=["POST"])
    return webrtc_app
