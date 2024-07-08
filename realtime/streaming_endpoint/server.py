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

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
logger.setLevel(logging.DEBUG)
pcs = set()
# relay = MediaRelay()


def offer(audio_driver, video_driver, text_driver):
    async def handshake(params: Dict[str, str]):
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        pc_id = "PeerConnection(%s)" % uuid.uuid4()
        pcs.add(pc)

        def log_info(msg, *args):
            logger.info(pc_id + " " + msg, *args)

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
                pcs.discard(pc)

        # For some unknown reason, making this funciton async breaks aiortc
        @pc.on("track")
        def on_track(track):
            log_info("Track %s received", track.kind)

            # avoid using relay.subscribe since it doesn't close properly when connection is stopped
            if track.kind == "audio":
                audio_driver.add_track(track)
            elif track.kind == "video":
                video_driver.add_track(track)

        # handle offer
        await pc.setRemoteDescription(offer)
        if video_driver.video_output_q:
            pc.addTrack(video_driver)
        if audio_driver.audio_output_q:
            pc.addTrack(audio_driver)

        # send answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    return handshake


async def get_active_connection_ids():
    return {"connections": [str(pc) for pc in pcs]}


@asynccontextmanager
async def on_shutdown(app: FastAPI):
    yield
    # close peer connections
    logger.info("Closing peer connections")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


def create_and_run_server(audio_driver, video_driver, text_driver):
    webrtc_app = FastAPI(lifespan=on_shutdown)
    webrtc_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    webrtc_app.add_api_route("/offer", offer(audio_driver, video_driver, text_driver), methods=["POST"])
    webrtc_app.add_api_route("/connections", get_active_connection_ids, methods=["GET"])
    return webrtc_app
