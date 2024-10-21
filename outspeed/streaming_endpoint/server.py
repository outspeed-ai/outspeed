import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Dict

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import HTTPException
from aiortc.rtcrtpsender import RTCRtpSender

from outspeed.server import RealtimeServer
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


def offer(audio_driver, video_driver, text_driver, audio_codec: str, video_codec: str):
    async def handshake(params: Dict[str, str]):
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        pc_id = "PeerConnection(%s)" % uuid.uuid4()
        pcs.add(pc)
        RealtimeServer().add_connection()

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


def create_and_run_server(
    audio_driver,
    video_driver,
    text_driver,
    audio_codec: str,
    video_codec: str,
):
    fastapi_app = RealtimeServer().get_app()
    fastapi_app.add_api_route(
        "/offer", offer(audio_driver, video_driver, text_driver, audio_codec, video_codec), methods=["POST"]
    )
    fastapi_app.add_event_handler("shutdown", on_shutdown)
