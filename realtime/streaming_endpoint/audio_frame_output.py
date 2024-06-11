import asyncio

from aiortc import MediaStreamTrack


class AudioOutputFrameProcessor(MediaStreamTrack):
    kind = "audio"
    audio_output_frame_q = asyncio.Queue()
    track = None

    def __init__(self, track=None):
        super().__init__()
        if track:
            AudioOutputFrameProcessor.track = track

    async def recv(self):
        frame = await AudioOutputFrameProcessor.audio_output_frame_q.get()
        return frame

    async def put_frame(self, frame):
        await AudioOutputFrameProcessor.audio_output_frame_q.put(frame)
