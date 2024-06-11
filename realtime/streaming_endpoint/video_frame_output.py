import asyncio

from aiortc import MediaStreamTrack


class VideoOutputFrameProcessor(MediaStreamTrack):
    kind = "video"
    video_output_frame_q = asyncio.Queue()
    track = None

    def __init__(self, track=None):
        super().__init__()
        if track:
            VideoOutputFrameProcessor.track = track

    async def recv(self):
        frame = await VideoOutputFrameProcessor.video_output_frame_q.get()
        return frame

    async def put_frame(self, frame):
        await VideoOutputFrameProcessor.video_output_frame_q.put(frame)
