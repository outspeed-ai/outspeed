import asyncio


class TextFrameOutputProcessor:
    kind = "text"
    track = None
    text_received_q = asyncio.Queue()

    def __init__(self, track=None):
        super().__init__()
        if track:
            TextFrameOutputProcessor.track = track

    async def send(self, text):
        await TextFrameOutputProcessor.track.send(text)

    def put_text(self, text):
        TextFrameOutputProcessor.text_received_q.put_nowait(text)

    async def recv(self):
        return await TextFrameOutputProcessor.text_received_q.get()
