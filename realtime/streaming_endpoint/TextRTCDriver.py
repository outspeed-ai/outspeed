

class TextRTCDriver:
    kind = "text"

    def __init__(self, text_input_q, text_output_q):
        self.text_input_q = text_input_q
        self.text_output_q = text_output_q
        self._track = None

    def put_text(self, text):
        if self.text_input_q:
            self.text_input_q.put_nowait(text)

    def add_track(self, track):
        self._track = track

    async def run_input(self):
        if not self.text_output_q:
            return
        while True:
            text = await self.text_output_q.get()
            if not self._track:
                continue
            self._track.send(text)
