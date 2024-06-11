import asyncio
import logging
import os
import time

import aiohttp

from realtime.plugins.base_plugin import Plugin
from realtime.streams import ByteStream, TextStream

logger = logging.getLogger(__name__)


class ElevenLabsTTS(Plugin):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice_id: str = "pNInz6obpgDQGcFmaJgB",
        model: str = "eleven_turbo_v2",
        output_format: str = "pcm_16000",
        optimize_streaming_latency: int = 4,
        stream: bool = True,
    ):
        super().__init__()

        api_key = api_key or os.environ.get("ELEVEN_LABS_API_KEY")
        if api_key is None:
            raise ValueError("Eleven Labs API key is required")
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._output_format = output_format
        self._optimize_streaming_latency = optimize_streaming_latency

        self.output_queue = ByteStream()
        self._generating = False

        self._stream = stream

    # async def run_tts(self, sentence) -> Iterator[bytes]:
    #     Not using elevenlabs SDK since calling the API directly is 2x faster time to first chunk
    #     audio_stream = await self.aclient.generate(
    #         text=sentence + " ",
    #         voice=self._voice_id,
    #         model=self._model,
    #         stream=True,
    #         optimize_streaming_latency=4,
    #         output_format="pcm_16000",
    #     )
    #     async for chunk in audio_stream:
    #         if chunk:
    #             yield chunk

    async def run(self, input_queue: TextStream) -> ByteStream:
        self.input_queue = input_queue
        self._task = asyncio.create_task(self.synthesize_speech())
        return self.output_queue

    async def synthesize_speech(self):
        async with aiohttp.ClientSession() as self.session:
            while True:
                text_chunk = await self.input_queue.get()
                if text_chunk is None:
                    continue
                self._generating = True
                start_time = time.time()
                logger.info("Generating TTS %s", text_chunk)
                if self._stream:
                    url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream"
                else:
                    url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}"
                payload = {"text": text_chunk, "model_id": self._model}
                querystring = {
                    "output_format": self._output_format,
                    "optimize_streaming_latency": self._optimize_streaming_latency,
                }
                headers = {
                    "xi-api-key": self._api_key,
                    "Content-Type": "application/json",
                }
                r = await self.session.post(url, json=payload, headers=headers, params=querystring)
                if r.status != 200:
                    logger.error("tts error %s", await r.text())
                    return

                logger.info("ElevenLabs TTS TTFB: %s", time.time() - start_time)
                if self._stream:
                    async for chunk in r.content:
                        if chunk:
                            self.output_queue.put_nowait(chunk)
                else:
                    self.output_queue.put_nowait(await r.read())

                self._generating = False

    async def close(self):
        await self.session.close()
        self._task.cancel()

    async def _interrupt(self):
        while True:
            user_speaking = await self.interrupt_queue.get()
            if self._generating and user_speaking:
                self._task.cancel()
                while not self.output_queue.empty():
                    self.output_queue.get_nowait()
                logger.info("Done cancelling TTS")
                self._generating = False
                self._task = asyncio.create_task(self.synthesize_speech())

    async def set_interrupt(self, interrupt_queue: asyncio.Queue):
        self.interrupt_queue = interrupt_queue
        self._interrupt_task = asyncio.create_task(self._interrupt())
