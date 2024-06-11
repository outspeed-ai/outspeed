from __future__ import annotations

import asyncio
import json
import logging
import os
from urllib.parse import urlencode

import aiohttp

from realtime.plugins.base_plugin import Plugin
from realtime.streams import TextStream

_KEEPALIVE_MSG: str = json.dumps({"type": "KeepAlive"})
_CLOSE_MSG: str = json.dumps({"type": "CloseStream"})


logger = logging.getLogger(__name__)


class DeepgramSTT(Plugin):
    def __init__(
        self,
        *,
        language="en-US",
        detect_language: bool = False,
        interim_results: bool = True,
        punctuate: bool = True,
        smart_format: bool = True,
        model="nova-2",
        api_key: str | None = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        min_silence_duration: int = 100,
        confidence_threshold=0.8,
    ) -> None:
        api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if api_key is None:
            raise ValueError("Deepgram API key is required")
        self._api_key = api_key
        self.language = language
        self.detect_language = detect_language
        self.interim_results = interim_results
        self.punctuate = punctuate
        self.smart_format = smart_format
        self.model = model
        self.min_silence_duration = min_silence_duration
        self.endpointing = min_silence_duration

        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._speaking = False
        self.confidence_threshold = confidence_threshold

        self._session = aiohttp.ClientSession()

        self._closed = False
        self.output_queue = TextStream()

    async def close(self):
        self.input_queue.put_nowait(_CLOSE_MSG)
        await asyncio.sleep(0.2)

        await self._session.close()
        self._task.cancel()

    async def run(self, input_queue) -> None:
        try:
            self.input_queue = input_queue
            self._task = asyncio.create_task(self._run_ws())
            return self.output_queue
        except Exception:
            logger.error("deepgram task failed")

    async def _run_ws(self) -> None:
        live_config = {
            "model": self.model,
            "punctuate": self.punctuate,
            "smart_format": self.smart_format,
            "encoding": "linear16",
            "sample_rate": self._sample_rate,
            "channels": self._num_channels,
            "endpointing": self.endpointing,
        }

        live_config["language"] = self.language

        headers = {"Authorization": f"Token {self._api_key}"}

        url = f"wss://api.deepgram.com/v1/listen?{urlencode(live_config).lower()}"
        ws = await self._session.ws_connect(url, headers=headers)

        async def keepalive_task():
            # if we want to keep the connection alive even if no audio is sent,
            # Deepgram expects a keepalive message.
            # https://developers.deepgram.com/reference/listen-live#stream-keepalive
            try:
                while True:
                    await ws.send_str(_KEEPALIVE_MSG)
                    await asyncio.sleep(5)
            except Exception:
                pass

        async def send_task():
            while True:
                data = await self.input_queue.get()

                if data == _CLOSE_MSG:
                    self._closed = True
                    await ws.send_str(data)
                    break

                bytes = data.to_ndarray().tobytes()
                await ws.send_bytes(bytes)

        async def recv_task():
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if self._closed:
                        return

                    raise Exception("deepgram connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.error("unexpected deepgram message type %s", msg.type)
                    continue

                try:
                    data = json.loads(msg.data)
                    if not "is_final" in data:
                        break
                    is_final = data["is_final"]
                    top_choice = data["channel"]["alternatives"][0]
                    confidence = top_choice["confidence"]
                    if top_choice["transcript"] and confidence > self.confidence_threshold and is_final:
                        logger.info("deepgram transcript: %s", top_choice["transcript"])
                        await self.output_queue.put(top_choice["transcript"])
                except Exception as e:
                    logger.error("failed to process deepgram message %s", e)

        await asyncio.gather(send_task(), recv_task(), keepalive_task())
