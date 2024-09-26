import asyncio
import logging
import os
from typing import Optional

import aiohttp
import regex  # Note: We use the 'regex' module, not the built-in 're'

from outspeed.plugins.base_plugin import Plugin
from outspeed.streams import TextStream

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)

class VisemeToAudio(Plugin):
    def __init__(
        self,
        api_key: Optional[str] = None,
        language_code: str = "en-US",
        tts_voice: str = "en-US-Standard-H",
        tts_rate: float = 0.9,
        tts_pitch: float = 0
    ):
        self.input_queue = None
        self._api_key = api_key or os.getenv("GOOGLE_TTS_KEY")
        self.output_queue = TextStream()
        self._language_code = language_code
        self._tts_voice = tts_voice
        self._tts_rate = tts_rate
        self._tts_pitch = tts_pitch
        self.session: Optional[aiohttp.ClientSession] = None
        self._task: Optional[asyncio.Task] = None

    def run(self, input_queue: TextStream) -> TextStream:
        self.input_queue = input_queue
        self._task = asyncio.create_task(self.viseme_to_audio())
        return self.output_queue

    async def viseme_to_audio(self):
        try:
            async with aiohttp.ClientSession() as self.session:
                while True:
                    viseme_obj = await self.input_queue.get()
                    # Print and log the viseme object
                    logger.info(f"Viseme object: {viseme_obj}")
                    if "text" in viseme_obj:
                        ssml = "<speak>"
                        for i, x in enumerate(viseme_obj["text"]):
                            # Add mark
                            if i > 0:
                                ssml += f" <mark name='{x['mark']}'/>"

                            # Add word
                            word = x["word"]
                            word = word.replace("&", "&amp;")
                            word = word.replace("<", "&lt;")
                            word = word.replace(">", "&gt;")
                            word = word.replace('"', "&quot;")
                            word = word.replace("'", "&apos;")
                            word = regex.sub(r"^\p{Dash_Punctuation}$", '<break time="750ms"/>', word)
                            ssml += word

                        ssml += "</speak>"

                        headers = {"Content-Type": "application/json; charset=utf-8"}
                        body = {
                            "input": {"ssml": ssml},
                            "voice": {
                                "languageCode": self._language_code,
                                "name": self._tts_voice,
                            },
                            "audioConfig": {
                                "audioEncoding": "OGG-OPUS",
                                "speakingRate": self._tts_rate,
                                "pitch": self._tts_pitch,
                                "volumeGainDb": 0,
                            },
                            "enableTimePointing": [1],  # Timepoint information for mark tags
                        }
                        url = f"https://texttospeech.googleapis.com/v1beta1/text:synthesize?key={self._api_key}"
                        async with self.session.post(url, json=body, headers=headers) as r:
                            if r.status != 200:
                                logger.error("TTS error %s", await r.text())
                                return
                            data = await r.json()
                            viseme_obj["data"] = data
                    await self.output_queue.put(viseme_obj)
        except Exception as e:
            logger.error(f"Error in viseme_to_audio: {e}")
            raise asyncio.CancelledError
                # return line
