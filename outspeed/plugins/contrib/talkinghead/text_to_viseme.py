import asyncio
import logging
import os
from typing import Optional

import regex  # Note: We use the 'regex' module, not the built-in 're'

from outspeed.plugins.base_plugin import Plugin
from outspeed.plugins.contrib.talkinghead.lipsync_en import LipsyncEn
from outspeed.streams import TextStream

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

class TextToViseme(Plugin):
    def __init__(
        self,
        *,
        lipsync_lang: str = "en",
    ):
        super().__init__()
        self.input_queue = None
        self._lipsync = LipsyncEn()
        self._lipsync_lang = lipsync_lang
        self.output_queue = TextStream()

    def lip_sync_pre_process_text(self, s):
        return self._lipsync.preProcessText(s)

    def lip_sync_words_to_visemes(self, w):
        r = self._lipsync.wordsToVisemes(w)
        return r

    async def lip_sync(self):
        logger.info("Starting lip sync")
        try:

            # Regular expressions
            dividersSentence = regex.compile(r"[!\.\?\n\p{Extended_Pictographic}]", regex.U)
            dividersWord = regex.compile(r"[ ]", regex.U)
            speakables = regex.compile(r"[\p{L}\p{N},\.\'!€\$\+\p{Dash_Punctuation}%&\?]", regex.U)
            lipsync_lang = self._lipsync_lang

            while True:
                s = await self.input_queue.get()
                letters = list(s)

                markdownWord = ""  # markdown word
                textWord = ""  # text-to-speech word
                markId = 0  # SSML mark id
                ttsSentence = []  # Text-to-speech sentence
                lipsyncAnim = []  # Lip-sync animation sequence

                for i in range(len(letters)):
                    isLast = i == len(letters) - 1
                    c = letters[i]
                    isSpeakable = bool(speakables.match(c))
                    isEndOfSentence = bool(dividersSentence.match(c))
                    isEndOfWord = bool(dividersWord.match(c))

                    # Add letter to subtitles
                    markdownWord += c

                    # Add letter to spoken word
                    if isSpeakable:
                        textWord += c

                    # Add words to sentence and animations
                    if isEndOfWord or isEndOfSentence or isLast:
                        # Add to text-to-speech sentence
                        if len(textWord):
                            textWord_processed = self.lip_sync_pre_process_text(textWord)
                            if len(textWord_processed):
                                ttsSentence.append({"mark": markId, "word": textWord_processed})

                        # Push subtitles to animation queue
                        if len(markdownWord):
                            lipsyncAnim.append(
                                {
                                    "mark": markId,
                                    "template": {"name": "subtitles"},
                                    "ts": [0],
                                    "vs": {"subtitles": markdownWord},
                                }
                            )
                            markdownWord = ""

                        # Push visemes to animation queue
                        if len(textWord):
                            v = self.lip_sync_words_to_visemes(textWord_processed)
                            if v and v.get("visemes") and len(v["visemes"]):
                                d = v["times"][-1] + v["durations"][-1]
                                for j in range(len(v["visemes"])):
                                    viseme = v["visemes"][j]
                                    t1 = (v["times"][j] - 0.6) / d
                                    t2 = (v["times"][j] + 0.5) / d
                                    t3 = (v["times"][j] + v["durations"][j] + 0.5) / d
                                    visemeData = {
                                        "mark": markId,
                                        "template": {"name": "viseme"},
                                        "ts": [t1, t2, t3],
                                        "vs": {"viseme_" + viseme: [None, 0.9 if viseme in ("PP", "FF") else 0.6, 0]},
                                    }
                                    lipsyncAnim.append(visemeData)
                            textWord = ""
                            markId += 1

                    # Process sentences
                    if isEndOfSentence or isLast:
                        # Send sentence to Text-to-speech queue
                        if len(ttsSentence) or (isLast and len(lipsyncAnim)):
                            o = {"anim": lipsyncAnim}
                            if len(ttsSentence):
                                o["text"] = ttsSentence
                            await self.output_queue.put(o)

                            # Reset sentence and animation sequence
                            ttsSentence = []
                            textWord = ""
                            markId = 0
                            lipsyncAnim = []

                        await self.output_queue.put({"break": 100})

                await self.output_queue.put({"break": 100})
        except Exception as e:
            logger.error(f"Error in lip_sync: {e}")
            raise asyncio.CancelledError

    def run(self, input_queue: TextStream) -> TextStream:
        self.input_queue = input_queue
        self._task = asyncio.create_task(self.lip_sync())
        return self.output_queue

