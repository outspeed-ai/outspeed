from concurrent.futures import ThreadPoolExecutor
import faulthandler
import json

faulthandler.enable()


import asyncio
import logging
import os
import time

import aiohttp

from realtime.plugins.base_plugin import Plugin
from realtime.streams import ByteStream, TextStream
import azure.cognitiveservices.speech as speechsdk

logger = logging.getLogger(__name__)


viseme_id_to_mouth_shapes = {
    0: "X",
    1: "D",
    2: "D",
    3: "F",
    4: "C",
    5: "C",
    6: "B",
    7: "F",
    8: "E",
    9: "E",
    10: "F",
    11: "D",
    12: "C",
    13: "F",
    14: "C",
    15: "B",
    16: "F",
    17: "H",
    18: "G",
    19: "H",
    20: "H",
    21: "A",
}


class AzureTTS(Plugin):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice_id: str = "en-US-AvaMultilingualNeural",
        output_format: str = "pcm_16000",
        optimize_streaming_latency: int = 4,
        stream: bool = True,
        azure_speech_region: str | None = None,
    ):
        super().__init__()

        azure_speech_key = api_key or os.getenv("AZURE_SPEECH_KEY")
        azure_speech_region = azure_speech_region or os.getenv("AZURE_SPEECH_REGION")
        if not azure_speech_key:
            raise ValueError("Please set AZURE_SPEECH_KEY environment variable or pass it as a parameter")
        if not azure_speech_region:
            raise ValueError("Please set AZURE_SPEECH_REGION environment variable or pass it as a parameter")
        self._api_key = azure_speech_key
        self._voice_id = voice_id
        self._output_format = output_format
        self._optimize_streaming_latency = optimize_streaming_latency

        self.output_queue = ByteStream()
        self.viseme_stream = TextStream()
        self._viseme_data = {"mouthCues": []}
        self._generating = False
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)

        self._stream = stream
        self._speech_config = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_speech_region)
        self._speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
        )
        self._speech_config.speech_synthesis_voice_name = voice_id
        try:
            self._speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self._speech_config, audio_config=None)
            self._speech_synthesizer.viseme_received.connect(self.viseme_received_cb)
            # connection = speechsdk.Connection.from_speech_synthesizer(self._speech_synthesizer)
            # connection.open(True)
        except Exception as e:
            logger.debug(f"Error while connecting to Azure synthesizer: {e}")

    def viseme_received_cb(self, evt: speechsdk.SessionEventArgs):
        if len(self._viseme_data["mouthCues"]) > 0:
            self._viseme_data["mouthCues"][-1]["end"] = (evt.audio_offset) / 10000000.0
        self._viseme_data["mouthCues"].append(
            {
                "value": viseme_id_to_mouth_shapes[evt.viseme_id],
                "start": (evt.audio_offset) / 10000000.0 if len(self._viseme_data["mouthCues"]) > 0 else 0.0,
                "end": (evt.audio_offset + 30000.0) / 10000000.0,
            }
        )

    async def run(self, input_queue: TextStream) -> ByteStream:
        self.input_queue = input_queue
        self._task = asyncio.create_task(self.synthesize_speech())
        return self.output_queue, self.viseme_stream

    async def synthesize_speech(self):
        while True:
            text_chunk = await self.input_queue.get()
            if text_chunk is None:
                continue
            self._generating = True
            start_time = time.time()
            logger.info("Generating TTS %s", text_chunk)

            if self._stream:
                result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool_executor,
                    lambda: self._speech_synthesizer.start_speaking_text_async(text_chunk).get(),
                )
                audio_data_stream = speechsdk.AudioDataStream(result)
                audio_buffer = bytes(4000)
                filled_size = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool_executor,
                    lambda: audio_data_stream.read_data(audio_buffer),
                )
                logger.info("Azure TTS TTFB: %s", time.time() - start_time)
                while filled_size > 0:
                    await self.output_queue.put(audio_buffer)
                    filled_size = await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool_executor,
                        lambda: audio_data_stream.read_data(audio_buffer),
                    )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool_executor,
                    lambda: self._speech_synthesizer.speak_text_async(text_chunk).get(),
                )
                audio_data = result.audio_data
                logger.info("Azure TTS TTFB: %s", time.time() - start_time)
                await self.output_queue.put(audio_data)
                await self.viseme_stream.put(json.dumps(self._viseme_data))
                self._viseme_data["mouthCues"] = []

            self._generating = False

    async def close(self):
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
