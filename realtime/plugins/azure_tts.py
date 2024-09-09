from enum import Enum
import faulthandler
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

faulthandler.enable()

import asyncio
import logging
import os

import azure.cognitiveservices.speech as speechsdk

from realtime.data import AudioData
from realtime.plugins.base_plugin import Plugin
from realtime.streams import ByteStream, TextStream
from realtime.utils import tracing

logger = logging.getLogger(__name__)

# Constant for converting Azure's audio offset ticks to seconds
AUDIO_OFFSET_TICKS_PER_SECOND = 1e7

# Mapping of Azure viseme IDs to mouth shapes
# TODO: deprecate dictionary. keeping it for now for backwards compatibility
viseme_id_to_mouth_shapes: Dict[int, str] = {
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


class AzureOutputFormat(Enum):
    pcm_16000 = speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
    mp3_16000 = speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3


class AzureTTS(Plugin):
    """
    A plugin for Azure Text-to-Speech synthesis.

    This class handles the conversion of text to speech using Azure's Cognitive Services,
    including streaming audio output and viseme (mouth shape) data for lip-syncing.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        voice_id: str = "en-US-AvaMultilingualNeural",
        output_format: Union[AzureOutputFormat, str] = "pcm_16000",
        optimize_streaming_latency: int = 4,
        stream: bool = True,
        azure_speech_region: Optional[str] = None,
    ):
        """
        Initialize the AzureTTS plugin.

        Args:
            api_key (Optional[str]): Azure Speech API key. If not provided, it will be read from environment variables.
            voice_id (str): The ID of the voice to use for synthesis.
            output_format (str): The audio output format.
            optimize_streaming_latency (int): Latency optimization level for streaming.
            stream (bool): Whether to stream the audio output or not.
            azure_speech_region (Optional[str]): Azure Speech service region. If not provided, it will be read from environment variables.
        """
        super().__init__()

        # Set up Azure Speech configuration
        self._api_key = api_key or os.getenv("AZURE_SPEECH_KEY")
        self._azure_speech_region = azure_speech_region or os.getenv("AZURE_SPEECH_REGION")
        if not self._api_key:
            raise ValueError("Please set AZURE_SPEECH_KEY environment variable or pass it as a parameter")
        if not self._azure_speech_region:
            raise ValueError("Please set AZURE_SPEECH_REGION environment variable or pass it as a parameter")

        self._voice_id = voice_id
        if isinstance(output_format, str):
            output_format = AzureOutputFormat[output_format]
        self._output_format = output_format.value
        self._optimize_streaming_latency = optimize_streaming_latency
        self._stream = stream

        # TODO: Remove hardcoded sample rate, channels, and sample width
        if self._output_format == speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm:
            self.sample_rate = 16000
            self.channels = 1
            self.sample_width = 2
        elif self._output_format == speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3:
            self.sample_rate = 16000
            self.channels = 1
            self.sample_width = 2
        else:
            raise ValueError(f"Unsupported Azure TTS output format: {self._output_format}")

        # Initialize streams and data structures
        self.output_queue = ByteStream()
        self._text_queue = TextStream()
        self.viseme_stream = TextStream()
        self._viseme_data: Dict[str, List[Dict[str, Union[str, int, float]]]] = {"mouthCues": []}
        self._generating = False
        self._task: Optional[asyncio.Task] = None
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)

        # Set up Azure Speech configuration
        self._speech_config = speechsdk.SpeechConfig(subscription=self._api_key, region=self._azure_speech_region)
        self._speech_config.set_speech_synthesis_output_format(self._output_format)
        self._speech_config.speech_synthesis_voice_name = voice_id

        # Initialize speech synthesizer
        try:
            self._speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self._speech_config, audio_config=None)
            self._speech_synthesizer.viseme_received.connect(self.viseme_received_cb)
        except Exception as e:
            logger.debug(f"Error while connecting to Azure synthesizer: {e}")

    def viseme_received_cb(self, evt: speechsdk.SpeechSynthesisVisemeEventArgs) -> None:
        """
        Callback function for handling viseme events.

        This function is called when a viseme event is received from the Azure Speech SDK.
        It updates the viseme data and sends it through the viseme stream.

        Args:
            evt (speechsdk.SpeechSynthesisVisemeEventArgs): The viseme event data.
        """
        current_time = evt.audio_offset / AUDIO_OFFSET_TICKS_PER_SECOND

        if self._viseme_data["mouthCues"]:
            # Update the end time of the previous viseme
            self._viseme_data["mouthCues"][-1]["end"] = current_time

        # Add the new viseme data
        new_viseme = {
            "value": viseme_id_to_mouth_shapes[evt.viseme_id],  # TODO: deprecate 'value' key
            "azure_viseme_id": evt.viseme_id,
            "start": current_time if self._viseme_data["mouthCues"] else 0.0,
            "end": (evt.audio_offset + AUDIO_OFFSET_TICKS_PER_SECOND) / AUDIO_OFFSET_TICKS_PER_SECOND,
        }
        self._viseme_data["mouthCues"].append(new_viseme)

        # Send updated viseme data through the stream
        self.viseme_stream.put_nowait(json.dumps(self._viseme_data))

    def run(self, input_queue: TextStream) -> Tuple[ByteStream, TextStream]:
        """
        Start the TTS synthesis process.

        Args:
            input_queue (TextStream): The input stream of text to synthesize.

        Returns:
            Tuple[ByteStream, TextStream]: A tuple containing the audio output stream and the viseme stream.
        """
        self.input_queue = input_queue
        self._task = asyncio.gather(self.synthesize_speech(), self._process_text())
        return self.output_queue, self.viseme_stream

    async def _process_text(self):
        while True:
            text_chunk = await self.input_queue.get()
            if self._generating:
                continue

            await self._text_queue.put(text_chunk)

    async def synthesize_speech(self) -> None:
        """
        Main loop for speech synthesis.

        This method continuously reads from the input queue, synthesizes speech,
        and sends the audio data to the output queue.
        """
        while True:
            text_chunk = await self._text_queue.get()
            if not text_chunk:
                continue

            self._generating = True
            tracing.register_event(tracing.Event.TTS_START)
            logger.info("Generating TTS %s", text_chunk)

            if self._stream:
                await self._stream_synthesis(text_chunk)
            else:
                await self._batch_synthesis(text_chunk)

            tracing.register_event(tracing.Event.TTS_END)
            tracing.log_timeline()
            self._viseme_data = {"mouthCues": []}
            self._generating = False
            await self.output_queue.put(None)

    async def _stream_synthesis(self, text_chunk: str) -> None:
        """
        Perform streaming speech synthesis.

        Args:
            text_chunk (str): The text to synthesize.
        """
        total_audio_bytes = 0
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
        tracing.register_event(tracing.Event.TTS_TTFB)
        while filled_size > 0:
            total_audio_bytes += filled_size
            audio_data = AudioData(
                audio_buffer,
                sample_rate=self.sample_rate,
                channels=self.channels,
                sample_width=self.sample_width,
            )
            await self.output_queue.put(audio_data)
            audio_buffer = bytes(4000)
            filled_size = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool_executor,
                lambda: audio_data_stream.read_data(audio_buffer),
            )
        tracing.register_metric(tracing.Metric.TTS_TOTAL_BYTES, total_audio_bytes)

    async def _batch_synthesis(self, text_chunk: str) -> None:
        """
        Perform batch speech synthesis.

        Args:
            text_chunk (str): The text to synthesize.
        """
        result = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool_executor,
            lambda: self._speech_synthesizer.speak_text_async(text_chunk).get(),
        )
        audio_data = result.audio_data
        tracing.register_metric(tracing.Metric.TTS_TOTAL_BYTES, len(audio_data))
        tracing.register_event(tracing.Event.TTS_TTFB)
        audio_data = AudioData(
            audio_data,
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_width=self.sample_width,
        )
        await self.output_queue.put(audio_data)

    async def close(self) -> None:
        """
        Close the plugin and cancel any ongoing tasks.
        """
        if self._task:
            self._task.cancel()

    async def _interrupt(self) -> None:
        """
        Handle interruptions in speech synthesis.

        This method listens for interrupt signals and cancels ongoing synthesis if necessary.
        """
        while True:
            user_speaking = await self.interrupt_queue.get()
            if self._generating and user_speaking:
                self._task.cancel()
                while not self.output_queue.empty():
                    self.output_queue.get_nowait()
                logger.info("Done cancelling TTS")
                self._generating = False
                self._task = asyncio.create_task(self.synthesize_speech())

    async def set_interrupt(self, interrupt_queue: asyncio.Queue) -> None:
        """
        Set up the interrupt mechanism.

        Args:
            interrupt_queue (asyncio.Queue): The queue to receive interrupt signals.
        """
        self.interrupt_queue = interrupt_queue
        self._interrupt_task = asyncio.create_task(self._interrupt())
