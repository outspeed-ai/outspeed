import asyncio
import logging
import os
from typing import Optional

import aiohttp

from realtime.data import AudioData
from realtime.plugins.base_plugin import Plugin
from realtime.streams import AudioStream, ByteStream, TextStream
from realtime.utils import tracing

logger = logging.getLogger(__name__)


class ElevenLabsTTS(Plugin):
    """
    A plugin for text-to-speech synthesis using the Eleven Labs API.

    This plugin converts input text to speech audio using Eleven Labs' API,
    supporting both streaming and non-streaming modes.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        voice_id: str = "pNInz6obpgDQGcFmaJgB",
        model: str = "eleven_turbo_v2_5",
        output_format: str = "pcm_16000",
        optimize_streaming_latency: int = 4,
        stream: bool = True,
    ):
        """
        Initialize the ElevenLabsTTS plugin.

        Args:
            api_key (Optional[str]): Eleven Labs API key. If not provided, it will be read from the environment.
            voice_id (str): ID of the voice to use for synthesis.
            model (str): ID of the model to use for synthesis.
            output_format (str): Audio output format ('pcm_16000' or 'pcm_8000').
            optimize_streaming_latency (int): Latency optimization level for streaming.
            stream (bool): Whether to use streaming mode for audio generation.

        Raises:
            ValueError: If the API key is not provided or if an unsupported output format is specified.
        """
        super().__init__()

        # Set up API key
        self._api_key = api_key or os.environ.get("ELEVEN_LABS_API_KEY")
        if self._api_key is None:
            raise ValueError("Eleven Labs API key is required")

        # Store initialization parameters
        self._voice_id = voice_id
        self._model = model
        self._output_format = output_format
        self._optimize_streaming_latency = optimize_streaming_latency
        self._stream = stream

        # Set sample rate based on output format
        if self._output_format == "pcm_16000":
            self.sample_rate = 16000
        elif self._output_format == "pcm_24000":
            self.sample_rate = 24000
        elif self._output_format == "mp3_22050_32":
            self.sample_rate = 22050
        elif self._output_format == "mp3_44100_128":
            self.sample_rate = 44100
        else:
            raise ValueError(f"Unsupported output format: {self._output_format}")

        # Initialize output queue and state variables
        self.output_queue = ByteStream()
        self._generating = False
        self.session: Optional[aiohttp.ClientSession] = None
        self._task: Optional[asyncio.Task] = None
        self.interrupt_queue: Optional[asyncio.Queue] = None
        self._interrupt_task: Optional[asyncio.Task] = None

    def run(self, input_queue: TextStream) -> AudioStream:
        """
        Start the text-to-speech synthesis process.

        Args:
            input_queue (TextStream): Queue containing input text chunks.

        Returns:
            AudioStream: Queue that will contain the generated audio data.
        """
        self.input_queue = input_queue
        self._task = asyncio.create_task(self.synthesize_speech())
        return self.output_queue

    async def synthesize_speech(self):
        """
        Main loop for speech synthesis. Processes input text and generates audio output.
        """
        try:
            async with aiohttp.ClientSession() as self.session:
                while True:
                    # Get the next text chunk from the input queue
                    text_chunk = await self.input_queue.get()
                    if not text_chunk:
                        continue

                    self._generating = True
                    tracing.register_event(tracing.Event.TTS_START)
                    logger.info("Generating TTS %s", text_chunk)

                    # Prepare API request
                    url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}"
                    url += "/stream" if self._stream else ""
                    payload = {"text": text_chunk, "model_id": self._model}
                    querystring = {
                        "output_format": self._output_format,
                        # "optimize_streaming_latency": self._optimize_streaming_latency,
                    }
                    headers = {
                        "xi-api-key": self._api_key,
                        "Content-Type": "application/json",
                    }

                    # Send API request
                    async with self.session.post(url, json=payload, headers=headers, params=querystring) as r:
                        if r.status != 200:
                            logger.error("TTS error %s", await r.text())
                            return

                        # Process the API response
                        first_chunk = True
                        audio_byte_data = b""
                        audio_buffer = b""

                        if self._stream:
                            # Streaming mode: process chunks as they arrive
                            async for chunk in r.content:
                                if chunk:
                                    if first_chunk:
                                        tracing.register_event(tracing.Event.TTS_TTFB)
                                        first_chunk = False
                                    audio_byte_data += chunk
                                    audio_buffer += chunk
                                    if len(audio_buffer) >= 4000:
                                        if len(audio_buffer) % 2 != 0:
                                            self.output_queue.put_nowait(
                                                AudioData(audio_buffer[:-1], sample_rate=self.sample_rate)
                                            )
                                            audio_buffer = audio_buffer[-1:]
                                        else:
                                            self.output_queue.put_nowait(
                                                AudioData(audio_buffer, sample_rate=self.sample_rate)
                                            )
                                            audio_buffer = b""
                            if len(audio_buffer) > 0:
                                self.output_queue.put_nowait(AudioData(audio_buffer, sample_rate=self.sample_rate))
                                audio_buffer = b""
                        else:
                            # Non-streaming mode: process entire response at once
                            audio_byte_data = await r.read()
                            tracing.register_event(tracing.Event.TTS_TTFB)
                            self.output_queue.put_nowait(AudioData(audio_byte_data, sample_rate=self.sample_rate))

                    # Finalize the audio generation
                    tracing.register_event(tracing.Event.TTS_END)
                    tracing.register_metric(tracing.Metric.TTS_TOTAL_BYTES, len(audio_byte_data))
                    tracing.log_timeline()
                    self.output_queue.put_nowait(None)
                    self._generating = False

        except Exception as e:
            logger.error("Error in Eleven Labs TTS: %s", e)
            self._generating = False

    async def close(self):
        """
        Close the plugin, terminating any ongoing processes.
        """
        if self.session:
            await self.session.close()
        if self._task:
            self._task.cancel()

    async def _interrupt(self):
        """
        Handle interruptions to the TTS process, e.g., when the user starts speaking.
        """
        while True:
            user_speaking = await self.interrupt_queue.get()
            if self._generating and user_speaking:
                if self._task:
                    self._task.cancel()
                while not self.output_queue.empty():
                    self.output_queue.get_nowait()
                logger.info("Done cancelling TTS")
                self._generating = False
                self._task = asyncio.create_task(self.synthesize_speech())

    async def set_interrupt(self, interrupt_queue: asyncio.Queue):
        """
        Set up the interrupt mechanism for the TTS process.

        Args:
            interrupt_queue (asyncio.Queue): Queue for receiving interrupt signals.
        """
        self.interrupt_queue = interrupt_queue
        self._interrupt_task = asyncio.create_task(self._interrupt())
