from __future__ import annotations

import asyncio
import json
import logging
import os
import websockets
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

from realtime.data import AudioData, SessionData
from realtime.plugins.base_plugin import Plugin
from realtime.streams import TextStream
from realtime.utils import tracing
from pydub import AudioSegment


logger = logging.getLogger(__name__)
SENTENCE_TERMINATORS = [".", "!", "?", "\n", "\r"]
WHISPER_SAMPLE_RATE = 16000


class WhisperSTT(Plugin):
    """
    A plugin for real-time speech-to-text using Deepgram's API.

    This class handles the connection to Deepgram's WebSocket API, sends audio data,
    and processes the returned transcriptions.
    """

    def __init__(
        self,
        *,
        language: str = None,
        detect_language: bool = False,
        interim_results: bool = True,
        punctuate: bool = True,
        smart_format: bool = True,
        model: str = "distil-whisper/distil-small.en",
        api_key: Optional[str] = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        sample_width: int = 2,
        min_silence_duration: int = 100,
        confidence_threshold: float = 0.8,
        base_url: Optional[str] = None,
        vad_threshold: float = 0.5,
    ) -> None:
        """
        Initialize the DeepgramSTT plugin.

        :param language: The language code for transcription.
        :param detect_language: Whether to enable language detection.
        :param interim_results: Whether to return interim results.
        :param punctuate: Whether to add punctuation to the transcript.
        :param smart_format: Whether to apply smart formatting to the transcript.
        :param model: The Deepgram model to use for transcription.
        :param api_key: The Deepgram API key. If None, it will be read from the DEEPGRAM_API_KEY environment variable.
        :param sample_rate: The sample rate of the audio in Hz.
        :param num_channels: The number of audio channels.
        :param sample_width: The width of each audio sample in bytes.
        :param min_silence_duration: The minimum duration of silence to trigger end of speech, in milliseconds.
        :param confidence_threshold: The minimum confidence score to accept a transcription.
        """
        api_key = api_key or os.environ.get("WHISPER_API_KEY")
        if api_key is None:
            raise ValueError("Whisper API key is required")

        if base_url is None:
            raise ValueError("Whisper base URL is required")

        if sample_rate != 16000 and sample_rate != 8000:
            logging.error(f"Invalid sample rate: {sample_rate}. Whisper sample rate needs to be 16000 or 8000")
            raise ValueError("Whisper sample rate needs to be 16000 or 8000")

        self._api_key: str = api_key
        self.base_url: str = base_url
        self.language: str = language
        self.detect_language: bool = detect_language
        self.interim_results: bool = interim_results
        self.punctuate: bool = punctuate
        self.vad_threshold: float = vad_threshold
        self.smart_format: bool = smart_format
        self.model: str = model
        self.min_silence_duration: int = min_silence_duration
        self.endpointing: int = min_silence_duration

        self._sample_rate: int = sample_rate
        self._num_channels: int = num_channels
        self._sample_width: int = sample_width
        self._speaking: bool = False
        self.confidence_threshold: float = confidence_threshold

        self._closed: bool = False
        self.output_queue: TextStream = TextStream()
        self._audio_duration_received: float = 0.0
        self.input_queue: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._whisper_chunk_size = 512 if self._sample_rate == 16000 else 256
        self._whisper_chunk_bytes = self._whisper_chunk_size * self._num_channels * 2

    def resample_audio(self, audio_bytes: bytes) -> bytes:
        if self._sample_rate == 16000:
            return audio_bytes

        audio_segment = AudioSegment(
            data=bytes(audio_bytes),
            sample_width=2,  # Assuming 16-bit audio
            frame_rate=self._sample_rate,  # Whisper sample rate
            channels=1,
        )

        # Resample the audio to 16000 Hz
        resampled_audio = audio_segment.set_frame_rate(WHISPER_SAMPLE_RATE)

        # Convert the resampled audio back to bytes
        data = resampled_audio.raw_data
        if len(data) < 1024:
            data += b"\0" * (1024 - len(data))

        return data

    def run(self, input_queue: asyncio.Queue) -> TextStream:
        """
        Start the Deepgram STT process.

        :param input_queue: The queue to receive audio data from.
        :return: The output queue for transcribed text.
        """
        self.input_queue = input_queue
        self._task = asyncio.create_task(self._run_ws())
        return self.output_queue

    async def _connect_ws(self) -> None:
        """Connect to the Deepgram WebSocket API."""
        live_config: Dict[str, Any] = {
            "model_name": self.model,
            "sample_rate": WHISPER_SAMPLE_RATE,  # Whisper sample rate
            "channels": self._num_channels,
            "language": self.language,
            "min_silence_ms": self.min_silence_duration,
            "thresh": self.vad_threshold,
        }

        # headers = {"Authorization": f"Token {self._api_key}"}

        # url = f"{self.base_url}?{urlencode(live_config).lower()}"
        url = self.base_url
        try:
            socket = await websockets.connect(url)
            await socket.send(json.dumps(live_config))
            self._ws = socket
        except Exception:
            logger.error("Whisper connection failed", exc_info=True)
            raise asyncio.CancelledError()

    async def _run_ws(self) -> None:
        """Run the main WebSocket communication loop with Deepgram."""
        try:
            await asyncio.gather(self._send_task(), self._recv_task())
        except Exception:
            logger.error("Whisper task failed", exc_info=True)

    async def _send_task(self) -> None:
        """
        Send audio data to Deepgram through the WebSocket.

        :param ws: The WebSocket connection to Deepgram.
        """
        try:
            buffer = b""
            while True:
                data: AudioData | SessionData = await self.input_queue.get()
                if not self._ws:
                    await self._connect_ws()

                if isinstance(data, SessionData):
                    await self.output_queue.put(data)
                    continue

                bytes_data = data.get_bytes()
                buffer += bytes_data
                self._audio_duration_received += len(bytes_data) / (
                    self._sample_rate * self._num_channels * self._sample_width
                )
                while len(buffer) >= self._whisper_chunk_bytes:
                    resampled_buffer = self.resample_audio(buffer[: self._whisper_chunk_bytes])
                    await self._ws.send(resampled_buffer)
                    buffer = buffer[self._whisper_chunk_bytes :]
        except Exception:
            logger.error("Whisper send task failed", exc_info=True)
            raise asyncio.CancelledError()

    async def _recv_task(self) -> None:
        """
        Receive and process transcription results from Deepgram.

        :param ws: The WebSocket connection to Deepgram.
        """
        try:
            buffer = ""
            while True:
                if not self._ws:
                    await asyncio.sleep(0.2)
                    continue

                try:
                    msg = await asyncio.wait_for(self._ws.recv(), timeout=1)
                except asyncio.TimeoutError:
                    if buffer:
                        await self.output_queue.put(buffer)
                        buffer = ""
                    continue
                logger.info(f"Whisper msg: {msg}")

                if not msg:
                    continue

                if msg[-1] in SENTENCE_TERMINATORS:
                    await self.output_queue.put(buffer + msg)
                    buffer = ""
                else:
                    buffer += msg
        except Exception:
            logger.error("Whisper receive task failed", exc_info=True)
            raise asyncio.CancelledError()
