import asyncio
import logging
import os
from typing import List, Optional, Union

import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech.audio import (
    AudioStreamFormat,
    AudioStreamWaveFormat,
    PushAudioInputStream,
)

from outspeed.data import AudioData, SessionData
from outspeed.plugins.base_plugin import Plugin
from outspeed.streams import AudioStream, TextStream


class AzureTranscriber(Plugin):
    def __init__(
        self,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        languages: List[str] = ["en-US"],
        min_silence_duration: int = 100,
        confidence_threshold: float = 0.8,
        max_silence_duration: int = 2,
    ):
        self._sample_rate: Optional[int] = None
        self._num_channels: Optional[int] = None
        self._sample_width: Optional[int] = None
        self.min_silence_duration = min_silence_duration
        self.confidence_threshold = confidence_threshold
        self.max_silence_duration = max_silence_duration
        self.output_queue = TextStream()

        self.api_key = api_key or os.getenv("AZURE_SPEECH_KEY")
        if not self.api_key:
            raise ValueError("Azure Speech API key is required")

        self.region = region or os.getenv("AZURE_SPEECH_REGION")
        if not self.region:
            raise ValueError("Azure Speech API region is required")

        self._speech_config = speechsdk.SpeechConfig(
            subscription=self.api_key,
            region=self.region,
        )

        self.languages = languages

        self._initialized_azure_connection = False
        self._audio_duration_received = 0

    def recognized_sentence_final(self, evt):
        logging.info(f"Azure STT: {evt.result.text}")
        if evt.result.text:
            self.output_queue.put_nowait(evt.result.text)

    def recognized_sentence_stream(self, evt):
        logging.debug(f"Azure Intermediate STT: {evt.result.text}")

    def run(self, input_queue: AudioStream) -> TextStream:
        """
        Start the Deepgram STT process.

        :param input_queue: The queue to receive audio data from.
        :return: The output queue for transcribed text.
        """
        self.input_queue = input_queue
        self._task = asyncio.create_task(self._run_loop())
        return self.output_queue

    async def _connect_ws(self) -> None:
        try:
            audio_stream_format = AudioStreamFormat(
                samples_per_second=self._sample_rate,
                wave_stream_format=AudioStreamWaveFormat.PCM,
                bits_per_sample=self._sample_width * 8,
                channels=self._num_channels,
            )

            self.push_stream = PushAudioInputStream(audio_stream_format)

            self._audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)

            speech_params = {
                "speech_config": self._speech_config,
                "audio_config": self._audio_config,
            }

            if len(self.languages) > 1:
                self._speech_config.set_property(
                    property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
                    value="Continuous",
                )
                auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                    languages=self.languages
                )

                speech_params["auto_detect_source_language_config"] = auto_detect_source_language_config
            else:
                speech_params["language"] = self.languages[0]

            self._speech = speechsdk.SpeechRecognizer(**speech_params)

            def stop_cb(evt):
                logging.debug("CLOSING on {}".format(evt))
                self._speech.stop_continuous_recognition()
                self._ended = True

            self._speech.recognizing.connect(lambda x: self.recognized_sentence_stream(x))
            self._speech.recognized.connect(lambda x: self.recognized_sentence_final(x))
            self._speech.session_started.connect(lambda evt: logging.debug("SESSION STARTED: {}".format(evt)))

            self._speech.session_stopped.connect(stop_cb)
            self._speech.canceled.connect(stop_cb)
            self._speech.start_continuous_recognition_async()
            self._initialized_azure_connection = True
        except Exception:
            logging.error("Azure connection failed", exc_info=True)
            raise asyncio.CancelledError()

    async def _run_loop(self):
        try:
            while True:
                data: Union[AudioData, SessionData] = await self.input_queue.get()

                if isinstance(data, SessionData):
                    await self.output_queue.put(data)
                    continue

                if not data:
                    continue

                if not self._initialized_azure_connection:
                    self._sample_rate = data.sample_rate
                    self._num_channels = data.channels
                    self._sample_width = data.sample_width
                    await self._connect_ws()

                bytes_data = data.get_bytes()
                self._audio_duration_received += len(bytes_data) / (
                    self._sample_rate * self._num_channels * self._sample_width
                )
                self.push_stream.write(bytes_data)
        except Exception:
            logging.error("Azure send task failed", exc_info=True)
            raise asyncio.CancelledError

    async def close(self):
        self._ended = True
        self._speech.stop_continuous_recognition_async()
