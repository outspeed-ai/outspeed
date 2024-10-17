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
        sample_rate: int = 16000,
        languages: List[str] = ["en-US"],
        num_channels: int = 1,
        sample_width: int = 2,
        min_silence_duration: int = 100,
        confidence_threshold: float = 0.8,
        max_silence_duration: int = 2,
    ):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.sample_width = sample_width
        self.min_silence_duration = min_silence_duration
        self.confidence_threshold = confidence_threshold
        self.max_silence_duration = max_silence_duration
        self.output_queue = TextStream()

        format = AudioStreamFormat(
            samples_per_second=sample_rate,
            wave_stream_format=AudioStreamWaveFormat.PCM,
            bits_per_sample=sample_width * 8,
            channels=num_channels,
        )

        self.push_stream = PushAudioInputStream(format)

        self.audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)

        self.api_key = api_key or os.getenv("AZURE_SPEECH_KEY")
        if not self.api_key:
            raise ValueError("Azure Speech API key is required")

        self.region = region or os.getenv("AZURE_SPEECH_REGION")
        if not self.region:
            raise ValueError("Azure Speech API region is required")

        speech_config = speechsdk.SpeechConfig(
            subscription=self.api_key,
            region=self.region,
        )

        speech_params = {
            "speech_config": speech_config,
            "audio_config": self.audio_config,
        }

        self.languages = languages

        if len(self.languages) > 1:
            speech_config.set_property(
                property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
                value="Continuous",
            )
            auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                languages=self.languages
            )

            speech_params["auto_detect_source_language_config"] = auto_detect_source_language_config
        else:
            speech_params["language"] = self.languages[0]

        self.speech = speechsdk.SpeechRecognizer(**speech_params)
        self._initialized_azure_connection = False
        self._audio_duration_received = 0

    def recognized_sentence_final(self, evt):
        logging.info(f"Azure STT: {evt.result.text}")
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

            def stop_cb(evt):
                logging.debug("CLOSING on {}".format(evt))
                self.speech.stop_continuous_recognition()
                self._ended = True

            self.speech.recognizing.connect(lambda x: self.recognized_sentence_stream(x))
            self.speech.recognized.connect(lambda x: self.recognized_sentence_final(x))
            self.speech.session_started.connect(lambda evt: logging.debug("SESSION STARTED: {}".format(evt)))

            self.speech.session_stopped.connect(stop_cb)
            self.speech.canceled.connect(stop_cb)
            self.speech.start_continuous_recognition_async()
            self._initialized_azure_connection = True
        except Exception:
            logging.error("Azure connection failed", exc_info=True)
            raise asyncio.CancelledError()

    async def _run_loop(self):
        try:
            while True:
                data: Union[AudioData, SessionData] = await self.input_queue.get()
                if not self._initialized_azure_connection:
                    await self._connect_ws()

                if isinstance(data, SessionData):
                    await self.output_queue.put(data)
                    continue

                bytes_data = data.get_bytes()
                self._audio_duration_received += len(bytes_data) / (
                    self.sample_rate * self.num_channels * self.sample_width
                )
                self.push_stream.write(bytes_data)
        except Exception:
            logging.error("Azure send task failed", exc_info=True)
            raise asyncio.CancelledError()

    async def close(self):
        self._ended = True
        self.speech.stop_continuous_recognition_async()
