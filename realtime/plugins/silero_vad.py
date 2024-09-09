import asyncio
import threading
from enum import Enum
import logging

from realtime.data import AudioData
from realtime.plugins.base_plugin import Plugin
from realtime.plugins.VAD.silero_model import SileroVADModel
from realtime.streams import AudioStream, TextStream
from realtime.utils.audio import calculate_audio_volume, exp_smoothing


class VADState(Enum):
    QUIET = 1
    STARTING = 2
    SPEAKING = 3
    STOPPING = 4


class SileroVAD(Plugin):
    def __init__(
        self,
        audio_sample_rate: int = 8000,
        min_speech_duration_seconds: float = 0.2,
        min_silence_duration_seconds: float = 0.25,
        activation_threshold: float = 0.5,
        min_volume: float = 0.6,
        num_channels: int = 1,
    ):
        self.model = SileroVADModel(sample_rate=audio_sample_rate, num_channels=num_channels)

        self._sample_rate = audio_sample_rate
        self._activation_threshold = activation_threshold

        self._loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self.output_queue = TextStream()
        self.user_speaking = False

        self._vad_buffer = b""
        self._num_channels = num_channels

        # Volume exponential smoothing
        self._smoothing_factor = 0.2
        self._prev_volume = 0
        self._min_volume = min_volume

        self._silero_chunk_size = 512 if self._sample_rate == 16000 else 256
        self._silero_chunk_bytes = self._silero_chunk_size * self._num_channels * 2

        self._min_speech_duration_seconds = min_speech_duration_seconds
        self._min_silence_duration_seconds = min_silence_duration_seconds
        self._speech_duration_seconds = 0
        self._silence_duration_seconds = 0
        self._vad_state = VADState.QUIET

        logging.info(
            f"Initialized SileroVAD with sample rate: {audio_sample_rate}, activation threshold: {activation_threshold}, min volume: {min_volume}"
        )

    async def run(self, input_queue: AudioStream) -> TextStream:
        self.input_queue = input_queue
        self._vad_thread = threading.Thread(target=self.execute_vad, daemon=True)
        self._vad_thread.start()
        logging.info("Starting SileroVAD execution")
        return self.output_queue

    def _get_smoothed_volume(self, audio: bytes) -> float:
        volume = calculate_audio_volume(audio, self._sample_rate)
        return exp_smoothing(volume, self._prev_volume, self._smoothing_factor)

    def execute_vad(self):
        try:
            while True:
                future = asyncio.run_coroutine_threadsafe(self.input_queue.get(), self._loop)
                audio_data: AudioData = future.result()
                buffer = audio_data.get_bytes()
                self._vad_buffer += buffer

                if len(self._vad_buffer) < self._silero_chunk_bytes:
                    continue

                audio_frames = self._vad_buffer[: self._silero_chunk_bytes]
                self._vad_buffer = self._vad_buffer[self._silero_chunk_bytes :]

                confidence = self.model.voice_confidence(audio_frames)

                volume = self._get_smoothed_volume(audio_frames)
                self._prev_volume = volume

                speaking = confidence >= self._activation_threshold and volume >= self._min_volume

                logging.debug(f"VAD confidence: {confidence:.2f}, volume: {volume:.2f}, speaking: {speaking}")

                duration_seconds = self._get_speech_duration_seconds(audio_frames)
                if speaking:
                    if self._vad_state == VADState.QUIET:
                        self._vad_state = VADState.STARTING
                        self._speech_duration_seconds = duration_seconds
                    elif self._vad_state == VADState.STARTING:
                        self._speech_duration_seconds += duration_seconds
                    elif self._vad_state == VADState.STOPPING:
                        self._vad_state = VADState.SPEAKING
                        self._vad_stopping_count = 0
                else:
                    if self._vad_state == VADState.STARTING:
                        self._vad_state = VADState.QUIET
                        self._speech_duration_seconds = 0
                    elif self._vad_state == VADState.SPEAKING:
                        self._vad_state = VADState.STOPPING
                        self._silence_duration_seconds = duration_seconds
                    elif self._vad_state == VADState.STOPPING:
                        self._silence_duration_seconds += duration_seconds

                if (
                    self._vad_state == VADState.STARTING
                    and self._speech_duration_seconds >= self._min_speech_duration_seconds
                ):
                    self._vad_state = VADState.SPEAKING
                    self._speech_duration_seconds = 0
                    logging.info("Speech detected, transitioning to SPEAKING state")

                if (
                    self._vad_state == VADState.STOPPING
                    and self._silence_duration_seconds >= self._min_silence_duration_seconds
                ):
                    self._vad_state = VADState.QUIET
                    self._silence_duration_seconds = 0
                    logging.info("Silence detected, transitioning to QUIET state")

                asyncio.run_coroutine_threadsafe(self.output_queue.put(self._vad_state), self._loop)
        except Exception as e:
            logging.error(f"Error in VAD execution: {str(e)}")

    def _get_speech_duration_seconds(self, audio_frames: bytes) -> float:
        return len(audio_frames) / (self._sample_rate * self._num_channels * 2)
