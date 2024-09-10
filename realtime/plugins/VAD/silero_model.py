import logging
import os
import time

import numpy as np
import torch

# Brute force torchaudio to use ffmpeg 7, otherwise it will use ffmpeg 6 which causes issues on mac
os.environ["TORIO_USE_FFMPEG_VERSION"] = "7"
# Although torchaudio is not utilized directly in this context, it is necessary to attempt its import as it is a dependency for Silero.
import torchaudio  # noqa: F401
from silero_vad import load_silero_vad


class SileroVADModel:
    """
    A Voice Activity Detection (VAD) model using Silero VAD.

    This class implements a VAD model that uses the Silero VAD model to detect voice activity
    in audio samples.

    Attributes:
        _sample_rate (int): The sample rate of the audio input.
        _num_channels (int): The number of audio channels.
        _model: The loaded Silero VAD model.
        _last_reset_time (float): The timestamp of the last model state reset.
        _model_reset_states_time (float): The time interval for resetting model states.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        num_channels: int,
        model_reset_states_time: float = 5.0,
    ):
        """
        Initialize the SileroVADModel.

        Args:
            sample_rate (int): The sample rate of the audio input (must be 16000 or 8000).
            num_channels (int): The number of audio channels.
            model_reset_states_time (float): The time interval for resetting model states.

        Raises:
            ValueError: If the sample rate is not 16000 or 8000.
        """
        self._sample_rate = sample_rate

        if sample_rate != 16000 and sample_rate != 8000:
            logging.error(f"Invalid sample rate: {sample_rate}. Silero VAD sample rate needs to be 16000 or 8000")
            raise ValueError("Silero VAD sample rate needs to be 16000 or 8000")

        self._num_channels = num_channels

        logging.info(f"Initializing SileroVADModel with sample rate: {sample_rate}, channels: {num_channels}")
        # self._model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False)
        self._model = load_silero_vad()
        logging.info("Silero VAD model loaded successfully")

        self._last_reset_time = 0
        self._model_reset_states_time = model_reset_states_time

    def voice_confidence(self, buffer: bytes) -> float:
        """
        Calculate the voice confidence for the given audio buffer.

        Args:
            buffer (bytes): The audio buffer to analyze.

        Returns:
            float: The voice confidence score (0.0 to 1.0).
        """
        try:
            audio_int16 = np.frombuffer(buffer, np.int16)
            # Divide by 32768 because we have signed 16-bit data.
            audio_float32 = np.frombuffer(audio_int16, dtype=np.int16).astype(np.float32) / 32768.0
            new_confidence = self._model(torch.from_numpy(audio_float32), self._sample_rate).item()

            # We need to reset the model from time to time because it doesn't
            # really need all the data and memory will keep growing otherwise.
            curr_time = time.time()
            diff_time = curr_time - self._last_reset_time
            if diff_time >= self._model_reset_states_time:
                logging.debug(f"Resetting Silero VAD model states after {diff_time:.2f} seconds")
                self._model.reset_states()
                self._last_reset_time = curr_time

            logging.debug(f"Voice confidence: {new_confidence:.4f}")
            return new_confidence
        except Exception as e:
            # This comes from an empty audio array
            logging.error(f"Error analyzing audio with Silero VAD: {e}")
            return 0
