import sys

if sys.version_info[:2] < (3, 9):
    raise RuntimeError("This version of Outspeed requires at least Python 3.9")
if sys.version_info[:2] >= (3, 13):
    raise RuntimeError("This version of Outspeed does not support Python 3.13+")

import sentry_sdk

sentry_sdk.init(
    dsn="https://8842715aaa1b7fd845f8a55eea150394@o4506805333983232.ingest.us.sentry.io/4507603326795776",
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100%
    # of sampled transactions.
    # We recommend adjusting this value in production.
    profiles_sample_rate=1.0,
)

import logging


class ColorFormatter(logging.Formatter):
    """
    Colors each log message based on its log level. Uses Select Graphic Rendition escape sequences.
    Wikipedia: https://en.wikipedia.org/wiki/ANSI_escape_code#SGR
    """

    # foreground colors
    grey = "\x1b[90m"
    green = "\x1b[92m"
    yellow = "\x1b[93m"
    red = "\x1b[91m"
    reset = "\x1b[0m"

    # format string
    format = "%(asctime)s | %(levelname)-5.5s | %(message)s"

    # log level to color mapping
    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.format)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def configure_logging(level=logging.INFO):
    """
    Configure the root logger with a ColorFormatter that logs each message to the console with its log level.
    Each log level is colored based on its severity.

    Args:
        level (int): The log level to set for the root logger. Defaults to `logging.INFO`.
    """

    # get the root logger & set the desired level
    logger = logging.getLogger()
    logger.setLevel(level)

    # Create a stream handler
    handler = logging.StreamHandler()
    handler.setLevel(level)  # Set the handler level

    # Set the custom formatter
    handler.setFormatter(ColorFormatter())

    # Add the handler to the root logger
    logger.addHandler(handler)


configure_logging()

try:
    from .app import App  # noqa: F401
    from .data import AudioData, ImageData, TextData, SessionData  # noqa: F401
    from .ops.map import map  # noqa: F401
    from .ops.merge import merge  # noqa: F401
    from .plugins.azure_tts import AzureTTS  # noqa: F401
    from .plugins.cartesia_tts import CartesiaTTS  # noqa: F401
    from .plugins.deepgram_stt import DeepgramSTT  # noqa: F401
    from .plugins.eleven_labs_tts import ElevenLabsTTS  # noqa: F401
    from .plugins.fireworks_llm import FireworksLLM  # noqa: F401
    from .plugins.groq_llm import GroqLLM  # noqa: F401
    from .plugins.token_aggregator import TokenAggregator  # noqa: F401
    from .streaming_endpoint import streaming_endpoint  # noqa: F401
    from .streams import AudioStream, TextStream, VideoStream, VADStream  # noqa: F401
    from .web_endpoint import web_endpoint  # noqa: F401
    from .websocket import websocket  # noqa: F401
    from .utils.clock import Clock  # noqa: F401
    from .plugins.silero_vad import SileroVAD  # noqa: F401
    from .utils.vad import VADState  # noqa: F401
    from .plugins.whisper_stt import WhisperSTT  # noqa: F401
    from .plugins.openai_llm import OpenAILLM  # noqa: F401
    from .plugins.key_frame_detector import KeyFrameDetector  # noqa: F401
    from .plugins.gemini_vision import GeminiVision  # noqa: F401
    from .plugins.openai_vision import OpenAIVision  # noqa: F401
    from .plugins.openai_realtime.openai_realtime import OpenAIRealtime  # noqa: F401
    from .tool import Tool  # noqa: F401
    from .server import RealtimeServer  # noqa: F401
except Exception:
    print()
    print("#" * 50)
    print("#" + "Something with the Outspeed installation seems broken.".center(48) + "#")
    print("#" * 50)
    print()
    raise

__all__ = [
    "configure_logging",
    "streaming_endpoint",
    "App",
    "web_endpoint",
    "websocket",
    "AudioData",
    "ImageData",
    "TextData",
    "CartesiaTTS",
    "DeepgramSTT",
    "GroqLLM",
    "TokenAggregator",
    "map",
    "merge",
    "AzureTTS",
    "ElevenLabsTTS",
    "FireworksLLM",
    "Clock",
    "SileroVAD",
    "VADStream",
    "SessionData",
    "VADState",
    "WhisperSTT",
    "OpenAILLM",
    "KeyFrameDetector",
    "GeminiVision",
    "OpenAIVision",
    "OpenAIRealtime",
    "Tool",
    "RealtimeServer",
]
