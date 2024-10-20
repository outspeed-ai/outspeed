import sys

if sys.version_info[:2] < (3, 9):
    raise RuntimeError("This version of Outspeed requires at least Python 3.9")
if sys.version_info[:2] >= (3, 13):
    raise RuntimeError("This version of Outspeed does not support Python 3.13+")


import logging
import platform
import os

import certifi
import coloredlogs

# Fix for "certificate verify failed" error in Python
# Sets SSL cert file to certifi's trusted certificate bundle
# ref: https://stackoverflow.com/a/42334357,
# ref: https://www.happyassassin.net/posts/2015/01/12/a-note-about-ssltls-trusted-certificate-stores-and-platforms/
def cross_platform_where():
    if platform.system() == 'Darwin':  # macOS
        if os.path.exists('/etc/ssl/cert.pem'):
            return '/etc/ssl/cert.pem'
        else:
            return certifi.where()
    elif os.name == 'posix':  # Linux and other POSIX systems
        if os.path.exists('/etc/ssl/certs/ca-certificates.crt'):
            return '/etc/ssl/certs/ca-certificates.crt'
        elif os.path.exists('/etc/ssl/cert.pem'):
            return '/etc/ssl/cert.pem'
        else:
            return certifi.where()
    else:  # Windows and others
        return certifi.where()

os.environ['SSL_CERT_FILE'] = cross_platform_where()

def configure_logging(level=logging.INFO):
    """
    Configure the root logger with a coloredlogs that logs each message to console with a color based on its log level.

    Args:
        level (int): The log level to set for the root logger. Defaults to `logging.INFO`.
    """

    # By default the install() function installs a handler on the root logger,
    # this means that log messages from your code and log messages from the
    # libraries that you use will all show up on the terminal.
    coloredlogs.install(level=level, fmt="%(asctime)s %(levelname)s %(message)s")

    # If you don't want to see log messages from libraries, you can pass a

    # specific logger object to the install() function. In this case only log
    # messages originating from that logger will show up on the terminal.
    # coloredlogs.install(level="DEBUG", logger=logger)


configure_logging()

try:
    from .app import App  # noqa: F401
    from .data import AudioData, ImageData, SessionData, TextData  # noqa: F401
    from .ops.map import map  # noqa: F401
    from .ops.merge import merge  # noqa: F401
    from .plugins.azure_stt import AzureTranscriber  # noqa: F401
    from .plugins.azure_tts import AzureTTS  # noqa: F401
    from .plugins.cartesia_tts import CartesiaTTS  # noqa: F401
    from .plugins.deepgram_stt import DeepgramSTT  # noqa: F401
    from .plugins.eleven_labs_tts import ElevenLabsTTS  # noqa: F401
    from .plugins.fireworks_llm import FireworksLLM  # noqa: F401
    from .plugins.gemini_vision import GeminiVision  # noqa: F401
    from .plugins.groq_llm import GroqLLM  # noqa: F401
    from .plugins.key_frame_detector import KeyFrameDetector  # noqa: F401
    from .plugins.openai_llm import OpenAILLM  # noqa: F401
    from .plugins.openai_realtime.openai_realtime import OpenAIRealtime  # noqa: F401
    from .plugins.openai_vision import OpenAIVision  # noqa: F401
    from .plugins.silero_vad import SileroVAD  # noqa: F401
    from .plugins.token_aggregator import TokenAggregator  # noqa: F401
    from .plugins.whisper_stt import WhisperSTT  # noqa: F401
    from .server import RealtimeServer  # noqa: F401
    from .streaming_endpoint import streaming_endpoint  # noqa: F401
    from .streams import AudioStream, TextStream, VADStream, VideoStream  # noqa: F401
    from .tool import Tool  # noqa: F401
    from .utils.clock import Clock  # noqa: F401
    from .utils.vad import VADState  # noqa: F401
    from .web_endpoint import web_endpoint  # noqa: F401
    from .websocket import websocket  # noqa: F401
except Exception:
    print()
    print("#" * 50)
    print("#" + "Something with the Outspeed installation seems broken.".center(48) + "#")
    print("#" * 50)
    print()
    raise

import av

av.logging.set_level(av.logging.PANIC)


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
    "AzureTranscriber",
]
