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
