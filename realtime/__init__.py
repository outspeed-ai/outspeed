import sys

if sys.version_info[:2] < (3, 9):
    raise RuntimeError("This version of Realtime requires at least Python 3.9")
if sys.version_info[:2] >= (3, 13):
    raise RuntimeError("This version of Realtime does not support Python 3.13+")

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
    from .function import function  # noqa: F401
    from .streaming_endpoint import streaming_endpoint  # noqa: F401
    from .web_endpoint import web_endpoint  # noqa: F401
    from .websocket import websocket  # noqa: F401
except Exception:
    print()
    print("#" * 50)
    print("#" + "Something with the Realtime installation seems broken.".center(48) + "#")
    print("#" * 50)
    print()
    raise

__all__ = ["function", "streaming_endpoint", "App", "web_endpoint", "websocket"]
