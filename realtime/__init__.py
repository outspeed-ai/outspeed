import sys

if sys.version_info[:2] < (3, 9):
    raise RuntimeError("This version of Realtime requires at least Python 3.9")
if sys.version_info[:2] >= (3, 13):
    raise RuntimeError("This version of Realtime does not support Python 3.13+")

try:
    from .app import App  # noqa: F401
    from .function import function  # noqa: F401
    from .streaming_endpoint import streaming_endpoint  # noqa: F401
    from .web_endpoint import web_endpoint  # noqa: F401
except Exception:
    print()
    print("#" * 50)
    print("#" + "Something with the Realtime installation seems broken.".center(48) + "#")
    print("#" * 50)
    print()
    raise

__all__ = ["function", "streaming_endpoint", "App", "web_endpoint"]
