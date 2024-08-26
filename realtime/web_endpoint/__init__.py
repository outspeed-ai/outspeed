import functools
import logging

from realtime.server import RealtimeServer

logger = logging.getLogger(__name__)


def web_endpoint(method: str = "POST", path: str = "/"):
    def decorator(func):
        fastapi_app = RealtimeServer().get_app()
        fastapi_app.add_api_route(path, func, methods=[method])

    return decorator
