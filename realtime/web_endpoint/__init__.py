import asyncio
import functools
import inspect
import logging
import os
import time

import uvicorn

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

logger = logging.getLogger(__name__)


def web_endpoint(method: str = "POST", path: str = "/"):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            webrtc_app = FastAPI()
            webrtc_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
            webrtc_app.add_api_route(path, func, methods=[method])

            HOSTNAME = "0.0.0.0"
            PORT = int(os.getenv("HTTP_PORT", 8080))
            server = uvicorn.Server(config=uvicorn.Config(webrtc_app, host=HOSTNAME, port=PORT, log_level="info"))
            await server.serve()

        return wrapper

    return decorator
