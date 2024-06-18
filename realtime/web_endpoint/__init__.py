import asyncio
import functools
import inspect
import logging
import os
import ssl
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
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(os.environ["SSL_CERT_PATH"], keyfile=os.environ["SSL_KEY_PATH"])
            server = uvicorn.Server(
                config=uvicorn.Config(
                    webrtc_app,
                    host=HOSTNAME,
                    port=PORT,
                    log_level="info",
                    ssl_keyfile=os.environ["SSL_KEY_PATH"],
                    ssl_certfile=os.environ["SSL_CERT_PATH"],
                )
            )
            await server.serve()

        return wrapper

    return decorator
