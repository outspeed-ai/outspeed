import logging
import os
import ssl
import socket
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


class RealtimeServer:
    """
    A singleton class representing a real-time server using FastAPI.
    """

    _instance = None  # Class variable to hold the single instance
    _initialized: bool = False  # Flag to check if __init__ has been called
    _connections: int = 0  # Counter for active connections

    def __new__(cls) -> "RealtimeServer":
        """
        Ensure only one instance of RealtimeServer is created.
        """
        if cls._instance is None:
            cls._instance = super(RealtimeServer, cls).__new__(cls)
            cls._instance.__init__()
            cls._initialized = True
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize the RealtimeServer instance.
        """
        if self._initialized:
            return
        self.app: FastAPI = FastAPI()
        self.app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
        self.HOSTNAME: str = os.getenv("OUTSPEED_HOSTNAME", "localhost")
        self.HOST_IP_ADDRESS: str = os.getenv("OUTSPEED_HOST_IP_ADDRESS", "0.0.0.0")
        self.PORT: int = int(
            os.getenv("OUTSPEED_HTTP_PORT") or os.getenv("HTTP_PORT", 8080)
        )  # Keeping HTTP_PORT for backward compatibility
        self._is_ssl_enabled: bool = False
        self.server: Optional[uvicorn.Server] = None

    async def start(self) -> None:
        """
        Start the server with SSL configuration.
        """
        self.app.add_api_route("/connections", self.get_connections, methods=["GET"])
        if (
            os.getenv("SSL_CERT_PATH")
            and os.getenv("SSL_KEY_PATH")
            and os.path.exists(os.environ["SSL_CERT_PATH"])
            and os.path.exists(os.environ["SSL_KEY_PATH"])
        ):
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(os.environ["SSL_CERT_PATH"], keyfile=os.environ["SSL_KEY_PATH"])
            self.server = uvicorn.Server(
                config=uvicorn.Config(
                    self.app,
                    host=self.HOST_IP_ADDRESS,
                    port=self.PORT,
                    log_level="info",
                    ssl_keyfile=os.environ["SSL_KEY_PATH"],
                    ssl_certfile=os.environ["SSL_CERT_PATH"],
                )
            )
            self._is_ssl_enabled = True
        else:
            # Local server
            self.app.add_api_route("/", self.get_local_offer_url, methods=["GET"])

            while is_port_in_use(self.HOST_IP_ADDRESS, self.PORT):
                logging.info(f"Port {self.PORT} is in use. Trying next port...")
                self.PORT += 1

            logging.info(f"Local server detected. Use http://{self.HOSTNAME}:{self.PORT}/ as Function URL.")

            self.server = uvicorn.Server(
                config=uvicorn.Config(
                    self.app,
                    host=self.HOST_IP_ADDRESS,
                    port=self.PORT,
                    log_level="info",
                )
            )

        await self.server.serve()

    async def get_connections(self) -> Dict[str, List[str]]:
        """
        Get the current connections status.
        Returns a dictionary with a list of active connections (if any).
        """
        return {"connections": ["active_connection"] if self._connections > 0 else []}

    def add_connection(self) -> None:
        """
        Increment the connection counter.
        """
        self._connections += 1

    def remove_connection(self) -> None:
        """
        Decrement the connection counter.
        """
        # TODO: Log when number of connections < 0
        self._connections = max(self._connections - 1, 0)

    def get_app(self) -> FastAPI:
        """
        Get the FastAPI application instance.
        """
        return self.app

    async def get_local_offer_url(self) -> Dict[str, str]:
        """
        Get the local offer URL.
        """
        return {"address": f"{'https' if self._is_ssl_enabled else 'http'}://{self.HOSTNAME}:{self.PORT}"}

    async def shutdown(self):
        """
        Shutdown the server.
        """
        await self.server.shutdown()


def is_port_in_use(host: str, port: str):
    """
    Test if a port is in use. Uses `socket.connect_ex()` to check if it was
    able to connect to the given host and port or not.

    Args:
        host (str): The host to test.
        port (str): The port to test.

    Returns:
        bool: True if the port is in use, False otherwise.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, port)) == 0
