import asyncio
import datetime
from enum import Enum
import os
import logging
from typing import Any, Optional

import aiohttp


class Metric(Enum):
    SDK_SERVER_STARTED = "sdk_server_started"
    """
    HTTP server started.
    """

    SDK_SERVER_SHUTDOWN = "sdk_server_shutdown"
    """
    HTTP server shutdown.
    """

    SDK_OFFER_CALLED = "sdk_offer_called"
    """
    /offer endpoint called.
    """

    SDK_WEBRTC_PC_CONNECTED = "sdk_webrtc_pc_connected"
    """
    WebRTC PeerConnection state is connected.
    """

    SDK_WEBRTC_PC_CLOSED = "sdk_webrtc_pc_closed"
    """
    WebRTC PeerConnection state is closed.
    """

    SDK_WEBRTC_PC_FAILED = "sdk_webrtc_pc_failed"
    """
    WebRTC PeerConnection state is failed.
    """


BACKEND_URL = os.getenv("BACKEND_URL")
JOB_METRICS_ENDPOINT = os.getenv("JOB_METRICS_ENDPOINT")

backend_metrics_url = None
if BACKEND_URL and JOB_METRICS_ENDPOINT:
    if not JOB_METRICS_ENDPOINT.startswith("/"):
        JOB_METRICS_ENDPOINT = "/" + JOB_METRICS_ENDPOINT

    backend_metrics_url = BACKEND_URL + JOB_METRICS_ENDPOINT

    # delete the environment variables
    del os.environ["BACKEND_URL"]
    del os.environ["JOB_METRICS_ENDPOINT"]


def send_metric(metric: Metric, value: Optional[Any] = None):
    """
    Sends a metric to the backend.

    Args:
        metric (Metric): The metric to send.
        value (Optional[Any]): The value to send. If not provided, the current timestamp will be used.
    """

    try:
        if not backend_metrics_url:
            logging.debug("Skipped metric push...")
            return

        job_id = os.getenv("JOB_ID")
        if not job_id:
            logging.debug("Skipped metric push...")
            return

        # set value to current timestamp if not provided
        if not value and type(value) is not int and type(value) is not float:
            value = datetime.datetime.now().timestamp()

        # send the request asynchronously
        asyncio.create_task(
            _send_to_backend(
                payload={
                    "job_id": job_id,
                    metric.value: value,
                },
                metric=metric,
            )
        )
    except Exception as e:
        logging.error(f"{metric.value} push failed: error: {e}")


async def _send_to_backend(payload: dict, metric: Metric):
    assert backend_metrics_url

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(backend_metrics_url, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logging.error(f"{metric.value} push failed: status {resp.status}, reason: {text}")
    except Exception as e:
        logging.error(f"{metric.value} push failed: error: {e}")
