import datetime
from enum import Enum
import os
import logging
from typing import Any, Optional

import requests


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


def send_metric(metric: Metric, value: Optional[Any] = None):
    """
    Sends a metric to the backend.

    Args:
        metric (Metric): The metric to send.
        value (Optional[Any]): The value to send. If not provided, the current timestamp will be used.
    """

    if not BACKEND_URL:
        logging.info("Skipped metric push...")
        return

    job_id = os.getenv("JOB_ID")
    if not job_id:
        logging.info("Skipped metric push...")
        return

    if not value and type(value) is not int and type(value) is not float:
        value = datetime.datetime.now().timestamp()

    res = requests.post(
        BACKEND_URL,
        json={
            "job_id": job_id,
            metric.value: value,
        },
    )

    if res.status_code != 200:
        logging.error(f"{metric.value} push failed: status {res.status_code}, reason: {res.text}")
