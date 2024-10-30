import asyncio
import base64
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Tuple

import websockets

import outspeed as sp

FRAME_RATE = 25

### TODO: change to your websocket url
DEEPREEL_WEBSOCKET_URL = ""


class DeepreelPlugin:
    """
    A plugin for processing audio input and generating video output using a WebSocket connection.
    """

    def __init__(self, websocket_url: str):
        """
        Initialize the DeepreelPlugin.

        Args:
            websocket_url (str): The URL of the WebSocket server to connect to.
        """
        self.websocket_url: str = websocket_url
        self.audio_samples: int = 0
        self.audio_input_stream: sp.AudioStream
        self.image_output_stream: sp.VideoStream
        self.audio_output_stream: sp.AudioStream
        self._ws: websockets.WebSocketClientProtocol
        self._audio_library: List[Tuple[float, sp.AudioData]] = []
        self._ws = None
        self._loop = asyncio.get_running_loop()
        self._stop_event = threading.Event()
        self._speaking = False

    def run(self, audio_input_stream: sp.AudioStream) -> Tuple[sp.VideoStream, sp.AudioStream]:
        """
        Set up the plugin and start processing audio input.
        """
        self.audio_input_stream = audio_input_stream
        self.image_output_stream = sp.VideoStream()
        self.audio_output_stream = sp.AudioStream()
        self._send_thread = threading.Thread(target=self.send_task)
        self._send_thread.start()
        self._recv_thread = threading.Thread(target=self.recv_task)
        self._recv_thread.start()

        return self.image_output_stream, self.audio_output_stream

    async def _connect_ws(self):
        """
        Coroutine to establish a WebSocket connection.
        """
        ws = await websockets.connect(self.websocket_url)
        await ws.send(json.dumps({"metadata": {"sent_frame_buffer": 10}}))
        return ws

    def connect(self):
        """
        Establish a WebSocket connection using the provided URL.
        """
        try:
            # Ensure the event loop is set in the current thread
            self._ws = asyncio.run_coroutine_threadsafe(self._connect_ws(), self._loop).result()
        except Exception as e:
            logging.error(f"Error connecting to websocket: {e}")
            return

    def send_task(self):
        """
        Continuously send audio data to the WebSocket server.
        """
        try:
            while not self._stop_event.is_set():
                try:
                    audio_data: sp.AudioData = asyncio.run_coroutine_threadsafe(
                        asyncio.wait_for(self.audio_input_stream.get(), timeout=0.2), self._loop
                    ).result()
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logging.error(f"Error getting audio data: {e}")
                    break
                if audio_data is None:
                    continue
                if not self._ws:
                    self.connect()
                if isinstance(audio_data, sp.SessionData):
                    continue
                audio_duration = audio_data.get_duration_seconds()
                audio_bytes = audio_data.resample(16000).get_bytes()
                audio_start = audio_data.get_start_seconds()
                audio_json = {
                    "audio_data": base64.b64encode(audio_bytes).decode("utf-8"),
                    "start_seconds": audio_start,
                    "duration": audio_duration,
                    "audio_format": "pcm_16000",
                    "id": str(uuid.uuid4()),
                }
                audio_start += audio_duration
                asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(audio_json)), self._loop)
        except Exception as e:
            logging.error(f"Error sending audio data: {e}")
            raise asyncio.CancelledError()

    def recv_task(self):
        """
        Continuously receive and process video data from the WebSocket server.
        """
        start_time = 0.0
        silence_flag = True
        try:
            while not self._stop_event.is_set():
                if not self._ws:
                    time.sleep(0.1)
                    continue

                try:
                    msg = asyncio.run_coroutine_threadsafe(
                        asyncio.wait_for(self._ws.recv(), timeout=0.2), self._loop
                    ).result()
                except asyncio.TimeoutError:
                    continue

                first_image = self.image_output_stream.get_first_element_without_removing()
                if first_image and first_image.extra_tags["silence_flag"] and self._speaking:
                    self._speaking = False

                response_data: Dict[str, Any] = json.loads(msg)
                images = response_data.get("image_data", [])
                if images and not images[0]["silence_flag"]:
                    print("received frames: ", len(images), time.time())
                for image in images:
                    # Decode the base64 string to bytes
                    if image["silence_flag"] and not silence_flag:
                        print("silence flag to true", time.time())
                        silence_flag = True
                    elif not image["silence_flag"] and silence_flag:
                        print("silence flag to false", time.time())
                        self._speaking = True
                        silence_flag = False
                    start_time = max(sp.Clock.get_playback_time(), start_time)
                    # print([image[x] for x in image if x != "image" and x != "audio"], time.time())
                    image_bytes = base64.b64decode(image["image"])
                    audio_bytes = base64.b64decode(image["audio"])

                    image_data = sp.ImageData(
                        data=image_bytes,
                        format="jpeg",
                        frame_rate=FRAME_RATE,
                        relative_start_time=start_time,
                        extra_tags={"frame_idx": image["frame_idx"], "silence_flag": image["silence_flag"]},
                    )
                    audio_data = sp.AudioData(
                        data=audio_bytes,
                        sample_rate=16000,
                        channels=1,
                        sample_width=2,
                        format="wav",
                        relative_start_time=start_time,
                    )
                    asyncio.run_coroutine_threadsafe(self.image_output_stream.put(image_data), self._loop)
                    asyncio.run_coroutine_threadsafe(self.audio_output_stream.put(audio_data), self._loop)
                    start_time += 1.0 / FRAME_RATE
        except Exception as e:
            logging.error(f"Error receiving video data: {e}")
            raise asyncio.CancelledError()


@sp.App()
class DeepReelBot:
    async def setup(self):
        self.deepreel_node = DeepreelPlugin(websocket_url=DEEPREEL_WEBSOCKET_URL)

        self.llm_node = sp.OpenAIRealtime(
            initiate_conversation_with_greeting="Hi. How are you?",
            voice_id="shimmer",
            system_prompt="You are a helpful sales agent for Outspeed. No special characters in responses.",
        )

    @sp.streaming_endpoint()
    async def run(
        self, audio_input_stream: sp.AudioStream, message_stream: sp.TextStream, video_input_stream: sp.VideoStream
    ):
        """
        The main processing pipeline for the bot.
        """
        audio_output_stream, chat_history_stream = self.llm_node.run(message_stream, audio_input_stream)

        video_stream, audio_stream = self.deepreel_node.run(audio_output_stream)

        return video_stream, audio_stream, chat_history_stream

    async def teardown(self):
        """
        Clean up resources when the bot is shutting down. Currently empty, but can be implemented if needed.
        """
        pass


if __name__ == "__main__":
    bot = DeepReelBot()
    bot.start()
