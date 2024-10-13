import asyncio
import base64
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Tuple

import websockets
from PIL import Image

import outspeed as sp

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
FRAME_RATE = 25


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

        Args:
            audio_input_stream (AudioStream): The input audio stream to process.

        Returns:
            Tuple[VideoStream, AudioStream]: The output video and audio streams.
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
            send_first_interrupt = True
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
                    send_first_interrupt = True

                    continue
                if not self._ws:
                    self.connect()
                if isinstance(audio_data, sp.SessionData):
                    continue
                audio_duration = audio_data.get_duration_seconds()
                audio_bytes = audio_data.resample(16000).get_bytes()
                audio_start = audio_data.get_start_seconds()
                # if send_first_interrupt:
                #     audio_json = {"interrupt": True, "start_frame_idx": 0}
                #     send_first_interrupt = False
                #     print("sending first interrupt")
                #     await self._ws.send(json.dumps(audio_json))
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

    def superimpose_image(
        self, base_image: Image.Image, overlay_image: Image.Image, position: Tuple[int, int]
    ) -> Image.Image:
        """
        Superimpose the overlay_image onto the base_image at the specified position.

        Args:
            base_image (Image.Image): The base PIL Image object.
            overlay_image (Image.Image): The overlay PIL Image object to be superimposed.
            position (Tuple[int, int]): A tuple (x, y) representing the position to place the overlay.

        Returns:
            Image.Image: The resulting PIL Image with the overlay applied.

        Raises:
            ValueError: If the overlay image cannot be superimposed due to size constraints.
        """
        try:
            start = time.time()
            # Ensure both images have an alpha channel
            overlay_image = overlay_image.convert("RGBA")

            # Create a copy of the base image to avoid modifying the original
            combined_image = base_image.copy()
            # Paste the overlay image onto the base image at the given position
            combined_image.paste(overlay_image, position, mask=overlay_image)
            # combined_image.save("combined.png")
            # resized_image = combined_image.resize((1080, 1080), Image.LANCZOS)

            return combined_image
        except Exception as e:
            logging.error(f"Error superimposing images: {e}")
            raise ValueError(f"Failed to superimpose images: {e}")

    def recv_task(self):
        """
        Continuously receive and process video data from the WebSocket server.
        """
        start_time = 0.0
        silence_flag = True
        # david_img = Image.open("David_00000.png").convert("RGBA")
        # david_img = david_img.resize((david_img.size[0] // 2, david_img.size[1] // 2), Image.LANCZOS)
        # print(f"Loaded base image 'David_00000.png' with size {david_img.size}")
        # maya_img = Image.open("maya2.png").convert("RGBA")
        # print(f"Loaded base image 'maya.png' with size {maya_img.size}")
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
                        # diff = max(self.image_output_stream.qsize() - self.audio_output_stream.qsize(), 0)
                        # print(self.image_output_stream.qsize(), self.audio_output_stream.qsize(), start_time)
                        # i = 1
                        # while True:
                        #     if not self.audio_output_stream.get_element_at_index(i):
                        #         break
                        #     if not self.image_output_stream.get_element_at_index(i + diff):
                        #         break
                        #     start_time -= image["duration"]
                        # print(self.image_output_stream.qsize(), self.audio_output_stream.qsize(), start_time)
                    start_time = max(sp.Clock.get_playback_time(), start_time)
                    # print([image[x] for x in image if x != "image" and x != "audio"], time.time())
                    image_bytes = base64.b64decode(image["image"])
                    audio_bytes = base64.b64decode(image["audio"])

                    # try:
                    #     overlay_img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
                    #     # Superimpose the received image onto david_img at (164, 711)
                    #     combined_img = self.superimpose_image(
                    #         maya_img,
                    #         overlay_img,
                    #         (
                    #             maya_img.size[0] // 2 - overlay_img.size[0] // 2,
                    #             maya_img.size[1] - overlay_img.size[1],
                    #         ),
                    #     )
                    # except IOError as io_error:
                    #     logging.error(f"Error decoding received image: {io_error}")
                    #     continue

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
                    # while (
                    #     self._audio_library
                    #     and image["start_seconds"] >= self._audio_library[0][0]
                    # ):
                    #     audio_data: sp.AudioData = self._audio_library.pop(0)[1]
                    #     audio_data.relative_start_time = start_time
                    #     self.audio_output_stream.put_nowait(audio_data)
                    # start_time += image["duration"]
        except Exception as e:
            logging.error(f"Error receiving video data: {e}")
            raise asyncio.CancelledError()

    def set_interrupt_stream(self, interrupt_stream: sp.VADStream):
        if isinstance(interrupt_stream, sp.VADStream):
            self.interrupt_queue = interrupt_stream
        else:
            raise ValueError("Interrupt stream must be a VADStream")
        self._interrupt_task = asyncio.create_task(self._interrupt())

    async def _interrupt(self):
        while True:
            vad_state: sp.VADState = await self.interrupt_queue.get()
            if vad_state == sp.VADState.SPEAKING and (self._speaking or not self.audio_input_stream.empty()):
                logging.info("Interrupting Deepreel")
                # self._stop_event.set()
                # while self._send_thread.is_alive() or self._recv_thread.is_alive():
                #     logging.info("Waiting for threads to stop...")
                #     await asyncio.sleep(0.05)  # Non-blocking wait
                while self.audio_input_stream.qsize() > 0:
                    self.audio_input_stream.get_nowait()
                diff = max(self.image_output_stream.qsize() - self.audio_output_stream.qsize(), 0)
                print(self.image_output_stream.qsize(), self.audio_output_stream.qsize())
                i = FRAME_RATE
                temp = None
                while True:
                    if not self.audio_output_stream.get_element_at_index(i):
                        break
                    temp = self.image_output_stream.get_element_at_index(i + diff)
                await self._ws.send(json.dumps({"interrupt": True, "start_frame_idx": temp.extra_tags["frame_idx"]}))

                logging.info("Done cancelling Deepreel")
                self._stop_event.clear()
                self._send_thread = threading.Thread(target=self.send_task)
                self._send_thread.start()
                self._recv_thread = threading.Thread(target=self.recv_task)
                self._recv_thread.start()


@sp.App()
class DeepReelBot:
    """
    A bot that processes audio input, generates responses, and produces video output.
    """

    async def setup(self):
        """
        Set up the bot. Currently empty, but can be used for initialization if needed.
        """
        self.deepreel_node = DeepreelPlugin(websocket_url="ws://13.89.58.206:8765/ws")
        self.llm_node = sp.OpenAIRealtime(
            initiate_conversation_with_greeting="Hi. How are you?",
            voice_id="shimmer",
            system_prompt="You are a helpful sales agent for Outspeed. No special characters in responses.",
        )

    @sp.streaming_endpoint()
    async def run(
        self, audio_input_stream: sp.AudioStream, message_stream: sp.TextStream, video_input_stream: sp.VideoStream
    ) -> Tuple[sp.VideoStream, sp.AudioStream]:
        """
        The main processing pipeline for the bot.

        Args:
            audio_input_stream (sp.AudioStream): The input audio stream.
            message_stream (sp.TextStream): The input text message stream.

        Returns:
            Tuple[sp.VideoStream, sp.AudioStream]: The output video and audio streams.
        """

        async def use_video_stream(video_input_queue):
            while True:
                frame = await video_input_queue.get()

        asyncio.create_task(use_video_stream(video_input_stream))

        audio_output_stream = self.llm_node.run(message_stream, audio_input_stream)

        video_stream, audio_stream = self.deepreel_node.run(audio_output_stream)

        return video_stream, audio_stream

    async def teardown(self):
        """
        Clean up resources when the bot is shutting down. Currently empty, but can be implemented if needed.
        """
        pass


if __name__ == "__main__":
    try:
        bot = DeepReelBot()
        bot.start()
    except Exception:
        pass
    except KeyboardInterrupt:
        pass
