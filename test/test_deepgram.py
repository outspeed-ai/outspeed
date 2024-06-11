import asyncio
import os
import time
import wave

from realtime.plugins.deepgram_stt import DeepgramSTT

TEST_AUDIO_TRANSCRIPT = "the people who are crazy enough to think they can change the world are the ones who do"
TEST_AUDIO_FILEPATH = os.path.join(os.path.dirname(__file__), "change-sophie.wav")


def read_wav_file(filename: str):
    with wave.open(filename, "rb") as file:
        frames = file.getnframes()

        if file.getsampwidth() != 2:
            raise ValueError("Require 16-bit WAV files")

        return file.readframes(frames), file.getframerate()


async def test():
    frame, sample_rate = read_wav_file(TEST_AUDIO_FILEPATH)
    client = DeepgramSTT(api_key=os.environ.get("DEEPGRAM_API_KEY"), sample_rate=sample_rate)
    chunk_size = sample_rate // 100
    input_queue = asyncio.Queue()

    output_queue = await client.arun(input_queue=input_queue)
    for i in range(0, len(frame), chunk_size):
        await asyncio.sleep(0.01)
        data = frame[i : i + chunk_size]
        await input_queue.put(data)
    start_time = time.time()
    print(start_time)
    while True:
        c = await output_queue.get()
        print(f"=== OpenAI LLM TTFB: {time.time() - start_time}")
        if not c:
            break
        print(c, end="")
    await client.aclose()


