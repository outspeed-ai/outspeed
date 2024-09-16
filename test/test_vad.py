import asyncio
import pytest
import os
import logging
from pydub import AudioSegment

from realtime.data import AudioData
from realtime.plugins.silero_vad import SileroVAD, VADState
from realtime.streams import AudioStream, VADStream


logging.basicConfig(level=logging.DEBUG)


@pytest.mark.asyncio
async def test_run():
    """
    Test the run method of SileroVAD without mocking.

    This test checks if:
    1. The run method returns a TextStream.
    2. The output queue receives VADState values when processing real audio.
    """

    silero_vad = SileroVAD(sample_rate=16000)
    # Create an input queue
    input_queue = AudioStream()

    # Create some sample audio data (silent audio)
    sample_rate = 16000
    duration_ms = 500
    num_samples = int(sample_rate * duration_ms / 1000)
    silent_audio = AudioData(b"\x00" * (num_samples * 2), sample_rate=sample_rate, channels=1)

    # Put sample audio data into the input queue
    await input_queue.put(silent_audio)

    # Run the SileroVAD
    output_stream = silero_vad.run(input_queue)

    # Check if the returned stream is a TextStream
    assert isinstance(output_stream, VADStream)

    # Wait for a short time to allow the VAD to process
    await asyncio.sleep(1.0)

    # Check if we received a VADState from the output queue
    vad_state = await asyncio.wait_for(output_stream.get(), timeout=4.0)
    assert isinstance(vad_state, VADState)
    # Since we provided silent audio, we expect the state to be NOT_SPEAKING
    assert vad_state == VADState.QUIET

    # Clean up
    await input_queue.put(None)  # Signal end of stream

    # Wait for the VAD to finish processing
    await asyncio.sleep(0.5)


@pytest.mark.asyncio
async def test_run_with_speech():
    """
    Test the run method of SileroVAD with real speech audio from a file.

    This test:
    1. Loads a speech audio file.
    2. Checks if the VAD correctly detects speech in the input audio.
    3. Verifies that the output queue receives the expected VADState values.

    Requires:
    - speech.mp3 file in the same directory as the test file.
    - pydub library for audio file handling.
    """
    silero_vad = SileroVAD(sample_rate=16000)
    input_queue = AudioStream()

    # Load the speech.mp3 file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    speech_file = os.path.join(script_dir, "data/speech.mp3")
    audio = AudioSegment.from_mp3(speech_file)

    # Convert to the required format (16kHz, mono, 16-bit PCM)
    audio = audio.set_frame_rate(16000).set_channels(1)

    speech_audio_bytes = audio.raw_data
    # Create AudioData object
    speech_audio = AudioData(speech_audio_bytes, sample_rate=audio.frame_rate, channels=audio.channels)

    await input_queue.put(speech_audio)

    output_stream = silero_vad.run(input_queue)

    # Wait for the VAD to process
    await asyncio.sleep(1.5)

    vad_states = []
    while True:
        try:
            vad_state = await asyncio.wait_for(output_stream.get(), timeout=2.0)
            vad_states.append(vad_state)
        except asyncio.TimeoutError:
            break

    seen = []
    unique_vad_states = [x for x in vad_states if x not in seen and not seen.append(x)]
    print(unique_vad_states, seen)
    # Check if we received a SPEAKING state
    assert unique_vad_states == [
        VADState.QUIET,
        VADState.STARTING,
        VADState.SPEAKING,
        VADState.STOPPING,
    ]
    # Clean up
    await input_queue.put(None)
    await asyncio.sleep(0.5)
