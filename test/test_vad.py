import pytest
import asyncio
from realtime.plugins.silero_vad import SileroVAD, VADState
from realtime.streams import AudioStream, TextStream
from realtime.data import AudioData


@pytest.mark.asyncio
async def test_run():
    """
    Test the run method of SileroVAD without mocking.

    This test checks if:
    1. The run method returns a TextStream.
    2. The output queue receives VADState values when processing real audio.
    """

    silero_vad = SileroVAD()
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
    output_stream = await silero_vad.run(input_queue)

    # Check if the returned stream is a TextStream
    assert isinstance(output_stream, TextStream)

    # Wait for a short time to allow the VAD to process
    await asyncio.sleep(1.0)

    # Check if we received a VADState from the output queue
    vad_state = await asyncio.wait_for(output_stream.get(), timeout=2.0)
    assert isinstance(vad_state, VADState)
    # Since we provided silent audio, we expect the state to be NOT_SPEAKING
    assert vad_state == VADState.QUIET

    # Clean up
    await input_queue.put(None)  # Signal end of stream

    # Wait for the VAD to finish processing
    await asyncio.sleep(0.5)
