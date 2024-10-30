import asyncio
import json
import logging
import os

import requests

import outspeed as sp


class CustomLLM:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.output_queue = sp.TextStream()
        self.chat_history = []

        if self.system_prompt:
            self.chat_history.append({"role": "system", "content": self.system_prompt})

    def run(self, input_queue: sp.TextStream) -> sp.TextStream:
        self.input_queue = input_queue
        asyncio.create_task(self.process_input())
        return self.output_queue

    async def process_input(self):
        try:
            while True:
                input_text = await self.input_queue.get()

                if isinstance(input_text, sp.SessionData):
                    self.output_queue.put(input_text)
                    continue

                self.chat_history.append({"role": "user", "content": input_text})

                print(os.getenv("OPENAI_API_KEY"))
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"messages": self.chat_history, "model": "gpt-4o-mini"},
                    headers={
                        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                        "Content-Type": "application/json",
                    },
                )

                print(response.json())
                self.chat_history.append(
                    {"role": "assistant", "content": response.json()["choices"][0]["message"]["content"]}
                )

                await self.output_queue.put(self.chat_history[-1]["content"])

        except asyncio.CancelledError:
            return
        except Exception as e:
            logging.error(e)
            raise asyncio.CancelledError


@sp.App()
class VoiceBot:
    """
    This class handles the setup, running, and teardown of various AI services
    used to process audio input, generate responses, and convert text to speech.
    """

    async def setup(self) -> None:
        """
        This method is called when the app starts. It should be used to set up services, load models, and perform any necessary initialization.
        """
        self.deepgram_node = sp.DeepgramSTT()
        self.llm_node = CustomLLM(
            system_prompt="You are a helpful assistant. Keep your answers very short. No special characters in responses.",
        )
        self.token_aggregator_node = sp.TokenAggregator()
        self.tts_node = sp.CartesiaTTS(
            voice_id="95856005-0332-41b0-935f-352e296aa0df",
        )
        self.vad_node = sp.SileroVAD(min_volume=0)

    @sp.streaming_endpoint()
    async def run(self, audio_input_queue: sp.AudioStream, text_input_queue: sp.TextStream):
        """
        It sets up and runs the various AI services in a pipeline to process audio input and generate audio output.
        """
        deepgram_stream: sp.TextStream = self.deepgram_node.run(audio_input_queue)

        vad_stream: sp.VADStream = self.vad_node.run(audio_input_queue.clone())

        text_input_queue = sp.map(text_input_queue, lambda x: json.loads(x).get("content"))

        llm_input_queue: sp.TextStream = sp.merge(
            [deepgram_stream, text_input_queue],
        )

        llm_token_stream: sp.TextStream
        llm_token_stream = self.llm_node.run(llm_input_queue)

        token_aggregator_stream: sp.TextStream = self.token_aggregator_node.run(llm_token_stream)
        tts_stream: sp.AudioStream = self.tts_node.run(token_aggregator_stream)

        # self.llm_node.set_interrupt_stream(vad_stream)
        self.token_aggregator_node.set_interrupt_stream(vad_stream.clone())
        self.tts_node.set_interrupt_stream(vad_stream.clone())

        return tts_stream

    async def teardown(self) -> None:
        """
        Clean up resources when the VoiceBot is shutting down.
        """
        await self.deepgram_node.close()
        await self.llm_node.close()
        await self.token_aggregator_node.close()
        await self.tts_node.close()


if __name__ == "__main__":
    # Start the VoiceBot when the script is run directly
    VoiceBot().start()
