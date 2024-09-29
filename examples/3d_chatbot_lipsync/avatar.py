import json

import outspeed as sp
import outspeed.plugins.contrib.talkinghead as talkinghead


def check_and_load(x):
    print(f"Input: {x}")
    print(f"Input type: {type(x)}")
    res = json.loads(x)
    print(f"Parsed JSON: {res}")
    text = res.get("text")
    print(f"Extracted text: {text}")
    print(f"Text type: {type(text)}")
    return text


@sp.App()
class Chatbot:
    """
    A bot that uses WebSocket to interact with clients, processing audio and text data.

    Methods:
        setup(): Prepares any necessary configurations.
        run(ws: WebSocket): Handles the WebSocket connection and processes data.
        teardown(): Cleans up any resources or configurations.
    """

    async def setup(self):
        self.deepgram_node = sp.DeepgramSTT(sample_rate=48000)
        self.llm_node = sp.GroqLLM(
            system_prompt="You are a language tutor who teaches English.\
            You will always reply with a JSON object.\
            Each message has a text and facialExpression property.\
            The text property is a response to the user (no emoji).\
            The different facial expressions are: smile, sad, angry, and default.",
            temperature=0.9,
            response_format={"type": "json_object"},
            stream=False,
        )
        self.text_to_viseme_node = talkinghead.TextToViseme()
        self.viseme_to_audio_node = talkinghead.VisemeToAudio(api_key="AIzaSyDKnQEdeUyNeFMQciUkHDuX4AbeplQyuWg")


    @sp.websocket()
    async def run(self, audio_input_stream: sp.AudioStream, message_stream: sp.TextStream):
        deepgram_stream = self.deepgram_node.run(audio_input_stream)
        combined_text_stream = sp.merge([deepgram_stream, message_stream])

        llm_token_stream, _ = self.llm_node.run(combined_text_stream)
        json_text_stream = sp.map(llm_token_stream, check_and_load)

        lip_syncd_stream = self.text_to_viseme_node.run(json_text_stream)
        viseme_to_audio_stream = self.viseme_to_audio_node.run(lip_syncd_stream)

        return viseme_to_audio_stream

    async def teardown(self):
        pass


if __name__ == "__main__":
    v = Chatbot()
    v.start()
