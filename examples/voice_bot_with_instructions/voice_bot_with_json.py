import outspeed as sp
import json

# So there's 2 types of JSONs:
# 1. Output from our primary LLM (llm_node): This is where most logic
# for the application sits. It takes in user's voice and responds with
# "increment"/"decrement" JSONs or "speak" JSON
# The former is an operation to be executed on the frontend, and the latter
# is the LLM response

# 2. JSONs sent from the frontend. These are of two types:
# "speak_instruction" and "prompt_instruction"
# - "speak_instruction" is simply sent to the backend for the voice model
# to parrot back to the user
# - "prompt_instruction" is a prompt that user sends to the backend LLM
# for information. I have created a new LLM to process this type of message.
# However, you can send this instruction to the primary LLM (llm_node). You'll
# have to change the LLM prompt accordingly so that it accepts that.

@sp.App()
class JsonVoiceBot:
    async def setup(self) -> None:
        self.deepgram_node = sp.DeepgramSTT(min_silence_duration=300)
        self.llm_node = sp.GroqLLM(
            stream=False,
            system_prompt="""You are a helpful assistant. Keep your answers very short.  No special characters in responses.
            User will say whether they want to increment or decrement a timer in minutes. Based on user input, return the following json:
            { type: "increment", quantity: <number>},
            { type: "decrement", quantity: <number>},
            { type: "speak", text: <str>}

            If the user does not ask anything to do with the timer, then simply respond with the json:
            { type: "speak", text: <str>}
            where the text is your normal response
            """,
            response_format={"type": "json_object"}
        )
        self.llm_prompt_node = sp.GroqLLM(
            stream=False,
            system_prompt="""You will receive a user prompt that requires you to respond to what the user asks.
            Respond to the user in plain text. No emojis.
            """
        )

        self.tts_node = sp.CartesiaTTS(
            voice_id="95856005-0332-41b0-935f-352e296aa0df",
            volume=0.7,
        )

        self.vad_node = sp.SileroVAD(
            min_volume=0.6, min_speech_duration_seconds=0.5)

    @sp.streaming_endpoint()
    async def run(self, audio_input_queue: sp.AudioStream, text_input_queue: sp.TextStream) -> sp.AudioStream:
        # Set up the AI service pipeline
        vad_stream: sp.VADStream = self.vad_node.run(audio_input_queue.clone())

        deepgram_stream: sp.TextStream = self.deepgram_node.run(audio_input_queue)

        llm_token_stream: sp.TextStream
        chat_history_stream: sp.TextStream
        # the LLM output is a json with type "speak", "increment", "decrement".
        llm_token_stream, chat_history_stream = self.llm_node.run(deepgram_stream)

        # speak stream for sending to the tts. filter for type "speak"
        llm_speak_json_stream = sp.filter(llm_token_stream.clone(), lambda x: print('llm-speak-filter', x) or json.loads(x).get("type") == "speak")
        llm_speak_stream = sp.map(llm_speak_json_stream, lambda x: lambda x: print('llm-speak-map', x) or json.loads(x).get("text"))

        # increment/decrement operations stream. filter for type "increment", "decrement"
        operation_stream = sp.filter(llm_token_stream, lambda x: self._llm_out_op_predicate(x))

        # process text_input_queue from frontend
        speak_json_stream = sp.filter(text_input_queue.clone(), lambda x: print('speak-filter', x) or json.loads(x).get("type") == "speak_instruction")
        speak_text_stream = sp.map(speak_json_stream, lambda x: print('speak-map', x) or json.loads(x).get("text"))

        prompt_json_stream = sp.filter(text_input_queue, lambda x: json.loads(x).get("type") == "prompt_instruction")
        prompt_text_stream = sp.map(prompt_json_stream, lambda x: json.loads(x).get("text"))
        llm_prompt_resp_stream, _ = self.llm_prompt_node.run(prompt_text_stream)


        tts_input_stream: sp.TextStream = sp.merge(
            [llm_speak_stream, speak_text_stream, llm_prompt_resp_stream]
        )

        tts_stream: sp.AudioStream = self.tts_node.run(tts_input_stream)

        self.llm_node.set_interrupt_stream(vad_stream)
        self.tts_node.set_interrupt_stream(vad_stream.clone())

        return tts_stream, operation_stream

    async def teardown(self) -> None:
        await self.deepgram_node.close()
        await self.llm_node.close()
        await self.token_aggregator_node.close()
        await self.tts_node.close()
        await self.vad_node.close()

    def _llm_out_op_predicate(self, llm_out):
        structured_out = json.loads(llm_out)
        return structured_out["type"] == "increment" or structured_out["type"] == "decrement"

    def _process_text_in(self, text_input):
        structured_out = json.loads(text_input)

        if (structured_out["type"] == "speak_instruction"):
            print(structured_out["text"])
            return structured_out["text"]


        if (structured_out["type"] == "prompt_instruction"):
            return self.llm_prompt_node.run(structured_out["text"])
        else:
            return ""

if __name__ == "__main__":
    # Start the VoiceBot when the script is run directly
    JsonVoiceBot().start()

