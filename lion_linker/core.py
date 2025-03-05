import ollama
import openai
from groq import Groq


class LLMInteraction:
    def __init__(self, model_name, model_api_provider, ollama_host=None, model_api_key=None):
        self.model_name = model_name
        self.model_api_provider = model_api_provider
        self.model_api_key = model_api_key
        if ollama_host:
            self.ollama_client = ollama.Client(ollama_host)
        else:
            self.ollama_client = ollama.Client()  # Default Ollama host will be used

    def chat(self, message, stream=True):
        if self.model_api_provider == "ollama":
            return self._chat_ollama(message, stream)
        elif self.model_api_provider == "openai":
            self.model_api_key
            return self._chat_openai(message, stream)
        elif self.model_api_provider == "groq":
            return self._chat_groq(message, stream)
        else:
            raise ValueError(f"Unsupported API provider: {self.model_api_provider}")

    def _chat_ollama(self, message, stream):
        response = self.ollama_client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": message}],
        )
        return response["message"]["content"]

    def _chat_openai(self, message, stream):
        # Set the API key directly
        openai.api_key = self.model_api_key

        # Call the OpenAI API
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",  # Role can be "user", "system", or "assistant"
                    "content": message,  # The actual message content
                }
            ],
        )

        # Extract the chatbot's message from the response.
        # Assuming there's at least one response and taking the last one as the chatbot's reply.
        result = response.choices[0].message.content
        return result

    def _chat_groq(self, message, stream):
        client = Groq(api_key=self.model_api_key)

        chat_completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": message,
                }
            ],
        )

        result = chat_completion.choices[0].message.content
        return result
