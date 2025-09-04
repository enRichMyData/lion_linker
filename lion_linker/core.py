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
        
        # Initialize Hugging Face components if needed
        if self.model_api_provider == "huggingface":
            self._init_huggingface()

    def _init_huggingface(self):
        """Initialize Hugging Face tokenizer and model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            
            # Add padding token if it doesn't exist
            if self.hf_tokenizer.pad_token is None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
                
        except ImportError:
            raise ImportError(
                "Hugging Face transformers and torch are required for huggingface provider. "
                "Install with: pip install transformers torch"
            )

    def chat(self, message, stream=True):
        if self.model_api_provider == "ollama":
            return self._chat_ollama(message, stream)
        elif self.model_api_provider == "openai":
            self.model_api_key
            return self._chat_openai(message, stream)
        elif self.model_api_provider == "groq":
            return self._chat_groq(message, stream)
        elif self.model_api_provider == "huggingface":
            return self._chat_huggingface(message)
        else:
            raise ValueError(f"Unsupported API provider: {self.model_api_provider}")

    def _chat_ollama(self, message, stream):
        print(f"Using Ollama model: {self.model_name}")
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

    def _chat_huggingface(self, message):
        """Chat using Hugging Face transformers."""
        import torch
        
        # Format the message as a conversation
        formatted_message = f"<|user|>\n{message}\n<|assistant|>\n"
        
        # Tokenize the input
        inputs = self.hf_tokenizer(
            formatted_message,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.hf_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.hf_tokenizer.eos_token_id,
                eos_token_id=self.hf_tokenizer.eos_token_id
            )
        
        # Decode the response
        response = self.hf_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
