import os

import ollama
import openai
import torch
from openai import OpenAI


class LLMInteraction:
    def __init__(self, model_name, model_api_provider, ollama_host=None, model_api_key=None):
        self.model_name = model_name
        self.model_api_provider = model_api_provider
        self.model_api_key = model_api_key

        if self.model_api_provider.lower() not in {"ollama", "openrouter", "huggingface"}:
            raise ValueError(
                "The model api provider must be one of 'ollama', 'openrouter' or 'huggingface'."
                f"Provided: {self.model_api_provider}"
            )
        if self.model_api_provider == "ollama":
            self.ollama_client = ollama.Client(ollama_host) if ollama_host else ollama.Client()
            self.ollama_client.pull(model_name)
        elif self.model_api_provider == "openrouter":
            self.openai_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=model_api_key or os.getenv("OPENAI_API_KEY", None),
            )

        # Initialize Hugging Face components if needed
        if self.model_api_provider == "huggingface":
            self._init_huggingface()

    def _init_huggingface(self):
        """Initialize Hugging Face tokenizer and model."""
        try:
            import math

            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.model_name.startswith("osunlp"):
                pass

                # Set RoPE scaling factor
                context_size = 8192
                config = transformers.AutoConfig.from_pretrained(
                    self.model_name, attn_implementation="eager"
                )

                orig_ctx_len = getattr(config, "max_position_embeddings", None)
                if orig_ctx_len and context_size > orig_ctx_len:
                    scaling_factor = float(math.ceil(context_size / orig_ctx_len))
                    config.rope_scaling = {"type": "linear", "factor": scaling_factor}

                # Load model and tokenizer
                self.hf_model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                self.hf_model.resize_token_embeddings(32001)

                self.hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
                    self.model_name,
                    model_max_length=(
                        context_size if context_size > orig_ctx_len else orig_ctx_len
                    ),
                    padding_side="left",
                    use_fast=False,
                )
            else:
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    trust_remote_code=True,
                )

                # Tokenizer with optimized settings
                self.hf_tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    padding_side="left",
                    use_fast=True,  # Enable rust tokenizer
                    truncation_side="left",
                    trust_remote_code=True,
                )

        except ImportError:
            raise ImportError(
                "Hugging Face transformers and torch are required for huggingface provider. "
                "Install with: pip install transformers torch"
            )

    def chat(self, message, *args, stream=True, **kwargs):
        if self.model_api_provider == "ollama":
            return self._chat_ollama(message, *args, stream=stream, **kwargs)
        elif self.model_api_provider == "openrouter":
            return self._chat_openai(message, *args, stream=stream, **kwargs)
        elif self.model_api_provider == "huggingface":
            return self._chat_huggingface(message, *args, stream=stream, **kwargs)
        else:
            raise ValueError(f"Unsupported API provider: {self.model_api_provider}")

    def _chat_ollama(self, message, *args, stream=False, **kwargs):
        response = self.ollama_client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": message}],
        )
        return response["message"]["content"]

    def _chat_openai(self, message, *args, stream=False, **kwargs):
        # Set the API key directly
        openai.api_key = self.model_api_key

        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": message}],
        )

        # Extract the chatbot's message from the response.
        # Assuming there's at least one response and taking the last one as the chatbot's reply.
        result = response.choices[0].message.content
        return result

    def _chat_huggingface(self, message, *args, stream=False, **kwargs):
        """Chat using Hugging Face transformers."""
        is_tablellama = "tablellama" in self.model_name.lower()

        # Format the message as a conversation
        formatted_message = f"{message}"

        # Tokenize the input
        inputs = self.hf_tokenizer(
            formatted_message,
            return_tensors="pt",
            padding=True,
            truncation=True,
            pad_to_multiple_of=8 if is_tablellama else None,
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        default_kwargs = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
            "use_cache": True,
        }
        for default_k, default_v in default_kwargs.items():
            if default_k not in kwargs:
                kwargs[default_k] = default_v

        # Generate response
        with torch.no_grad():
            outputs = self.hf_model.generate(**inputs, **kwargs)

        # Decode the response
        response = self.hf_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return response.strip()
