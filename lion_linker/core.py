import requests
import json
import ollama
import aiohttp
import asyncio
import copy
import os
from dotenv import load_dotenv
import openai 
from groq import Groq


class APIClient:
    def __init__(self, url, token=None, parse_response_func=None):
        self.url = url
        self.token = token
        self.parse_response_func = parse_response_func

    async def fetch_entities(self, query, session):
        params = {'name': query, 'limit': 10}
        if self.token:
            params['token'] = self.token
        async with session.get(self.url, params=params, ssl=False) as response:
            response.raise_for_status()
            response_json = await response.json()
            if self.parse_response_func:
                return self.parse_response_func(response_json)
            return response_json

    async def fetch_multiple_entities(self, queries):
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_entities(query, session) for query in queries]
            results = await asyncio.gather(*tasks)
            return dict(zip(queries, results))

class PromptGenerator:
    def __init__(self, prompt_file):
        with open(prompt_file, 'r') as file:
            self.template = file.read()

    def generate_prompt(self, table_summary, row, column_name, entity_mention, candidates):
        template = copy.deepcopy(self.template)
        # Optimize candidates list by reducing the verbosity of the JSON representation
        optimized_candidates = []
        for candidate in candidates:
            optimized_candidate = {
                "id": candidate["id"],
                "name": candidate["name"],
                "description": candidate["description"].split('.')[0],  # Only keep the first sentence
                "types": [{"id": t["id"], "name": t["name"]} for t in candidate["types"]]
            }
            optimized_candidates.append(optimized_candidate)
        
        # Convert optimized candidates list to a compact JSON string
        candidates_text = json.dumps(optimized_candidates, separators=(',', ':'))

        # Replace placeholders in the template with actual values
        # Define a dictionary with placeholders as keys and corresponding values
        replacements = {
            '[SUMMARY]': table_summary.strip(),
            '[ROW]': row,
            '[COLUMN NAME]': column_name,
            '[ENTITY MENTION]': entity_mention,
            '[CANDIDATES]': candidates_text
        }

        # Replace each placeholder using the dictionary
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, value)
        
        return template

class LLMInteraction:
    def __init__(self, model_name, model_api_provider, model_api_key=None):
        self.model_name = model_name
        self.model_api_provider = model_api_provider
        self.model_api_key = model_api_key

    def chat(self, message, stream=True):
        #load_dotenv()  # Load variables from .env
        if self.model_api_provider == 'ollama':
            return self._chat_ollama(message, stream)
        elif self.model_api_provider == 'openai':
            OPENAI_API_KEY = self.model_api_key
            return self._chat_openai(message, stream)
        elif self.model_api_provider == 'groq':
            return self._chat_groq(message, stream)
        else:
            raise ValueError(f"Unsupported API provider: {self.model_api_provider}")

    def _chat_ollama(self, message, stream):
        session = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': message}],
            stream=stream,
        )
        result = ""
        for s in session:
            result += s['message']['content']
        return result

    def _chat_openai(self, message, stream):
        # Set the API key directly
        openai.api_key = self.model_api_key
        
        # Call the OpenAI API
        response = openai.ChatCompletion.create(
        model=self.model_name,
        messages=[
        {
            "role": "user",  # Role can be "user", "system", or "assistant"
            "content": message  # The actual message content
        }
        ]
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
            ]
        )

        result=chat_completion.choices[0].message.content
        return result
        
   