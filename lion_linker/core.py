import requests
import json
import ollama

class APIClient:
    def __init__(self, url, token=None, parse_response_func=None):
        self.url = url
        self.token = token
        self.parse_response_func = parse_response_func

    def fetch_entities(self, query):
        params = {'name': query, 'limit': 100}
        if self.token:
            params['token'] = self.token
        response = requests.get(self.url, params=params)
        response.raise_for_status()
        response_json = response.json()
        if self.parse_response_func:
            return self.parse_response_func(response_json)
        return response_json

class PromptGenerator:
    def __init__(self, prompt_file):
        with open(prompt_file, 'r') as file:
            self.template = file.read()

    def generate_prompt(self, table_summary, row, column_name, entity_mention, candidates):
        # Convert candidates list to a formatted JSON string
        candidates_text = json.dumps(candidates, indent=2)

        # Replace placeholders in the template with actual values
        # Define a dictionary with placeholders as keys and corresponding values
        replacements = {
            '[SUMMARY]': table_summary,
            '[ROW]': row,
            '[COLUMN NAME]': column_name,
            '[ENTITY MENTION]': entity_mention,
            '[CANDIDATES]': candidates_text
        }

        # Replace each placeholder using the dictionary
        for placeholder, value in replacements.items():
            self.template = self.template.replace(placeholder, value)
        
        return self.template

class LLMInteraction:
    def __init__(self, model_name):
        self.model_name = model_name

    def chat(self, message, stream=True):
        session = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': message}],
            stream=stream,
        )
        result = ""
        for s in session:
            result += s['message']['content']
        return result