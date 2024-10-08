import json
import ollama
import aiohttp
import asyncio
import copy


class APIClient:
    def __init__(self, url, token=None, parse_response_func=None, max_retries=3, backoff_factor=0.5, limit=10):
        self.url = url
        self.token = token
        self.parse_response_func = parse_response_func
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.limit = limit

    async def fetch_entities(self, query, session):
        params = {'name': query, 'limit': self.limit}
        if self.token:
            params['token'] = self.token

        retries = 0
        while retries < self.max_retries:
            try:
                async with session.get(self.url, params=params, ssl=False) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    if self.parse_response_func:
                        return self.parse_response_func(response_json)
                    return response_json
            except aiohttp.ClientResponseError as e:
                if e.status in {502, 503, 504}:  # Server errors
                    retries += 1
                    wait_time = self.backoff_factor * (2 ** retries)  # Exponential backoff
                    print(f"Error {e.status}, retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"ClientResponseError: {e}")
                    raise
            except aiohttp.ClientConnectionError as e:
                print(f"ConnectionError: {e}")
                raise
            except Exception as e:
                print(f"Unexpected error: {e}")
                raise

        raise Exception(f"Failed to fetch after {self.max_retries} retries")

    async def fetch_multiple_entities(self, queries):
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_entities(query, session) for query in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle failed requests (e.g., those returning exceptions)
            output = {}
            for query, result in zip(queries, results):
                if isinstance(result, Exception):
                    output[query] = f"Error: {str(result)}"
                else:
                    output[query] = result
            return output

class PromptGenerator:
    def __init__(self, prompt_file):
        with open(prompt_file, 'r') as file:
            self.template = file.read()

    def generate_prompt(self, table_summary, row, column_name, entity_mention, candidates, compact=True):
        template = copy.deepcopy(self.template)
        # Optimize candidates list by reducing the verbosity of the JSON representation
        optimized_candidates = []
        for candidate in candidates:
            optimized_candidate = {
                "id": candidate["id"],
                "name": candidate["name"],
                "description": candidate["description"],  
                "types": [{"id": t["id"], "name": t["name"]} for t in candidate["types"]]
            }
            optimized_candidates.append(optimized_candidate)
        
        if compact:
            # Convert optimized candidates list to a compact JSON string
            candidates_text = json.dumps(optimized_candidates, separators=(',', ':'))
        else:
            # Convert optimized candidates list to a pretty-printed JSON string
            candidates_text = json.dumps(optimized_candidates, indent=2)
        
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
            template = template.replace(placeholder, str(value))
        
        return template

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