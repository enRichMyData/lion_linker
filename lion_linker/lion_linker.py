
import pandas as pd
from lion_linker.core import APIClient, PromptGenerator, LLMInteraction
from lion_linker.utils import clean_data, parse_response
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LionLinker:
    def __init__(self, input_csv, prompt_file, model_name, api_url, api_token, output_csv, batch_size=1000):
        self.input_csv = input_csv
        self.prompt_file = prompt_file
        self.model_name = model_name
        self.api_url = api_url
        self.api_token = api_token
        self.output_csv = output_csv
        self.batch_size = batch_size

        logging.info('Initializing components...')
        # Initialize components
        self.api_client = APIClient(self.api_url, token=self.api_token, parse_response_func=parse_response)
        self.prompt_generator = PromptGenerator(self.prompt_file)
        self.llm_interaction = LLMInteraction(self.model_name)
        
        # Initialize the output CSV with headers
        pd.DataFrame(columns=['Row Index', 'Entity Mention', 'Generated Prompt']).to_csv(self.output_csv, index=False)
        
        logging.info('Setup completed.')

    def generate_table_summary(self, prompt=None):
        if prompt is None:
            prompt = "Provide a high-level summary of the table without getting into specific details. reply only with the summary nothing else."

        # Initialize an empty DataFrame
        df = pd.DataFrame()

        # Read the CSV file in chunks and concatenate them into a single DataFrame
        chunk_iter = pd.read_csv(self.input_csv, chunksize=1000)
        for chunk in chunk_iter:
            df = pd.concat([df, chunk], ignore_index=True)
        
        # Sample 5 rows from the DataFrame
        sample = df.sample(5)
        
        # Prepare the summary with the prompt and sample data
        prompt += "\nHere is a sample of the table data:\n"
        prompt += sample.to_string(index=False)
        
        return self.llm_interaction.chat(prompt)

    def process_chunk(self, chunk):
        cleaned_chunk = clean_data(chunk)
        table_summary = self.generate_table_summary(cleaned_chunk)
        
        results = []
        for index, row in cleaned_chunk.iterrows():
            entity_mention = row['title']  # Adjust this according to the actual column name
            candidates = self.api_client.fetch_entities(entity_mention)
            
            prompt = self.prompt_generator.generate_prompt(
                table_summary,
                row.to_dict(),
                'title',
                entity_mention,
                candidates
            )
            results.append({
                'Row Index': index,
                'Entity Mention': entity_mention,
                'Generated Prompt': prompt
            })
        
        return results

    def run(self):
        logging.info(f'Starting processing of {self.input_csv} in batches...')
        
        with pd.read_csv(self.input_csv, chunksize=self.batch_size) as reader:
            for chunk in tqdm(reader, desc="Processing Batches"):
                batch_results = self.process_chunk(chunk)
                pd.DataFrame(batch_results).to_csv(self.output_csv, mode='a', header=False, index=False)
        
        logging.info(f'Processing completed. Results are incrementally saved to {self.output_csv}')
