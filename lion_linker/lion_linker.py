import pandas as pd
from lion_linker.core import APIClient, PromptGenerator, LLMInteraction
from lion_linker.utils import clean_data, parse_response
import logging
import asyncio
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LionLinker:
    def __init__(self, input_csv, prompt_file, model_name, api_url, api_token, output_csv, batch_size=1000, mention_columns=None):
        self.input_csv = input_csv
        self.prompt_file = prompt_file
        self.model_name = model_name
        self.api_url = api_url
        self.api_token = api_token
        self.output_csv = output_csv
        self.batch_size = batch_size
        self.mention_columns = mention_columns or []  # List of columns containing entity mentions

        logging.info('Initializing components...')
        # Initialize components
        self.api_client = APIClient(self.api_url, token=self.api_token, parse_response_func=parse_response)
        self.prompt_generator = PromptGenerator(self.prompt_file)
        self.llm_interaction = LLMInteraction(self.model_name)
        
        # Initialize the output CSV with headers
        pd.DataFrame(columns=['Identifier', 'LLM Answer', 'Extracted Identifier']).to_csv(self.output_csv, index=False)
        
        logging.info('Setup completed.')
        self.table_summary = None  # Placeholder for the table summary

    def generate_table_summary(self, sample_data):
        prompt = "Provide a high-level summary of the table without getting into specific details. reply only with the summary nothing else."
        
        # Prepare the summary with the prompt and sample data
        prompt += "\nHere is a sample of the table data:\n"
        prompt += sample_data.to_string(index=False)
        
        return self.llm_interaction.chat(prompt)

    async def process_chunk(self, chunk):
        cleaned_chunk = clean_data(chunk)

        mentions = []
        for column in self.mention_columns:
            mentions.extend(cleaned_chunk[column].dropna().unique())

        # Run the async fetch candidates function
        mentions_to_candidates = await self.api_client.fetch_multiple_entities(mentions)
        results = []
        for idx, row in cleaned_chunk.iterrows():
            for column in self.mention_columns:
                entity_mention = row[column]
                candidates = mentions_to_candidates.get(entity_mention, [])
                row_str = ', '.join([f'{col}:{row[col]}' for col in cleaned_chunk.columns])
                prompt = self.prompt_generator.generate_prompt(
                    self.table_summary,  # Use the precomputed table summary
                    row_str,
                    column,
                    entity_mention,
                    candidates
                )
                # Creating identifier for the row and column
                identifier = f"{self.input_csv}-{idx}-{column}"
                
                # Call LLM for each prompt individually
                response = self.llm_interaction.chat(prompt)
                
                # Extract identifier from response
                extracted_identifier = self.extract_identifier_from_response(response)
                
                results.append({
                    'Identifier': identifier,
                    'LLM Answer': response.replace('\n', ''),  # Replace newlines with empty string
                    'Extracted Identifier': extracted_identifier
                })
        
        return results

    def extract_identifier_from_response(self, response):
        # Implement your logic to extract an identifier from the LLM response
        # This is a placeholder function; replace with actual extraction logic
        return response.split()[0]  # Example extraction logic

    async def compute_table_summary(self):
        # Read the first batch from the CSV
        first_batch = pd.read_csv(self.input_csv, chunksize=self.batch_size)
        first_chunk = next(first_batch, None)
        
        if first_chunk is not None:
            # Sample a few rows from the first chunk
            sample_data = first_chunk.sample(min(5, len(first_chunk)))
            self.table_summary = self.generate_table_summary(sample_data)
        else:
            raise ValueError("Not enough data to compute table summary")

    async def run(self):
        logging.info(f'Starting processing of {self.input_csv}...')

        # Compute table summary with a sample from the first batch
        await self.compute_table_summary()
        
        # Proceed with batch processing
        with pd.read_csv(self.input_csv, chunksize=self.batch_size) as reader:
            for chunk in tqdm(reader, desc="Processing Batches"):
                batch_results = await self.process_chunk(chunk)
                # Append results to CSV, ensuring proper handling of newlines
                pd.DataFrame(batch_results).to_csv(self.output_csv, mode='a', header=False, index=False, quoting=1, escapechar='\\')
        
        logging.info(f'Processing completed. Results are incrementally saved to {self.output_csv}')