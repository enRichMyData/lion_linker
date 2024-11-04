import os
import re
import pandas as pd
from lion_linker.core import APIClient, PromptGenerator, LLMInteraction
from lion_linker.utils import parse_response
import logging
from tqdm.asyncio import tqdm


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LionLinker:
    def __init__(self, 
                 input_csv, 
                 prompt_file, 
                 model_name, 
                 api_url, 
                 api_token, 
                 output_csv, 
                 batch_size=1000, 
                 mention_columns=None, 
                 api_limit=10,
                 compact_candidates=True):
        self.input_csv = input_csv
        self.prompt_file = prompt_file
        self.model_name = model_name
        self.api_url = api_url
        self.api_token = api_token
        self.api_limit = api_limit
        self.output_csv = output_csv
        self.batch_size = batch_size
        self.mention_columns = mention_columns or []  # List of columns containing entity mentions
        self.compact_candidates = compact_candidates

        logging.info('Initializing components...')
        # Initialize components
        self.api_client = APIClient(self.api_url, token=self.api_token, limit=self.api_limit, parse_response_func=parse_response)
        self.prompt_generator = PromptGenerator(self.prompt_file)
        logging.info(f'Model API provider is: {self.model_api_provider}')
        self.llm_interaction = LLMInteraction(self.model_name, self.model_api_provider, self.model_api_key)
        
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
        # Check if mention_columns are present in the chunk
        missing_columns = [col for col in self.mention_columns if col not in chunk.columns]
        if missing_columns:
            logging.error(f"Columns not found in the data: {', '.join(missing_columns)}")
            raise ValueError(f"Columns not found: {', '.join(missing_columns)}")

        column_to_index = {col:id_col for id_col, col in enumerate(chunk.columns)}
        mentions = []
        for column in self.mention_columns:
            mentions.extend(chunk[column].dropna().unique())

        # Run the async fetch candidates function
        mentions_to_candidates = await self.api_client.fetch_multiple_entities(mentions)
        results = []
        for id_row, row in chunk.iterrows():
            for column in self.mention_columns:
                entity_mention = row[column]
                candidates = mentions_to_candidates.get(entity_mention, [])
                row_str = ', '.join([f'{col}:{row[col]}' for col in chunk.columns])
                prompt = self.prompt_generator.generate_prompt(
                    self.table_summary,  # Use the precomputed table summary
                    row_str,
                    column,
                    entity_mention,
                    candidates,
                    compact=self.compact_candidates
                )
                id_col = column_to_index[column]
                
                # Remove the file extension from the input_csv to use in the identifier
                base_filename = os.path.splitext(os.path.basename(self.input_csv))[0]
                
                # Creating identifier for the row and column
                identifier = f"{base_filename}-{id_row}-{id_col}"
                
                # Call LLM for each prompt individually
                response = self.llm_interaction.chat(prompt)
                
                # Extract identifier from response
                extracted_identifier = self.extract_identifier_from_response(response)
                
                results.append({
                    'Identifier': identifier,
                    'LLM Answer': response.replace('\n', '').strip(),  # Replace newlines with empty string
                    'Extracted Identifier': extracted_identifier
                })
        
        return results

    def extract_identifier_from_response(self, response):
        """
        Extracts the last QID from the response, or 'NIL' if NIL appears after the last QID.
        Returns 'No Identifier' if neither QID nor NIL is present.
        
        Parameters:
        response (str): The response text to extract from.
        
        Returns:
        str: The last QID, 'NIL' if 'NIL' appears after the last QID, or 'No Identifier' if neither is found.
        """
        # Find all QIDs in the response (assuming QIDs start with 'Q' followed by digits)
        qids = re.findall(r'Q\d+', response)
        
        # Check if 'NIL' appears in the response
        nil_position = response.rfind('NIL')
        
        # If there are no QIDs and no NIL, return 'No Identifier'
        if not qids and nil_position == -1:
            return 'No Identifier'
        
        # If there are no QIDs but NIL is present, return 'NIL'
        if not qids:
            return 'NIL'
        
        # Find the position of the last QID in the response
        last_qid_position = response.rfind(qids[-1])
        
        # Return 'NIL' if it appears after the last QID, otherwise return the last QID
        if nil_position > last_qid_position:
            return 'NIL'
        
        return qids[-1]

    async def estimate_total_rows(self):
        # Get the size of the file in bytes
        file_size = os.path.getsize(self.input_csv)
        
        total_bytes = 0
        total_rows = 0
        chunks_to_sample = 5  # Number of chunks to sample

        with pd.read_csv(self.input_csv, chunksize=self.batch_size) as reader:
            for i, chunk in enumerate(reader):
                if i >= chunks_to_sample:
                    break
                # Sum up the size of each row in bytes
                chunk_bytes = chunk.apply(lambda row: len(row.to_csv(index=False, header=False)), axis=1).sum()
                total_bytes += chunk_bytes
                total_rows += len(chunk)

        if total_rows > 0:
            # Average size per row
            avg_row_size = total_bytes / total_rows
            
            # Estimate the total number of rows in the file
            estimated_total_rows = int(file_size / avg_row_size)
            
            return estimated_total_rows
        else:
            raise ValueError("Not enough data to estimate total rows")
        
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

        # Estimate the total number of rows
        estimated_total_rows = await self.estimate_total_rows()

        # Initialize tqdm with the estimated total
        pbar = tqdm(total=estimated_total_rows, desc="Processing Batches")

        # Compute table summary with a sample from the first batch
        await self.compute_table_summary()
        
        # Track the actual number of rows processed
        actual_rows_processed = 0

        # Proceed with batch processing
        with pd.read_csv(self.input_csv, chunksize=self.batch_size) as reader:
            for chunk in reader:
                batch_results = await self.process_chunk(chunk)
                # Append results to CSV, ensuring proper handling of newlines
                pd.DataFrame(batch_results).to_csv(self.output_csv, mode='a', header=False, index=False, quoting=1, escapechar='\\')
                
                # Update the progress bar with the actual number of rows processed
                actual_rows_processed += len(chunk)
                pbar.update(len(chunk))
                
                # If actual rows exceed estimated, adjust the progress bar total
                if actual_rows_processed > estimated_total_rows:
                    pbar.total = actual_rows_processed
                    pbar.refresh()

        # Ensure the progress bar reaches 100%
        pbar.n = pbar.total
        pbar.close()

        logging.info(f'Processing completed. Results are incrementally saved to {self.output_csv}')
