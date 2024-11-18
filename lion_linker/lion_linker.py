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
                 kg="wikidata",  
                 batch_size=1000, 
                 mention_columns=None, 
                 api_limit=10,
                 compact_candidates=True,
                 model_api_provider='ollama', 
                 model_api_key=None,
                 gt_columns=None,
                 debug_mode=False,
                 debug_n_rows=10,
                 ):
        self.input_csv = input_csv
        self.prompt_file = prompt_file
        self.model_name = model_name
        self.api_url = api_url
        self.api_token = api_token
        self.api_limit = api_limit
        self.output_csv = output_csv
        self.kg = kg
        self.batch_size = batch_size
        self.mention_columns = mention_columns or []  # List of columns containing entity mentions
        self.compact_candidates = compact_candidates
        self.model_api_provider = model_api_provider
        self.model_api_key = model_api_key
        self.gt_columns = gt_columns or []  # Columns to exclude from processing
        self.debug_mode = debug_mode
        self.debug_n_rows = debug_n_rows

        # Clear or initialize the debug prompts file if debug mode is active
        if self.debug_mode:
            with open("debug_prompts.txt", 'w') as f:
                f.write("Debug Prompts Log\n\n")        

        logging.info('Initializing components...')
        # Initialize components
        self.api_client = APIClient(self.api_url, token=self.api_token, kg=self.kg, limit=self.api_limit, parse_response_func=parse_response)
        self.prompt_generator = PromptGenerator(self.prompt_file)
        logging.info(f'Model API provider is: {self.model_api_provider}')
        self.llm_interaction = LLMInteraction(self.model_name, self.model_api_provider, self.model_api_key)
        
        # Initialize the output CSV with headers
        pd.DataFrame(columns=['Identifier', 'LLM Answer', 'Extracted Identifier', 'Score']).to_csv(self.output_csv, index=False)
        
        logging.info('Setup completed.')
        self.table_summary = None  # Placeholder for the table summary

    def generate_table_summary(self, sample_data):
        # Exclude GT columns for testing
        sample_data = sample_data.drop(columns=self.gt_columns, errors='ignore')

        prompt = "Provide a high-level summary of the table without getting into specific details. reply only with the summary nothing else."
        
        # Prepare the summary with the prompt and sample data
        prompt += "\nHere is a sample of the table data:\n"
        prompt += sample_data.to_string(index=False)
        
        return self.llm_interaction.chat(prompt)

    async def process_chunk(self, chunk):
        # Exclude GT columns from the chunk
        chunk = chunk.drop(columns=self.gt_columns, errors='ignore')

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

                # Automatically save the prompt to debug_prompts.txt if debug mode is active
                if self.debug_mode:
                    with open("debug_prompts.txt", 'a') as f:
                        f.write(f"Row {id_row}, Column {column}:\n{prompt}\n\n")
                
                id_col = column_to_index[column]
                
                # Remove the file extension from the input_csv to use in the identifier
                base_filename = os.path.splitext(os.path.basename(self.input_csv))[0]
                
                # Creating identifier for the row and column
                identifier = f"{base_filename}-{id_row}-{id_col}"
                
                # Call LLM for each prompt individually
                response = self.llm_interaction.chat(prompt)
                
                # Extract identifier and score from response
                extracted_identifier, score = self.extract_id_and_score_from_response(response)
                
                results.append({
                    'Identifier': identifier,
                    'LLM Answer': response.replace('\n', '').strip(),  # Replace newlines with empty string
                    'Extracted Identifier': extracted_identifier,
                    'Score': score
                })
        
        return results

    def extract_id_and_score_from_response(self, response):
        """
        Extracts the identifier and confidence level from the structured response.
        Returns 'NIL' if no identifier is found and 'low' as the default confidence level.

        Parameters:
        response (str): The response text to extract from.

        Returns:
        tuple: (identifier, confidence) where identifier is the extracted ID or 'NIL',
            and confidence is one of ['low', 'medium', 'high'].
        """
        try:
            # Regex to match the pattern: id: <ID>, confidence: <low/medium/high>
            match = re.search(r'id:\s*(\S+),\s*confidence:\s*(low|medium|high)', response, re.IGNORECASE)
            
            if match:
                identifier = match.group(1)
                confidence = match.group(2).lower()
                return identifier, confidence
        except Exception as e:
            logging.error(f"Error parsing response: {response}. Error: {str(e)}")
        
        # Default values if no match
        return 'NIL', 'low'

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

            if self.debug_mode and self.debug_n_rows:
                return min(estimated_total_rows, self.debug_n_rows)
            
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

        # Adjust the total rows if running in debug mode
        if self.debug_mode and self.debug_n_rows:
            logging.info(f"Debug mode enabled. Processing the first {self.debug_n_rows} rows only.")
            estimated_total_rows = min(self.debug_n_rows, estimated_total_rows)

        # Initialize tqdm with the estimated total
        pbar = tqdm(total=estimated_total_rows, desc="Processing Batches")

        # Compute table summary with a sample from the first batch
        await self.compute_table_summary()
        
        # Track the actual number of rows processed
        actual_rows_processed = 0

        # Proceed with batch processing
        with pd.read_csv(self.input_csv, chunksize=self.batch_size) as reader:
            for chunk in reader:
                # If debug mode is enabled, limit the rows processed
                if self.debug_mode and self.debug_n_rows:
                    remaining_rows = self.debug_n_rows - actual_rows_processed
                    if remaining_rows <= 0:
                        break
                    chunk = chunk.head(remaining_rows)

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
