import logging
import os
import re

import pandas as pd
from tqdm.asyncio import tqdm

from lion_linker.core import APIClient, LLMInteraction
from lion_linker.prompt.generator import PromptGenerator
from lion_linker.utils import parse_response

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LionLinker:
    def __init__(
        self,
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
        model_api_provider="ollama",
        ollama_host=None,
        model_api_key=None,
        gt_columns=None,
        table_ctx_size: int = 1,
        format_candidates=True,
        example_file=None
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
        self.ollama_host = ollama_host
        self.model_api_key = model_api_key
        self.gt_columns = gt_columns or []  # Columns to exclude from processing
        self.table_ctx_size = table_ctx_size
        self.format_candidates = format_candidates
        self.example_file = example_file
        
        if self.table_ctx_size < 0:
            raise ValueError(
                "Table context size must be at least 0. "
                f"Got table context size: {self.table_ctx_size}"
            )
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be at least 1. Got batch size: {self.batch_size}")
        if self.batch_size < 2 * self.table_ctx_size + 1:
            raise ValueError(
                "Batch size must be at least 2 * table context size + 1. "
                f"Got batch size: {self.batch_size}, table context size: {self.table_ctx_size}"
            )

        logging.info("Initializing components...")
        # Initialize components
        self.api_client = APIClient(
            self.api_url,
            token=self.api_token,
            kg=self.kg,
            limit=self.api_limit,
            parse_response_func=parse_response,
        )
        self.prompt_generator = PromptGenerator(self.prompt_file, self.example_file)
        logging.info(f"Model API provider is: {self.model_api_provider}")
        self.llm_interaction = LLMInteraction(
            self.model_name, self.model_api_provider, self.ollama_host, self.model_api_key
        )

        # Initialize the output CSV with headers
        pd.DataFrame(columns=["Identifier", "LLM Answer", "Extracted Identifier"]).to_csv(
            self.output_csv, index=False
        )

        logging.info("Setup completed.")
        self.table_summary = None  # Placeholder for the table summary

    def generate_table_summary(self, sample_data):
        # Exclude GT columns for testing
        sample_data = sample_data.drop(columns=self.gt_columns, errors="ignore")

        prompt = (
            "Provide a high-level summary of the table without getting into specific details. "
            "Reply only with the summary nothing else."
        )

        # Prepare the summary with the prompt and sample data
        prompt += "\nHere is a sample of the table data:\n"
        prompt += sample_data.to_string(index=False)

        return self.llm_interaction.chat(prompt)

    async def process_chunk(self, chunk: pd.DataFrame):
        # Exclude GT columns from the chunk
        chunk = chunk.drop(columns=self.gt_columns, errors="ignore")

        # Check if mention_columns are present in the chunk
        missing_columns = [col for col in self.mention_columns if col not in chunk.columns]
        if missing_columns:
            logging.error(f"Columns not found in the data: {', '.join(missing_columns)}")
            raise ValueError(f"Columns not found: {', '.join(missing_columns)}")

        column_to_index = {col: id_col for id_col, col in enumerate(chunk.columns)}
        mentions = []
        for column in self.mention_columns:
            mentions.extend(chunk[column].dropna().unique())

        # Run the async fetch candidates function
        mentions_to_candidates = await self.api_client.fetch_multiple_entities(mentions)
        results = []
        for loc_idx, (id_row, row) in enumerate(chunk.iterrows()):
            table_view = chunk.iloc[
                max(0, loc_idx - self.table_ctx_size) : loc_idx + self.table_ctx_size + 1
            ]
            table_list = [table_view.columns.tolist()] + table_view.values.tolist()
            for column in self.mention_columns:
                entity_mention = row[column]
                candidates = mentions_to_candidates.get(entity_mention, [])
                prompt = self.prompt_generator.generate_prompt(
                    table=table_list,
                    table_metadata=None,
                    table_summary=self.table_summary,  # Use the precomputed table summary
                    column_name=column,
                    entity_mention=entity_mention,
                    candidates=candidates,
                    compact=self.compact_candidates,
                    format_candidates=self.format_candidates,
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

                results.append(
                    {
                        "Identifier": identifier,
                        "LLM Answer": response.replace(
                            "\n", ""
                        ).strip(),  # Replace newlines with empty string
                        "Extracted Identifier": extracted_identifier,
                    }
                )

        return results

    def extract_identifier_from_response(self, response):
        """
        Extracts the last QID from the response, or 'NIL' if NIL appears after the last QID.
        Returns 'No Identifier' if neither QID nor NIL is present.

        Parameters:
        response (str): The response text to extract from.

        Returns:
        str: The last QID, 'NIL' if 'NIL' appears after the last QID,
        or 'No Identifier' if neither is found.
        """
        # Find all QIDs in the response (assuming QIDs start with 'Q' followed by digits)
        qids = re.findall(r"Q\d+", response)

        # Check if 'NIL' appears in the response
        nil_position = response.rfind("NIL")

        # If there are no QIDs and no NIL, return 'No Identifier'
        if not qids and nil_position == -1:
            return "No Identifier"

        # If there are no QIDs but NIL is present, return 'NIL'
        if not qids:
            return "NIL"

        # Find the position of the last QID in the response
        last_qid_position = response.rfind(qids[-1])

        # Return 'NIL' if it appears after the last QID, otherwise return the last QID
        if nil_position > last_qid_position:
            return "NIL"

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
                chunk_bytes = chunk.apply(
                    lambda row: len(row.to_csv(index=False, header=False)), axis=1
                ).sum()
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
    
    async def generate_sample_prompt(self, index: int = 0, random_row: bool = False) -> dict:
        """
        Generates sample prompt(s) using a single row from the CSV file.
        
        If random_row is True, a random row from the first batch is selected.
        Otherwise, the row at the specified index (0-based) in the first batch is used.
        
        Returns:
            dict: A mapping of mention column names to the generated prompt.
        """
        # Ensure that the table summary has been computed.
        if self.table_summary is None:
            await self.compute_table_summary()

        # Read the first chunk from the CSV file.
        try:
            chunk_iter = pd.read_csv(self.input_csv, chunksize=self.batch_size)
            chunk = next(chunk_iter)
        except StopIteration:
            raise ValueError("Input CSV is empty or not accessible.")

        # Remove ground truth (GT) columns if provided.
        if self.gt_columns:
            chunk = chunk.drop(columns=self.gt_columns, errors="ignore")

        # Select a row from the chunk.
        if random_row:
            # Select a random row from the chunk.
            sample_row_df = chunk.sample(n=1)
            relative_position = chunk.index.get_loc(sample_row_df.index[0])
        else:
            # Use the row specified by the index parameter.
            if index < 0 or index >= len(chunk):
                raise ValueError(f"Index {index} is out of range for the first chunk (length {len(chunk)}).")
            relative_position = index

        # Extract the sample row.
        sample_row = chunk.iloc[relative_position]

        # Create table context for the selected row using the table context size.
        start_idx = max(0, relative_position - self.table_ctx_size)
        end_idx = relative_position + self.table_ctx_size + 1
        table_view = chunk.iloc[start_idx:end_idx]
        table_list = [table_view.columns.tolist()] + table_view.values.tolist()

        prompts = {}
        # Generate a prompt for each column that contains entity mentions.
        for col in self.mention_columns:
            if col not in chunk.columns:
                logging.error(f"Column '{col}' not found in CSV.")
                continue

            entity_mention = sample_row[col]
            # Fetch candidate entities for the mention.
            candidates_dict = await self.api_client.fetch_multiple_entities([entity_mention])
            candidates = candidates_dict.get(entity_mention, [])
            prompt = self.prompt_generator.generate_prompt(
                table=table_list,
                table_metadata=None,
                table_summary=self.table_summary,
                column_name=col,
                entity_mention=entity_mention,
                candidates=candidates,
                compact=self.compact_candidates,
                format_candidates=self.format_candidates,
            )
            prompts[col] = prompt

        return prompts
    
    async def send_prompt_sample(self, prompt_sample: dict) -> dict:
        """
        Sends each prompt in the provided prompt sample to the LLM and returns the responses.

        Args:
            prompt_sample (dict): A dictionary mapping mention column names to prompt texts.
            
        Returns:
            dict: A dictionary mapping each mention column to a dictionary with keys:
                  'prompt', 'response', and 'extracted_identifier'.
        """
        results = {}
        for col, prompt in prompt_sample.items():
            response = self.llm_interaction.chat(prompt)
            results[col] = {
                "response": response.replace("\n", "").strip(),
                "extracted_identifier": self.extract_identifier_from_response(response)
            }
        return results
    
    async def run(self):
        logging.info(f"Starting processing of {self.input_csv}...")

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
                pd.DataFrame(batch_results).to_csv(
                    self.output_csv,
                    mode="a",
                    header=False,
                    index=False,
                    quoting=1,
                    escapechar="\\",
                )

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

        logging.info(f"Processing completed. Results are incrementally saved to {self.output_csv}")
