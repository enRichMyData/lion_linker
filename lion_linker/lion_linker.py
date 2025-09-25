import logging
import os
import re
from pathlib import Path

import pandas as pd
from tqdm.asyncio import tqdm

from lion_linker.core import LLMInteraction
from lion_linker.prompt.generator import PromptGenerator
from lion_linker.retrievers import RetrieverClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LionLinker:
    DEFAULT_ANSWER_FORMAT = """
        Identify the correct identifier (ID) for the entity mention from the list of candidates above.

        Respond using the following format, and nothing else:

        ANSWER:{ID}

        Instructions:
        - Replace {ID} with the actual identifier (e.g., Q42, apple-234abc or Apple)
        - If none of the candidates is correct, respond with: ANSWER:NIL
        - Do not add any explanations, extra text, or formatting.
        - The output must be exactly one line and must start with 'ANSWER:'

        Examples:
        - ANSWER:Q42
        - ANSWER:apple-234abc
        - ANSWER:Apple
        - ANSWER:NIL
    """  # noqa

    def __init__(
        self,
        input_csv: str | Path,
        model_name: str,
        retriever: RetrieverClient,
        output_csv: str | None = None,
        prompt_template: str = "base",
        chunk_size: int = 64,
        mention_columns: list | None = None,
        format_candidates: bool = True,
        compact_candidates: bool = True,
        model_api_provider: str = "ollama",
        ollama_host: str | None = None,
        model_api_key: str | None = None,
        few_shot_examples_file_path: str | None = None,
        gt_columns: list | None = None,
        table_ctx_size: int = 1,
        answer_format: str | None = None,
        **kwargs,
    ):
        """Initialize a LionLinker instance.

        Parameters:
            input_csv (str | Path): The file path to the input CSV file.
            model_name (str): The name of the model to use.
            retriever (RetrieverClient): An instance of RetrieverClient used to fetch candidates
                from the KB.
            output_csv (str, optional): The file path to the output CSV file.
                If not provided, the output file will be named based on the input file,
                with '_output' appended before the extension.
                Defaults to None.
            prompt_template (str, optional): The type of the template to use
                ('base', 'detailed', 'few_shot' or 'tablellama') or a file path to the prompt file.
                Defaults to 'base'.
            chunk_size (int, optional): The size of the chunks to process.
                Defaults to 64.
            mention_columns (list, optional): Columns to consider for mentions.
                Defaults to None.
            format_candidates (bool, optional): Whether to format the candidate results as in
                TableLlama (https://arxiv.org/abs/2311.09206).
                Defaults to True.
            compact_candidates (bool, optional): Whether to compact candidate entries.
                This is only used if `format_candidates` is False.
                Defaults to True.
            model_api_provider (str, optional): The provider for the model API.
                Supported providers: "ollama", "openrouter", "huggingface".
                Defaults to "ollama".
            ollama_host (str, optional): The host for the Ollama service.
                Defaults to None.
            model_api_key (str, optional): The API key for the model service.
                Required for "openrouter" providers.
                Defaults to None.
            few_shot_examples_file_path (str, optional): The file path to
                the few shot examples file.
                Defaults to None.
            gt_columns (list, optional): List of ground truth columns for reference.
                Defaults to None.
            table_ctx_size (int, optional): The context size for table data.
                This is the number of rows to include before and after the current row.
                Defaults to 1.
            answer_format (str, optional): The format for the answer.
                Defaults to None.
            **kwargs: Additional keyword arguments.
                      Supported hidden features:
                      - mention_to_qids: Dict mapping mentions to entity IDs to force in candidates
                      - id_extraction_pattern: Regex pattern for extracting IDs from responses
                      - prediction_suffix: Suffix for prediction column names
                      - kg_name: Name of the knowledge graph being used
                      - compute_table_summary: Whether to create table summary or not
                      - target_rows_ids: an iterable specifying a subset of rows to be processed
        """

        if not os.path.exists(input_csv) or os.path.splitext(input_csv)[1] != ".csv":
            raise ValueError(
                "Input CSV file does not exist or is not a CSV file." f"Input file: {input_csv}"
            )
        self.input_csv = input_csv
        self.prompt_template = prompt_template
        self.model_name = model_name
        self.retriever = retriever
        self.output_csv = output_csv
        if not self.output_csv:
            self.output_csv = os.path.splitext(input_csv)[0] + "_output.csv"
        self.chunk_size = chunk_size
        self.mention_columns = mention_columns or []  # List of columns containing entity mentions
        self.format_candidates = format_candidates
        self.compact_candidates = compact_candidates
        self.model_api_provider = model_api_provider
        self.ollama_host = ollama_host
        self.model_api_key = model_api_key
        self.few_shot_examples_file_path = few_shot_examples_file_path
        # Use a cleaned DEFAULT_ANSWER_FORMAT with extra spaces and newlines removed
        self.gt_columns = gt_columns or []  # Columns to exclude from processing
        self.table_ctx_size = table_ctx_size
        if self.table_ctx_size < 0:
            raise ValueError(
                "Table context size must be at least 0. "
                f"Got table context size: {self.table_ctx_size}"
            )
        if self.chunk_size < 1:
            raise ValueError(f"Chuck size must be at least 1. Got batch size: {self.chunk_size}")
        if self.chunk_size < 2 * self.table_ctx_size + 1:
            raise ValueError(
                "Batch size must be at least 2 * table context size + 1. "
                f"Got batch size: {self.chunk_size}, table context size: {self.table_ctx_size}"
            )

        logging.info(f"Model API provider is: {self.model_api_provider}")
        self.llm_interaction = LLMInteraction(
            self.model_name, self.model_api_provider, self.ollama_host, self.model_api_key
        )

        # Hidden parameter for mention to entity ID mapping
        self._mention_to_qids = kwargs.get("mention_to_qids", {})
        self.prediction_suffix = kwargs.get("prediction_suffix", "_pred_id")
        self.kg_name = kwargs.get("kg_name", "generic")
        self._target_row_ids: set[int] | None = None
        target_rows = kwargs.get("target_row_ids")
        if target_rows is not None:
            try:
                self._target_row_ids = {int(r) for r in target_rows}
            except (TypeError, ValueError) as exc:
                raise ValueError("target_row_ids must be an iterable of integers") from exc

        # TableLlama checks
        if "osunlp" in model_name:
            self.answer_format = ""
            self.table_summary = ""
            self.format_candidates = True
            self.prompt_template = "tablellama"
            pattern_str = kwargs.get("id_extraction_pattern", r"Q\d+")
        else:
            self.table_summary = None
            self.answer_format = answer_format or " ".join(
                LionLinker.DEFAULT_ANSWER_FORMAT.split()
            )
            pattern_str = kwargs.get("id_extraction_pattern", r"ANSWER:\s*([^\s]+)")

        logging.info("Initializing components...")
        self.prompt_generator = PromptGenerator(
            self.prompt_template,
            self.few_shot_examples_file_path,
            tablellama_format="osunlp" in model_name,
        )

        self._compiled_id_pattern = re.compile(pattern_str, re.IGNORECASE)
        logging.info(f"Knowledge graph: {self.kg_name}")

        # logging.info("Pulling model if necessary...")
        # self.llm_interaction.ollama_client.pull(self.model_name)
        logging.info("Setup completed.")

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
        if self._target_row_ids is not None:
            target_mask = chunk.index.isin(self._target_row_ids)
            mention_source = chunk[target_mask]
        else:
            mention_source = chunk

        mentions = []
        for column in self.mention_columns:
            mentions.extend(mention_source[column].dropna().unique())

        # Run the async fetch candidates function with the hidden mention_to_qids parameter
        kwargs = {}
        if self._mention_to_qids:
            kwargs["mention_to_qids"] = self._mention_to_qids

        if mentions:
            mentions_to_candidates = await self.retriever.fetch_multiple_entities(
                mentions, **kwargs
            )
        else:
            mentions_to_candidates = {}

        # Group results by row ID
        results_by_row = {}

        target_ids = self._target_row_ids
        for loc_idx, (id_row, row) in enumerate(chunk.iterrows()):
            if target_ids is not None and id_row not in target_ids:
                continue
            table_view = chunk.iloc[
                max(0, loc_idx - self.table_ctx_size) : loc_idx + self.table_ctx_size + 1
            ]
            table_list = [table_view.columns.tolist()] + table_view.values.tolist()

            # Initialize row result if not exists
            if id_row not in results_by_row:
                results_by_row[id_row] = {"id_row": id_row}

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
                    answer_format=self.answer_format,
                )
                id_col = column_to_index[column]

                # Remove the file extension from the input_csv to use in the identifier
                base_filename = os.path.splitext(os.path.basename(self.input_csv))[0]

                # Creating identifier for the row and column
                f"{base_filename}-{id_row}-{id_col}"

                # Call LLM for each prompt individually
                try:
                    response = self.llm_interaction.chat(prompt)
                except Exception as e:
                    logging.error(f"LLM interaction failed for mention '{entity_mention}': {e}")
                    response = "ANSWER:NIL"  # fallback or you could raise if preferred

                # Extract identifier from response
                if "osunlp" in self.model_name:
                    candidates2qid = {
                        " ".join(
                            (
                                candidate["name"]
                                + " [DESCRIPTION] "
                                + (
                                    candidate["description"]
                                    if candidate["description"] is not None
                                    else "None"
                                )
                                + " [TYPE] "
                                + ",".join(
                                    [
                                        t["name"]
                                        for t in candidate["types"]
                                        if t["name"] is not None
                                    ]
                                )
                            )
                            .lower()
                            .split()
                        ): candidate["id"]
                        for candidate in candidates
                    }
                    if response is not None:
                        response = response.lower()
                        response = response.replace("<", "", 1)
                        response = "".join(response.rsplit(">", 1))
                        response = " ".join(response.split())
                        extracted_identifier = candidates2qid.get(response, "No Identifier")
                    else:
                        extracted_identifier = "None"
                else:
                    extracted_identifier = self.extract_identifier_from_response(response)

                # Add column-specific results to the row
                results_by_row[id_row][f"{column}_llm_answer"] = (
                    " ".join(response.split()) if response is not None else "None"
                )
                results_by_row[id_row][f"{column}{self.prediction_suffix}"] = extracted_identifier

        # Convert to list of dictionaries
        results = list(results_by_row.values())
        return results

    def extract_identifier_from_response(self, response):
        """
        Extracts the identifier from a response using the compiled answer pattern.
        Supports arbitrary ID formats such as Wikidata IDs (e.g., Q42),
        Crunchbase slugs (e.g., apple-234abc), DBpedia IRIs (e.g., dbo:Apple), or NIL.

        Parameters:
            response (str): The model's response text.

        Returns:
            str: The extracted identifier (e.g., 'Q42', 'apple-234abc', 'Apple') or 'NIL',
                or 'No Identifier' if no valid match is found.
        """
        # Look for all matches with optional whitespace after 'ANSWER:'
        matches = self._compiled_id_pattern.findall(response)

        if matches:
            return matches[-1]  # Return the last match found

        return "No Identifier"

    async def estimate_total_rows(self):
        # Get the size of the file in bytes
        file_size = os.path.getsize(self.input_csv)

        total_bytes = 0
        total_rows = 0
        chunks_to_sample = 5  # Number of chunks to sample

        with pd.read_csv(self.input_csv, chunksize=self.chunk_size) as reader:
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
        first_batch = pd.read_csv(self.input_csv, chunksize=self.chunk_size)
        first_chunk = next(first_batch, None)

        if first_chunk is not None:
            # Sample a few rows from the first chunk
            sample_data = first_chunk.sample(min(5, len(first_chunk)))
            if self.table_summary is None:
                self.table_summary = self.generate_table_summary(sample_data)
        else:
            raise ValueError("Not enough data to compute table summary")

    async def generate_sample_prompt(
        self, random_row: bool = True, row_index: int | None = None
    ) -> dict:
        """
        Generates sample prompt(s) using a single row from the CSV file.
        If random_row is True, a random row from the first batch is selected;
        otherwise, the first row is used.

        Returns:
            dict: A mapping of mention column names to the generated prompt.
        """
        # Ensure that the table summary has been computed.
        if self.table_summary is None:
            await self.compute_table_summary()

        # Read the first chunk from the CSV file.
        try:
            chunk_iter = pd.read_csv(self.input_csv, chunksize=self.chunk_size)
            chunk = next(chunk_iter)
        except StopIteration:
            raise ValueError("Input CSV is empty or not accessible.")

        # Remove ground truth (GT) columns if provided.
        if self.gt_columns:
            chunk = chunk.drop(columns=self.gt_columns, errors="ignore")

        # Select a row from the chunk.
        if random_row and row_index is None:
            # Select a random row from the chunk.
            sample_row_df = chunk.sample(n=1)
            relative_position = chunk.index.get_loc(sample_row_df.index[0])
        else:
            # Use the row specified by the index parameter (default to 0 if not provided).
            if row_index is None:
                row_index = 0
            if row_index < 0 or row_index >= len(chunk):
                raise ValueError(
                    f"Index {row_index} is out of range for the first chunk (length {len(chunk)})."
                )
            relative_position = row_index

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
            # Fetch candidate entities for the mention with mention_to_qids if available
            kwargs = {}
            if self._mention_to_qids:
                kwargs["mention_to_qids"] = self._mention_to_qids

            candidates_dict = await self.retriever.fetch_multiple_entities(
                [entity_mention], **kwargs
            )
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
                answer_format=self.answer_format,
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
                "extracted_identifier": self.extract_identifier_from_response(response),
            }
        return results

    async def run(self):
        logging.info(f"Starting processing of {self.input_csv}...")

        # Estimate the total number of rows
        estimated_total_rows = await self.estimate_total_rows()

        # Initialize tqdm with the estimated total
        pbar = tqdm(total=estimated_total_rows, desc="Processing Batches")

        # Compute table summary with a sample from the first batch
        if self.table_summary is None:
            await self.compute_table_summary()

        # Track the actual number of rows processed
        actual_rows_processed = 0

        # Proceed with batch processing
        wrote_header = False

        with pd.read_csv(self.input_csv, chunksize=self.chunk_size) as reader:
            for chunk_id, chunk in enumerate(reader):
                batch_results = await self.process_chunk(chunk)
                batch_results = pd.DataFrame(batch_results)

                # Merge batch_results with the chunk
                if "id_row" not in chunk.columns:
                    chunk = chunk.reset_index().rename(columns={"index": "id_row"})
                merged_chunk = chunk.merge(batch_results, on="id_row", how="left")

                actual_rows_processed += len(chunk)
                pbar.update(len(chunk))
                if actual_rows_processed > estimated_total_rows:
                    pbar.total = actual_rows_processed
                    pbar.refresh()

                if self._target_row_ids is not None:
                    merged_chunk = merged_chunk[merged_chunk["id_row"].isin(self._target_row_ids)]

                if merged_chunk.empty:
                    continue

                if not wrote_header:
                    # Write the header for the first chunk
                    merged_chunk.to_csv(
                        self.output_csv,
                        mode="w",
                        index=False,
                        quoting=1,
                        escapechar="\\",
                    )
                    wrote_header = True
                else:
                    # Append results to CSV, ensuring proper handling of newlines
                    merged_chunk.to_csv(
                        self.output_csv,
                        mode="a",
                        header=False,
                        index=False,
                        quoting=1,
                        escapechar="\\",
                    )

        # Ensure the progress bar reaches 100%
        pbar.n = pbar.total
        pbar.close()

        logging.info(f"Processing completed. Results are incrementally saved to {self.output_csv}")
