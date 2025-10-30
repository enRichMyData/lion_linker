import json
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
    DEFAULT_ANSWER_FORMAT_TEMPLATE = """
        You must reply with a single JSON object exactly in the following structure:
        {{"answer":"<BEST_ID_OR_NIL>","{ranking_key}":[{{"rank":1,"id":"<CANDIDATE_ID>","score":0.9500,"label":"<CANDIDATE_LABEL>"}}]}}

        Strict rules:
        - Populate "answer" with the identifier of the best candidate. Use "NIL" if none apply.
        - Populate "{ranking_key}" with the top {ranking_size} candidates sorted from best to worst.
          If fewer than {ranking_size} candidates are available, include all of them.
        - Every candidate entry must contain:
          * "rank": an integer starting at 1 and increasing by 1 with each entry.
          * "id": the candidate identifier string.
          * "score": a float strictly between 0 and 1 (exclusive) with exactly {score_precision} decimal places.
            Scores must strictly decrease as rank increases.
          * "label": the short candidate name.
          * Optional "description": concise supporting context (omit if unavailable).
        - Do not add explanations, additional keys, Markdown, or text before/after the JSON.

        Example of a valid reply:
        {{"answer":"Q42","{ranking_key}":[{{"rank":1,"id":"Q42","score":0.9500,"label":"Douglas Adams"}},{{"rank":2,"id":"Q123","score":0.7300,"label":"Somebody Else"}}]}}
    """  # noqa

    ALLOWED_RANKING_SIZES = (3, 5)
    RANKING_KEY = "candidate_ranking"
    RANKING_SCORE_PRECISION = 4

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
        ranking_size: int = 5,
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
            ranking_size (int, optional): Number of candidates to expose in the ranking output.
                Must be either 3 or 5 and controls the scoring normalization.
                Defaults to 5.
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

        if ranking_size not in LionLinker.ALLOWED_RANKING_SIZES:
            raise ValueError(
                f"ranking_size must be one of {LionLinker.ALLOWED_RANKING_SIZES}. "
                f"Got ranking_size: {ranking_size}"
            )
        self.ranking_size = ranking_size

        default_answer_format = (
            answer_format
            if answer_format is not None
            else LionLinker.DEFAULT_ANSWER_FORMAT_TEMPLATE.format(
                ranking_key=LionLinker.RANKING_KEY,
                ranking_size=self.ranking_size,
                score_precision=LionLinker.RANKING_SCORE_PRECISION,
            ).strip()
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
            self.answer_format = default_answer_format
            self.table_summary = ""
            self.format_candidates = True
            self.prompt_template = "tablellama"
            pattern_str = kwargs.get("id_extraction_pattern", r'"answer"\s*:\s*"([^"]+)"')
        else:
            self.table_summary = None
            self.answer_format = default_answer_format
            pattern_str = kwargs.get("id_extraction_pattern", r'"answer"\s*:\s*"([^"]+)"')

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

    @classmethod
    def _validate_candidate_ranking(
        cls, formatted_ranking: str, entries: list[dict], requested_top_k: int
    ) -> list[dict]:
        if requested_top_k not in cls.ALLOWED_RANKING_SIZES:
            raise ValueError(
                f"requested_top_k must be one of {cls.ALLOWED_RANKING_SIZES}. "
                f"Got requested_top_k: {requested_top_k}"
            )

        expected_prefix = f'{{"{cls.RANKING_KEY}":'
        if not formatted_ranking.startswith(expected_prefix):
            raise ValueError(
                "Candidate ranking output must start with "
                f'{expected_prefix}. Received: {formatted_ranking}'
            )

        if not entries:
            if formatted_ranking != f'{{"{cls.RANKING_KEY}":[]}}':
                raise ValueError(
                    "Empty candidate rankings must be represented as "
                    f'{{"{cls.RANKING_KEY}":[]}}. Received: {formatted_ranking}'
                )
            return []

        if len(entries) > requested_top_k:
            raise ValueError(
                "Number of candidate ranking entries cannot exceed the requested top_k. "
                f"Entries: {len(entries)}, requested_top_k: {requested_top_k}"
            )

        previous_rank = 0
        previous_score = None
        normalized_entries: list[dict] = []
        for entry in entries:
            rank = entry.get("rank")
            if not isinstance(rank, int):
                raise ValueError(
                    "Candidate ranking 'rank' values must be integers. "
                    f"Received {rank!r} of type {type(rank)}."
                )

            if rank != previous_rank + 1:
                raise ValueError(
                    "Candidate ranking entries must be sequential starting from 1. "
                    f"Found rank sequence issue at rank {rank}."
                )

            score = entry.get("score")
            if not isinstance(score, (int, float)):
                raise ValueError(
                    "Candidate ranking scores must be numeric. "
                    f"Received type {type(score)} for rank {rank}."
                )

            score = round(float(score), cls.RANKING_SCORE_PRECISION)
            if not 0 < score < 1:
                raise ValueError(
                    "Candidate ranking scores must be strictly within the (0, 1) interval. "
                    f"Received score {score} for rank {rank}."
                )

            if previous_score is not None and score >= previous_score:
                raise ValueError(
                    "Candidate ranking scores must strictly decrease as rank increases. "
                    f"Received score {score} after {previous_score}."
                )

            entry_id = entry.get("id")
            if entry_id is None or entry_id == "":
                raise ValueError("Candidate ranking entries must include a non-empty 'id' value.")

            if not isinstance(entry_id, str):
                raise ValueError(
                    "Candidate ranking 'id' values must be strings. "
                    f"Received {entry_id!r} of type {type(entry_id)}."
                )

            label = entry.get("label")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(
                    "Candidate ranking entries must include a non-empty 'label' string. "
                    f"Received {label!r}."
                )

            description = entry.get("description")
            if description is not None and not isinstance(description, str):
                raise ValueError(
                    "Candidate ranking 'description' values must be strings when provided. "
                    f"Received {description!r} of type {type(description)}."
                )

            normalized_entry = {
                "rank": rank,
                "id": entry_id,
                "score": score,
                "label": label.strip(),
            }
            if description:
                normalized_entry["description"] = description.strip()

            normalized_entries.append(normalized_entry)
            previous_rank = rank
            previous_score = score

        return normalized_entries

    @classmethod
    def _parse_llm_json(
        cls, response: str, requested_top_k: int
    ) -> tuple[str, str, str, list[dict]]:
        if not response or not isinstance(response, str):
            raise ValueError("LLM response must be a non-empty string containing JSON.")

        response = response.strip()
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM response must be valid JSON. Received: {response}") from exc

        if not isinstance(payload, dict):
            raise ValueError("LLM response JSON must be an object with required keys.")

        if "answer" not in payload:
            raise ValueError('LLM response JSON must contain an "answer" field.')

        answer = payload["answer"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError('The "answer" field must be a non-empty string.')
        answer = answer.strip()

        if cls.RANKING_KEY not in payload:
            raise ValueError(
                f'LLM response JSON must contain a "{cls.RANKING_KEY}" field with candidate data.'
            )

        ranking_entries = payload[cls.RANKING_KEY]
        if not isinstance(ranking_entries, list):
            raise ValueError(
                f'"{cls.RANKING_KEY}" must be a list of candidate ranking objects. '
                f"Received: {type(ranking_entries)}"
            )

        normalized_entries = cls._validate_candidate_ranking(
            json.dumps({cls.RANKING_KEY: ranking_entries}, separators=(",", ":")),
            ranking_entries,
            requested_top_k,
        )

        canonical_payload = {
            "answer": answer,
            cls.RANKING_KEY: normalized_entries,
        }
        canonical_payload_str = json.dumps(canonical_payload, separators=(",", ":"))
        canonical_ranking_str = json.dumps(
            {cls.RANKING_KEY: normalized_entries}, separators=(",", ":")
        )

        return canonical_payload_str, answer, canonical_ranking_str, normalized_entries

    def _resolve_answer_identifier(self, answer: str, candidates: list[dict]) -> str:
        if not answer:
            return "No Identifier"

        if answer.upper() == "NIL":
            return "NIL"

        candidate_ids = {
            str(candidate.get("id")) for candidate in candidates if candidate.get("id") is not None
        }
        if answer in candidate_ids:
            return answer

        normalized_answer = answer.strip().lower()
        for candidate in candidates:
            name = candidate.get("name")
            if name and normalized_answer == str(name).strip().lower():
                return str(candidate.get("id"))

        return answer

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
                    response = '{"answer":"NIL","candidate_ranking":[]}'  # fallback

                # Extract identifier from response
                try:
                    (
                        canonical_payload,
                        answer_identifier_raw,
                        candidate_ranking,
                        _,
                    ) = LionLinker._parse_llm_json(response, self.ranking_size)
                    extracted_identifier = self._resolve_answer_identifier(
                        answer_identifier_raw, candidates
                    )
                except ValueError as parse_error:
                    logging.error(
                        "LLM response parsing failed for mention '%s': %s",
                        entity_mention,
                        parse_error,
                    )
                    canonical_payload = '{"answer":"NIL","candidate_ranking":[]}'
                    candidate_ranking = '{"candidate_ranking":[]}'
                    extracted_identifier = "No Identifier"

                # Add column-specific results to the row
                results_by_row[id_row][f"{column}_llm_answer"] = (
                    canonical_payload
                )
                results_by_row[id_row][f"{column}{self.prediction_suffix}"] = extracted_identifier
                results_by_row[id_row][f"{column}_candidate_ranking"] = candidate_ranking

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
        if response is None:
            return "No Identifier"

        try:
            _, answer, _, _ = LionLinker._parse_llm_json(response, self.ranking_size)
            return answer
        except ValueError:
            matches = self._compiled_id_pattern.findall(response)
            if matches:
                return matches[-1]
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
            try:
                canonical_payload, answer, candidate_ranking, _ = LionLinker._parse_llm_json(
                    response, self.ranking_size
                )
            except ValueError:
                canonical_payload = response.replace("\n", "").strip()
                answer = self.extract_identifier_from_response(response)
                candidate_ranking = '{"candidate_ranking":[]}'
            results[col] = {
                "response": canonical_payload,
                "extracted_identifier": answer,
                "candidate_ranking": candidate_ranking,
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
