from __future__ import annotations

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
        Reply with a single JSON object in exactly this shape (no extra text):
        {{
            "{ranking_key}": [
                {{
                "id": "<CANDIDATE_ID>",
                "confidence_label": "<HIGH|MEDIUM|LOW|null>",
                "confidence_score": <float|null>
                }}
            ],
            "explanation": "<short reasoning of the final decision>"
        }}

        Instructions:
        1. Always return the JSON object with both fields "{ranking_key}" and "explanation".
        2. Return between 1 and {ranking_size} candidates, sorted by descending "confidence_score", except in the NIL case.
        3. Each candidate must have the keys "id", "confidence_label", "confidence_score".
        4. "confidence_score" must lie in [0, 1]. Map it to "confidence_label" as:
        - HIGH if confidence_score ≥ 0.70
        - MEDIUM if 0.40 ≤ confidence_score < 0.70
        - LOW if confidence_score < 0.40
        5. If none of the provided candidates fits, return:
        - First entry:
            {{
            "id": "NIL",
            "confidence_label": "LOW",
            "confidence_score": 0
            }}
        - Then append all the original candidates (top5) in the same order as presented in the prompt, each with:
            "confidence_label": null,
            "confidence_score": null
        Use "explanation" to state why NIL was selected.
        6. Do not invent candidates. Score only the candidates that were provided in the prompt.
        7. Keep the output strictly valid JSON with no Markdown and no trailing commas.
        8. In explanation do not use double quotes inside the field so "explanation":"<your explanation here> (no double quotes in here but single quote is allowed)".
    """.strip()

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
        chunk_size: int = 16,
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
        match_confidence_threshold: float = 0.5,
        nil_insert_delta: float = 0.05,
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
            ranking_size (int, optional): Maximum number of candidates to request from the LLM.
                Must be either 3 or 5. Defaults to 5.
            match_confidence_threshold (float, optional): Minimum probability the top-ranked
                candidate must achieve (combined with a HIGH confidence label) to be marked as
                the final match. Must fall within (0, 1]. Defaults to 0.5.
            nil_insert_delta (float, optional): Deprecated; retained for backwards compatibility.
                It no longer affects the ranking behaviour.
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

        if not 0 < match_confidence_threshold <= 1:
            raise ValueError(
                "match_confidence_threshold must be within the (0, 1] interval. "
                f"Got: {match_confidence_threshold}"
            )
        self.match_confidence_threshold = match_confidence_threshold

        if nil_insert_delta < 0:
            raise ValueError(
                "nil_insert_delta must be non-negative. "
                f"Got: {nil_insert_delta}"
            )
        self.nil_insert_delta = nil_insert_delta

        default_answer_format = (
            answer_format
            if answer_format is not None
            else LionLinker.DEFAULT_ANSWER_FORMAT_TEMPLATE.format(
                ranking_key=LionLinker.RANKING_KEY,
                ranking_size=self.ranking_size,
            )
        )

        logging.info(f"Model API provider is: {self.model_api_provider}")
        self.llm_interaction = LLMInteraction(
            self.model_name,
            self.model_api_provider,
            self.ollama_host,
            self.model_api_key,
            ollama_headers=kwargs.get("ollama_headers"),
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
            pattern_str = kwargs.get("id_extraction_pattern", r'"id"\s*:\s*"([^"]+)"')
        else:
            self.table_summary = None
            self.answer_format = default_answer_format
            pattern_str = kwargs.get("id_extraction_pattern", r'"id"\s*:\s*"([^"]+)"')

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
        self._prompt_candidate_cache: dict[str, list[dict]] = {}

    @classmethod
    def _validate_candidate_ranking(cls, entries: list[dict], requested_top_k: int) -> list[dict]:
        if requested_top_k not in cls.ALLOWED_RANKING_SIZES:
            raise ValueError(
                f"requested_top_k must be one of {cls.ALLOWED_RANKING_SIZES}. "
                f"Got requested_top_k: {requested_top_k}"
            )

        if entries is None:
            entries = []

        if not isinstance(entries, list):
            raise ValueError("Candidate ranking must be provided as a list of objects.")

        nil_mode = False
        if entries:
            first_entry = entries[0]
            if isinstance(first_entry, dict):
                first_id = first_entry.get("id")
                if isinstance(first_id, str) and first_id.strip().upper() == "NIL":
                    nil_mode = True

        max_entries = requested_top_k + 1 if nil_mode else requested_top_k

        normalized_with_order: list[dict] = []
        for order_idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError("Each candidate ranking entry must be a JSON object.")

            raw_id = entry.get("id")
            if not isinstance(raw_id, str):
                raise ValueError("Candidate ranking entries must include a string 'id'.")
            candidate_id = raw_id.strip()
            if not candidate_id:
                raise ValueError("Candidate ranking entries must include a non-empty string 'id'.")

            score_value = entry.get("confidence_score")
            score: float | None
            if score_value is None:
                if not nil_mode or order_idx == 0:
                    raise ValueError(
                        "confidence_score must be numeric. "
                        f"Received null for id {candidate_id}."
                    )
                score = None
            else:
                if not isinstance(score_value, (int, float)):
                    raise ValueError(
                        "confidence_score must be numeric. "
                        f"Received type {type(score_value)} for id {candidate_id}."
                    )
                score = round(float(score_value), cls.RANKING_SCORE_PRECISION)
                if not 0 <= score <= 1:
                    raise ValueError(
                        "confidence_score must be within [0, 1]. "
                        f"Received {score} for id {candidate_id}."
                    )

            label_value = entry.get("confidence_label")
            if isinstance(label_value, str):
                canonical_label = label_value.strip().upper() or None
            elif label_value is None:
                canonical_label = None
            else:
                raise ValueError(
                    "confidence_label must be a string or null. "
                    f"Received type {type(label_value)} for id {candidate_id}."
                )

            normalized_with_order.append(
                {
                    "_order": order_idx,
                    "id": candidate_id,
                    "confidence_label": canonical_label,
                    "confidence_score": score,
                }
            )

        if not normalized_with_order:
            return []

        trimmed: list[dict] = []
        seen_ids: set[str] = set()

        if nil_mode:
            iterable = normalized_with_order
        else:
            iterable = sorted(
                normalized_with_order,
                key=lambda item: (-item["confidence_score"], item["_order"]),
            )

        for item in iterable:
            candidate_id_upper = item["id"].upper()
            if candidate_id_upper in seen_ids:
                continue
            seen_ids.add(candidate_id_upper)
            trimmed.append(
                {
                    "id": item["id"],
                    "confidence_label": item["confidence_label"],
                    "confidence_score": item["confidence_score"],
                }
            )
            if len(trimmed) >= max_entries:
                break

        return trimmed

    DEFAULT_EXPLANATION_FALLBACK = "LLM response did not include an explanation."

    def _parse_llm_json(self, response: str) -> tuple[list[dict], float | None, str]:
        if not response or not isinstance(response, str):
            raise ValueError("LLM response must be a non-empty string containing JSON.")

        response = response.strip()
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM response must be valid JSON. Received: {response}") from exc

        if not isinstance(payload, dict):
            raise ValueError("LLM response JSON must be an object containing the ranking key only.")

        unexpected_keys = set(payload.keys()) - {
            self.RANKING_KEY,
            "nil_score",
            "explanation",
        }
        if unexpected_keys:
            raise ValueError(
                "LLM response must not contain unexpected top-level keys. "
                f"Found: {', '.join(sorted(unexpected_keys))}"
            )

        if self.RANKING_KEY not in payload:
            raise ValueError(
                f'LLM response JSON must contain a "{self.RANKING_KEY}" list.'
            )

        ranking_entries = payload[self.RANKING_KEY]
        normalized_entries = LionLinker._validate_candidate_ranking(
            ranking_entries,
            self.ranking_size,
        )

        nil_score: float | None = None
        if "nil_score" in payload:
            nil_score_value = payload["nil_score"]
            if not isinstance(nil_score_value, (int, float)):
                raise ValueError("nil_score must be numeric.")
            nil_score = float(nil_score_value)
            if not 0 <= nil_score <= 1:
                raise ValueError("nil_score must be within [0, 1].")

        if "explanation" not in payload:
            raise ValueError(
                'LLM response JSON must contain an "explanation" string summarizing the decision.'
            )
        explanation_raw = payload["explanation"]
        if not isinstance(explanation_raw, str):
            raise ValueError('"explanation" must be a string.')
        explanation = explanation_raw.strip() or self.DEFAULT_EXPLANATION_FALLBACK

        return normalized_entries, nil_score, explanation


    def _default_nil_payload(self) -> tuple[list[dict], float | None, str]:
        return [], None, self.DEFAULT_EXPLANATION_FALLBACK

    def _determine_predicted_identifier(
        self, ranking_entries: list[dict], nil_score: float | None
    ) -> str:
        if nil_score is not None:
            nil_score = max(0.0, min(1.0, float(nil_score)))

        if not ranking_entries:
            return "NIL"

        top_entry = ranking_entries[0]
        candidate_id = str(top_entry.get("id", "")).strip()
        if not candidate_id or candidate_id.upper() == "NIL":
            return "NIL"

        score = float(top_entry.get("confidence_score", 0.0))
        label = str(top_entry.get("confidence_label", "")).upper()

        if nil_score is not None and nil_score >= max(score, self.match_confidence_threshold):
            return "NIL"

        if score >= self.match_confidence_threshold and label == "HIGH":
            return candidate_id

        return "NIL"

    def _enrich_candidate_ranking(
        self,
        ranked_entries: list[dict],
        candidates: list[dict],
        predicted_identifier: str,
        nil_score: float | None = None,
    ) -> list[dict]:
        if nil_score is not None:
            nil_score = max(0.0, min(1.0, float(nil_score)))

        candidate_lookup: dict[str, dict] = {}
        for candidate in candidates:
            candidate_id = candidate.get("id")
            if candidate_id is None:
                continue
            candidate_id_str = str(candidate_id).strip()
            if not candidate_id_str:
                continue
            candidate_lookup[candidate_id_str] = candidate
            candidate_lookup[candidate_id_str.upper()] = candidate

        effective_entries: list[dict] = [dict(entry) for entry in ranked_entries or []]
        if not effective_entries and candidates:
            for candidate in candidates[: self.ranking_size]:
                candidate_id = candidate.get("id")
                if candidate_id is None:
                    continue
                candidate_id_str = str(candidate_id).strip()
                if not candidate_id_str:
                    continue
                effective_entries.append(
                    {
                        "id": candidate_id_str,
                        "confidence_label": None,
                        "confidence_score": None,
                    }
                )

        fallback_score = nil_score if nil_score is not None else 0.0
        fallback_score = max(0.0, min(1.0, fallback_score))
        fallback_score = round(fallback_score, self.RANKING_SCORE_PRECISION)
        if fallback_score >= 0.70:
            fallback_label = "HIGH"
        elif fallback_score >= 0.40:
            fallback_label = "MEDIUM"
        else:
            fallback_label = "LOW"

        if predicted_identifier.upper() == "NIL":
            nil_entry_found = False
            for entry in effective_entries:
                entry_id = str(entry.get("id", "")).strip()
                if entry_id.upper() == "NIL":
                    entry.setdefault("confidence_label", fallback_label)
                    entry.setdefault("confidence_score", fallback_score)
                    nil_entry_found = True
                    break
            if not nil_entry_found:
                effective_entries.insert(
                    0,
                    {
                        "id": "NIL",
                        "confidence_label": fallback_label,
                        "confidence_score": fallback_score,
                    },
                )

        enriched_entries: list[dict] = []
        for entry in effective_entries:
            entry_id = str(entry.get("id", "")).strip()
            if not entry_id:
                continue

            score_value = entry.get("confidence_score")
            if isinstance(score_value, (int, float)):
                score = round(float(score_value), self.RANKING_SCORE_PRECISION)
            else:
                score = None
            label_value = entry.get("confidence_label")
            if isinstance(label_value, str) and label_value.strip():
                label = label_value.strip().upper()
            elif score is not None:
                if score >= 0.70:
                    label = "HIGH"
                elif score >= 0.40:
                    label = "MEDIUM"
                else:
                    label = "LOW"
            else:
                label = None

            base_info = candidate_lookup.get(entry_id) if entry_id.upper() != "NIL" else {}
            if not base_info and entry_id.upper() != "NIL":
                base_info = candidate_lookup.get(entry_id.upper(), {})

            raw_types = base_info.get("types", []) if isinstance(base_info, dict) else []
            types: list[dict[str, str]] = []
            for type_info in raw_types or []:
                if isinstance(type_info, dict):
                    type_id = type_info.get("id")
                    type_name = type_info.get("name")
                    if type_id or type_name:
                        types.append(
                            {
                                "id": str(type_id).strip() if type_id not in (None, "") else "",
                                "name": str(type_name).strip() if type_name not in (None, "") else "",
                            }
                        )
                elif isinstance(type_info, str):
                    types.append({"id": "", "name": type_info.strip()})

            description_value = ""
            if isinstance(base_info, dict):
                description_raw = base_info.get("description")
                if description_raw not in (None, ""):
                    description_value = str(description_raw)

            name_value = ""
            if isinstance(base_info, dict):
                candidate_name = base_info.get("name")
                if candidate_name not in (None, ""):
                    name_value = str(candidate_name)

            enriched_entry = {
                "id": entry_id,
                "confidence_label": label,
                "confidence_score": score,
                "name": name_value,
                "types": types,
                "description": description_value,
                "match": entry_id.upper() == predicted_identifier.upper(),
            }
            enriched_entries.append(enriched_entry)

        return enriched_entries

    def _enrich_output_csv(self) -> None:
        if not os.path.exists(self.output_csv):
            return

        df = pd.read_csv(self.output_csv)
        updated = False
        metadata_cols_to_drop = [col for col in df.columns if col.endswith("_candidate_metadata")]

        for column in self.mention_columns:
            answer_col = f"{column}_llm_answer"
            metadata_col = f"{column}_candidate_metadata"
            pred_col = f"{column}{self.prediction_suffix}"
            ranking_col = f"{column}_candidate_ranking"
            if answer_col not in df.columns:
                continue

            if ranking_col in df.columns and df[ranking_col].notna().any():
                continue

            enriched_answers: list[str] = []
            for _, row in df.iterrows():
                raw_answer = row.get(answer_col)
                raw_metadata = row.get(metadata_col)
                identifier = row.get(pred_col)

                ranking_entries: list[dict] = []
                explanation_value: str | None = None
                if isinstance(raw_answer, str) and raw_answer:
                    try:
                        payload = json.loads(raw_answer)
                        if isinstance(payload, dict):
                            ranking_entries = payload.get(self.RANKING_KEY, []) or []
                            explanation_raw = payload.get("explanation")
                            if isinstance(explanation_raw, str):
                                explanation_value = explanation_raw.strip()
                        elif isinstance(payload, list):
                            ranking_entries = payload
                    except (ValueError, TypeError):
                        ranking_entries = []
                if not explanation_value:
                    explanation_value = self.DEFAULT_EXPLANATION_FALLBACK

                candidate_metadata: list[dict] = []
                if isinstance(raw_metadata, str) and raw_metadata:
                    try:
                        metadata_payload = json.loads(raw_metadata)
                        if isinstance(metadata_payload, list):
                            candidate_metadata = metadata_payload
                    except (ValueError, TypeError):
                        candidate_metadata = []

                try:
                    final_rank_entries = LionLinker._validate_candidate_ranking(
                        ranking_entries,
                        self.ranking_size,
                    )
                except ValueError:
                    final_rank_entries = []

                predicted_identifier = str(identifier).strip() if isinstance(identifier, str) else ""
                enriched_ranking = self._enrich_candidate_ranking(
                    final_rank_entries,
                    candidate_metadata,
                    predicted_identifier,
                )
                enriched_payload = {
                    self.RANKING_KEY: enriched_ranking,
                    "explanation": explanation_value,
                }
                enriched_answers.append(json.dumps(enriched_payload, ensure_ascii=False))

            if enriched_answers:
                df[ranking_col] = enriched_answers
                updated = True

        if metadata_cols_to_drop:
            df = df.drop(columns=metadata_cols_to_drop)
            updated = True

        if updated:
            df.to_csv(
                self.output_csv,
                index=False,
                quoting=1,
                escapechar="\\",
            )

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
                    response = '{"candidate_ranking":[]}'  # fallback

                try:
                    ranking_entries, nil_score, explanation = self._parse_llm_json(response)
                except ValueError as parse_error:
                    logging.error(
                        "LLM response parsing failed for mention '%s': %s",
                        entity_mention,
                        parse_error,
                    )
                    ranking_entries, nil_score, explanation = self._default_nil_payload()

                predicted_identifier = self._determine_predicted_identifier(
                    ranking_entries,
                    nil_score,
                )
                answer_payload = {
                    self.RANKING_KEY: ranking_entries,
                    "explanation": explanation,
                }
                enriched_ranking = self._enrich_candidate_ranking(
                    ranking_entries,
                    candidates,
                    predicted_identifier,
                    nil_score,
                )
                api_answer_payload = {
                    self.RANKING_KEY: enriched_ranking,
                    "explanation": explanation,
                }
                candidate_ranking_json = json.dumps(api_answer_payload, ensure_ascii=False)
                answer_json = json.dumps(answer_payload, ensure_ascii=False)

                # Add column-specific results to the row
                results_by_row[id_row][f"{column}_llm_answer"] = answer_json
                results_by_row[id_row][f"{column}{self.prediction_suffix}"] = predicted_identifier
                results_by_row[id_row][f"{column}_candidate_ranking"] = candidate_ranking_json

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
            ranking_entries, nil_score, _ = self._parse_llm_json(response)
            return self._determine_predicted_identifier(ranking_entries, nil_score)
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
            dict: A mapping of mention column names to dictionaries containing:
                  - 'prompt': the generated prompt text
                  - 'candidates': the candidate list retrieved for that mention
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
            self._prompt_candidate_cache[col] = candidates
            prompts[col] = {"prompt": prompt, "candidates": candidates}

        return prompts

    async def send_prompt_sample(self, prompt_sample: dict) -> dict:
        """
        Sends each prompt in the provided prompt sample to the LLM and returns the responses.

        Args:
            prompt_sample (dict): A dictionary mapping mention column names to either:
                - a prompt string; or
                - a dictionary containing 'prompt' and 'candidates' keys as returned by
                  `generate_sample_prompt`.

        Returns:
            dict: A dictionary mapping each mention column to a dictionary with keys:
                  'prompt', 'response', 'extracted_identifier', 'candidate_ranking',
                  and 'candidates'.
        """
        results = {}
        for col, prompt_entry in prompt_sample.items():
            if isinstance(prompt_entry, dict):
                prompt = prompt_entry.get("prompt", "")
                candidates = prompt_entry.get("candidates")
            else:
                prompt = prompt_entry
                candidates = None

            if candidates is None:
                candidates = self._prompt_candidate_cache.get(col, [])
            if candidates is None:
                candidates = []

            response = self.llm_interaction.chat(prompt)
            try:
                ranking_entries, nil_score, explanation = self._parse_llm_json(response)
            except ValueError:
                logging.warning("Failed to parse LLM response for column '%s'", col)
                ranking_entries, nil_score, explanation = self._default_nil_payload()

            predicted_identifier = self._determine_predicted_identifier(
                ranking_entries,
                nil_score,
            )
            enriched_ranking = self._enrich_candidate_ranking(
                ranking_entries,
                candidates,
                predicted_identifier,
                nil_score,
            )
            answer_payload = {
                self.RANKING_KEY: ranking_entries,
                "explanation": explanation,
            }
            answer_json = json.dumps(answer_payload, ensure_ascii=False)
            results[col] = {
                "prompt": prompt,
                "response": answer_json,
                "extracted_identifier": predicted_identifier,
                "candidate_ranking": enriched_ranking,
                "candidates": candidates,
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
        self._enrich_output_csv()
