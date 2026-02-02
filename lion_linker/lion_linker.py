from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import pandas as pd
from tqdm.asyncio import tqdm

from lion_linker.core import LLMInteraction
from lion_linker.prompt.generator import PromptGenerator
from lion_linker.retrievers import RetrieverClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


if TYPE_CHECKING:
    from lion_linker.prompt_store import PromptStore
    from lion_linker.result_store import ResultStore


class LionLinker:
    DEFAULT_ANSWER_FORMAT_TEMPLATE = """
        ANSWER FORMAT (IMPORTANT):
        Return ONLY a JSON object with exactly these top-level keys (no extra text):
        {{
            "{ranking_key}": [
                {{
                    "id": "<CANDIDATE_ID>",
                    "score": <float|null>
                }}
            ],
            "explanation": "<short reasoning of the final decision>"
        }}

        Rules:
        1. Always return the JSON object with both fields "{ranking_key}" and "explanation".
        2. Return exactly {ranking_size} candidates (or all candidates if fewer are provided).
        3. Each candidate must have the keys "id" and "score".
        4. "score" must be in [0, 1] for normal cases.
        5. If none of the provided candidates fits (NIL case):
           - Return the original candidates (up to the first {ranking_size}) in the same order
             as presented in the prompt, each with "score": null.
           - Use "explanation" to state why NIL was selected.
        6. Do not invent candidates. Score only the candidates that were provided in the prompt.
        7. The final output must be valid JSON (no Markdown, no trailing commas).
        8. Very important: the value of "explanation" must not contain the double quote character (").
           - If you need to mention text that originally contains double quotes, rewrite them as single
             quotes.
           - Never output the character " inside the explanation string.
    """.strip()

    ALLOWED_RANKING_SIZES = (3, 5)
    RANKING_KEY = "candidate_ranking"
    RANKING_SCORE_PRECISION = 4
    MAX_TASKS_PER_PROMPT = 50
 
    def __init__(
        self,
        input_csv: str | Path | pd.DataFrame,
        model_name: str,
        retriever: RetrieverClient,
        output_csv: str | None = None,
        prompt_template: str = "base",
        chunk_size: int = 16,
        mention_columns: list | None = None,
        format_candidates: bool = False,
        compact_candidates: bool = False,
        model_api_provider: str = "ollama",
        ollama_host: str | None = None,
        model_api_key: str | None = None,
        few_shot_examples_file_path: str | None = None,
        gt_columns: list | None = None,
        answer_format: str | None = None,
        ranking_size: int = 5,
        max_tasks_per_prompt: int = 50,
        max_parallel_prompts: int = 1,
        max_candidates_per_task: int | None = None,
        match_confidence_threshold: float = 0.5,
        nil_insert_delta: float = 0.05,
        prompt_callback: Optional[Callable[[dict], None]] = None,
        prompt_store: "PromptStore | None" = None,
        mongo_prompts: bool | None = None,
        result_store: "ResultStore | None" = None,
        mongo_results: bool | None = None,
        **kwargs,
    ):
        """Initialize a LionLinker instance.

        Parameters:
            input_csv (str | Path | pandas.DataFrame): The file path to the input CSV file,
                or a pandas DataFrame to be processed.
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
                TableLlama (https://arxiv.org/abs/2311.09206) or the legacy bracket format.
                Defaults to False (JSON candidates).
            compact_candidates (bool, optional): Whether to use a compact candidate list
                (ID | TYPE | DESCRIPTION) when `format_candidates` is False.
                Defaults to False (JSON candidates).
            model_api_provider (str, optional): The provider for the model API.
                Supported providers: "ollama", "openrouter", "huggingface", "cerebras".
                Defaults to "ollama".
            ollama_host (str, optional): The host for the Ollama service.
                Defaults to None.
            model_api_key (str, optional): The API key for the model service.
                Required for "openrouter" and "cerebras" providers when no env variable is found.
                Defaults to None.
            few_shot_examples_file_path (str, optional): The file path to
                the few shot examples file.
                Defaults to None.
            gt_columns (list, optional): List of ground truth columns for reference.
                Defaults to None.
            answer_format (str, optional): The format for the answer.
                Defaults to None.
            ranking_size (int, optional): Maximum number of candidates to request from the LLM.
                Must be either 3 or 5. Defaults to 5.
            max_tasks_per_prompt (int, optional): Maximum number of tasks per LLM prompt.
                Must be between 1 and 50. Defaults to 50.
            max_parallel_prompts (int, optional): Maximum number of prompt batches to send
                concurrently. Defaults to 1.
            max_candidates_per_task (int, optional): Maximum number of candidates to include
                per task in the prompt. Defaults to the retriever's num_candidates.
            match_confidence_threshold (float, optional): Minimum probability the top-ranked
                candidate must achieve (combined with a HIGH confidence label derived from the score)
                to be marked as
                the final match. Must fall within (0, 1]. Defaults to 0.5.
            nil_insert_delta (float, optional): Deprecated; retained for backwards compatibility.
                It no longer affects the ranking behaviour.
            prompt_callback (callable, optional): Callback invoked with prompt payloads
                for debugging (prompt text, task ids, metadata).
            prompt_store (PromptStore, optional): Mongo-backed prompt logger. If provided,
                prompts are stored for troubleshooting.
            mongo_prompts (bool, optional): Enable Mongo-backed prompt logging via env vars
                (same mechanism as the candidate cache).
            result_store (ResultStore, optional): Mongo-backed result logger. If provided,
                final outputs are stored for troubleshooting.
            mongo_results (bool, optional): Enable Mongo-backed result logging via env vars
                (same mechanism as the candidate cache).
            **kwargs: Additional keyword arguments.
                      Supported hidden features:
                      - mention_to_qids: Dict mapping mentions to entity IDs to force in candidates
                      - id_extraction_pattern: Regex pattern for extracting IDs from responses
                      - prediction_suffix: Suffix for prediction column names
                      - kg_name: Name of the knowledge graph being used
                      - target_rows_ids: an iterable specifying a subset of rows to be processed
        """

        self._temp_input_path: Path | None = None
        if isinstance(input_csv, pd.DataFrame):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            temp_file.close()
            input_csv.to_csv(temp_file.name, index=False)
            self._temp_input_path = Path(temp_file.name)
            input_csv = temp_file.name
        if not isinstance(input_csv, (str, Path)):
            raise TypeError("input_csv must be a file path or a pandas DataFrame.")
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
        if self.chunk_size < 1:
            raise ValueError(f"Chuck size must be at least 1. Got batch size: {self.chunk_size}")

        if ranking_size not in LionLinker.ALLOWED_RANKING_SIZES:
            raise ValueError(
                f"ranking_size must be one of {LionLinker.ALLOWED_RANKING_SIZES}. "
                f"Got ranking_size: {ranking_size}"
            )
        self.ranking_size = ranking_size

        if max_tasks_per_prompt < 1 or max_tasks_per_prompt > self.MAX_TASKS_PER_PROMPT:
            raise ValueError(
                "max_tasks_per_prompt must be within [1, 50]. "
                f"Got: {max_tasks_per_prompt}"
            )
        self.max_tasks_per_prompt = max_tasks_per_prompt
        if max_parallel_prompts < 1:
            raise ValueError(
                "max_parallel_prompts must be at least 1. "
                f"Got: {max_parallel_prompts}"
            )
        self.max_parallel_prompts = max_parallel_prompts

        if max_candidates_per_task is None:
            max_candidates_per_task = getattr(self.retriever, "num_candidates", None)
        if max_candidates_per_task is not None and max_candidates_per_task < 1:
            raise ValueError(
                "max_candidates_per_task must be >= 1 when provided. "
                f"Got: {max_candidates_per_task}"
            )
        self.max_candidates_per_task = max_candidates_per_task

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
        self._prompt_callback = prompt_callback
        self._prompt_seq = 0
        self._prompt_tasks: list[asyncio.Task] = []
        prompt_store_obj = prompt_store
        if prompt_store_obj is None and mongo_prompts:
            try:
                from lion_linker.prompt_store import PromptStore as _PromptStore
            except Exception as exc:
                logging.info("Prompt store disabled (Mongo client unavailable): %s", exc)
                prompt_store_obj = None
            else:
                prompt_store_obj = _PromptStore.from_env(
                    enabled=mongo_prompts,
                    metadata={
                        "input_csv": str(self.input_csv),
                        "model_name": self.model_name,
                        "model_api_provider": self.model_api_provider,
                    },
                )
        self._prompt_store = prompt_store_obj
        self.prompt_run_id = getattr(prompt_store_obj, "run_id", None)

        result_store_obj = result_store
        if result_store_obj is None and mongo_results is None and prompt_store_obj is not None:
            mongo_results = True
        if result_store_obj is None and mongo_results:
            try:
                from lion_linker.result_store import ResultStore as _ResultStore
            except Exception as exc:
                logging.info("Result store disabled (Mongo client unavailable): %s", exc)
                result_store_obj = None
            else:
                result_store_obj = _ResultStore.from_env(
                    enabled=mongo_results,
                    run_id=self.prompt_run_id,
                    metadata={
                        "input_csv": str(self.input_csv),
                        "model_name": self.model_name,
                        "model_api_provider": self.model_api_provider,
                    },
                )
        self._result_store = result_store_obj
        self._result_seq = 0

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
            self.format_candidates = True
            self.prompt_template = "tablellama"
            pattern_str = kwargs.get("id_extraction_pattern", r'"id"\s*:\s*"([^"]+)"')
        else:
            self.answer_format = default_answer_format
            pattern_str = kwargs.get("id_extraction_pattern", r'"id"\s*:\s*"([^"]+)"')
        self.table_summary = None

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

    def _emit_prompt(self, payload: dict) -> None:
        if not payload:
            return
        prompt_payload = dict(payload)
        prompt_payload.setdefault("seq", self._prompt_seq)
        self._prompt_seq += 1

        if self._prompt_store is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                try:
                    asyncio.run(self._prompt_store.record(prompt_payload))
                except Exception as exc:
                    logging.warning("Prompt store write failed: %s", exc)
            else:
                task = loop.create_task(self._prompt_store.record(prompt_payload))
                self._prompt_tasks.append(task)
                if len(self._prompt_tasks) > 1000:
                    self._prompt_tasks = [item for item in self._prompt_tasks if not item.done()]

        if not self._prompt_callback:
            return
        try:
            self._prompt_callback(prompt_payload)
        except Exception as exc:
            logging.warning("Prompt callback failed: %s", exc)

    async def _flush_prompt_tasks(self) -> None:
        if not self._prompt_tasks:
            return
        tasks = [task for task in self._prompt_tasks if not task.done()]
        if not tasks:
            self._prompt_tasks = []
            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logging.debug("Prompt store task failed: %s", result)
        self._prompt_tasks = []

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

        nil_entry_seen = False

        normalized_with_order: list[dict] = []
        numeric_scores = 0
        null_scores = 0
        for order_idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError("Each candidate ranking entry must be a JSON object.")

            raw_id = entry.get("id")
            if not isinstance(raw_id, str):
                raise ValueError("Candidate ranking entries must include a string 'id'.")
            candidate_id = raw_id.strip()
            if not candidate_id:
                raise ValueError("Candidate ranking entries must include a non-empty string 'id'.")

            score_key = None
            if "score" in entry:
                score_key = "score"
            elif "confidence_score" in entry:
                score_key = "confidence_score"
            if score_key is None:
                raise ValueError("Candidate ranking entries must include 'score'.")
            score_value = entry.get(score_key)
            if candidate_id.upper() == "NIL":
                nil_entry_seen = True
                continue
            score: float | None
            if score_value is None:
                null_scores += 1
                score = None
            else:
                if not isinstance(score_value, (int, float)):
                    raise ValueError(
                        "score must be numeric. "
                        f"Received type {type(score_value)} for id {candidate_id}."
                    )
                score = float(score_value)
                if not 0 <= score <= 1:
                    raise ValueError(
                        "score must be within [0, 1]. "
                        f"Received {score} for id {candidate_id}."
                    )
                numeric_scores += 1

            normalized_with_order.append(
                {
                    "_order": order_idx,
                    "id": candidate_id,
                    "score": score,
                }
            )

        if not normalized_with_order:
            return []

        nil_mode = False
        if nil_entry_seen:
            nil_mode = True
        elif null_scores:
            if numeric_scores:
                raise ValueError("score must be all null when using NIL mode.")
            nil_mode = True

        if nil_mode:
            for item in normalized_with_order:
                item["score"] = None

        trimmed: list[dict] = []
        seen_ids: set[str] = set()

        iterable = normalized_with_order if nil_mode else sorted(
            normalized_with_order,
            key=lambda item: (-item["score"], item["_order"]),
        )

        for item in iterable:
            candidate_id_upper = item["id"].upper()
            if candidate_id_upper in seen_ids:
                continue
            seen_ids.add(candidate_id_upper)
            trimmed.append(
                {
                    "id": item["id"],
                    "score": item["score"],
                }
            )
            if len(trimmed) >= requested_top_k:
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

    def _parse_llm_task_payload(self, payload: dict) -> tuple[list[dict], float | None, str]:
        if not isinstance(payload, dict):
            raise ValueError("LLM task payload must be a JSON object.")
        if self.RANKING_KEY not in payload:
            raise ValueError(
                f'LLM task payload must contain a "{self.RANKING_KEY}" list.'
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
                'LLM task payload must contain an "explanation" string summarizing the decision.'
            )
        explanation_raw = payload["explanation"]
        if not isinstance(explanation_raw, str):
            raise ValueError('"explanation" must be a string.')
        explanation = explanation_raw.strip() or self.DEFAULT_EXPLANATION_FALLBACK

        return normalized_entries, nil_score, explanation

    def _parse_llm_json_array(
        self, response: str, expected_task_ids: list[str]
    ) -> dict[str, tuple[list[dict], float | None, str]]:
        if not response or not isinstance(response, str):
            raise ValueError("LLM response must be a non-empty string containing JSON.")

        response = response.strip()
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM response must be valid JSON. Received: {response}") from exc

        if isinstance(payload, dict):
            if len(expected_task_ids) != 1:
                raise ValueError("LLM response JSON must be an array of task objects.")
            ranking_entries, nil_score, explanation = self._parse_llm_task_payload(payload)
            return {expected_task_ids[0]: (ranking_entries, nil_score, explanation)}

        if not isinstance(payload, list):
            raise ValueError("LLM response JSON must be an array of task objects.")

        parsed: dict[str, tuple[list[dict], float | None, str]] = {}
        ordered_entries: list[dict] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            task_id = entry.get("taskId")
            if not isinstance(task_id, str) or not task_id.strip():
                ordered_entries.append(entry)
                continue
            if task_id in parsed:
                continue
            try:
                ranking_entries, nil_score, explanation = self._parse_llm_task_payload(entry)
            except ValueError:
                continue
            parsed[task_id] = (ranking_entries, nil_score, explanation)

        if not parsed and ordered_entries and len(ordered_entries) == len(expected_task_ids):
            for task_id, entry in zip(expected_task_ids, ordered_entries):
                try:
                    ranking_entries, nil_score, explanation = self._parse_llm_task_payload(entry)
                except ValueError:
                    parsed[task_id] = self._default_nil_payload()
                else:
                    parsed[task_id] = (ranking_entries, nil_score, explanation)

        for task_id in expected_task_ids:
            if task_id not in parsed:
                parsed[task_id] = self._default_nil_payload()

        return parsed

    def _build_task_id(self, base_filename: str, id_row: int, id_col: int) -> str:
        return f"{base_filename}-{id_row}-{id_col}"

    def _limit_candidates(self, candidates: list[dict], entity_mention: str) -> list[dict]:
        if self.max_candidates_per_task is None:
            return candidates
        if len(candidates) <= self.max_candidates_per_task:
            return candidates

        forced_ids_raw = self._mention_to_qids.get(entity_mention)
        forced_ids: list[str] = []
        if isinstance(forced_ids_raw, (list, tuple, set)):
            forced_ids = [str(fid) for fid in forced_ids_raw]
        elif forced_ids_raw:
            forced_ids = [str(forced_ids_raw)]

        forced_set = {fid.strip() for fid in forced_ids if fid}
        if not forced_set:
            return candidates[: self.max_candidates_per_task]

        forced_candidates = []
        remaining = []
        for candidate in candidates:
            candidate_id = str(candidate.get("id", "")).strip()
            if candidate_id and candidate_id in forced_set:
                forced_candidates.append(candidate)
            else:
                remaining.append(candidate)

        trimmed = forced_candidates + remaining
        return trimmed[: self.max_candidates_per_task]

    def _build_multitask_answer_format(self, num_tasks: int) -> str:
        lines: list[str] = []
        lines.append("ANSWER FORMAT (IMPORTANT):")
        lines.append(
            f"Return ONLY a JSON array with exactly {num_tasks} objects, one per task, in the same order as the tasks."
        )
        lines.append("")
        lines.append("Each object must have:")
        lines.append('  - "taskId": the TASK_ID string for that task.')
        lines.append(
            f'  - "{self.RANKING_KEY}": a list of exactly {self.ranking_size} objects '
            '(or all candidates if fewer are provided) { "id", "score" }.'
        )
        lines.append(
            '  - "explanation": short reasoning text, without double quotes (use single quotes if needed).'
        )
        lines.append("")
        lines.append("Confidence scores:")
        lines.append("  - score in [0, 1].")
        lines.append("")
        lines.append("If none of the candidates fits:")
        lines.append(
            f"  - Return the original candidates (up to the first {self.ranking_size}) in the same order "
            "as presented in the prompt, each with"
        )
        lines.append('    "score": null.')
        lines.append("  - Use explanation to state why NIL was selected.")
        lines.append("")
        lines.append("Do not invent candidates; score only the candidates provided in the prompt.")
        lines.append("The final output must be valid JSON (no trailing commas).")
        lines.append('Very important: the value of "explanation" must not contain the double quote character (").')
        return "\n".join(lines)

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

        score_value = top_entry.get("score")
        score = float(score_value) if isinstance(score_value, (int, float)) else 0.0
        if isinstance(score_value, (int, float)):
            if score >= 0.70:
                label = "HIGH"
            elif score >= 0.40:
                label = "MEDIUM"
            else:
                label = "LOW"
        else:
            label = None

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
                        "score": None,
                    }
                )

        if predicted_identifier.upper() == "NIL":
            for entry in effective_entries:
                entry["score"] = None

        seen_existing: set[str] = {
            str(entry.get("id", "")).strip().upper()
            for entry in effective_entries
            if isinstance(entry, dict)
        }
        if len(effective_entries) < self.ranking_size and candidates:
            for candidate in candidates:
                candidate_id = str(candidate.get("id", "")).strip()
                if not candidate_id:
                    continue
                candidate_id_upper = candidate_id.upper()
                if candidate_id_upper in seen_existing:
                    continue
                effective_entries.append(
                    {
                        "id": candidate_id,
                        "score": None,
                    }
                )
                seen_existing.add(candidate_id_upper)
                if len(effective_entries) >= self.ranking_size:
                    break

        enriched_entries: list[dict] = []
        for entry in effective_entries:
            entry_id = str(entry.get("id", "")).strip()
            if not entry_id:
                continue

            score_value = entry.get("score")
            if isinstance(score_value, (int, float)):
                score = float(score_value)
            else:
                score = None

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
                "score": score,
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

        payload = {"prompt": prompt, "type": "table_summary"}
        try:
            response = self.llm_interaction.chat(prompt)
        except Exception as exc:
            payload["error"] = str(exc)
            self._emit_prompt(payload)
            raise
        payload["response"] = response
        self._emit_prompt(payload)
        return response

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
        results_by_row: dict[int, dict] = {}
        tasks: list[dict[str, object]] = []

        base_filename = os.path.splitext(os.path.basename(self.input_csv))[0]
        target_ids = self._target_row_ids
        for id_row, row in chunk.iterrows():
            if target_ids is not None and id_row not in target_ids:
                continue
            table_list = [chunk.columns.tolist(), row.tolist()]

            if id_row not in results_by_row:
                results_by_row[id_row] = {"id_row": id_row}

            for column in self.mention_columns:
                entity_mention = row[column]
                candidates = mentions_to_candidates.get(entity_mention, [])
                candidates = self._limit_candidates(candidates, entity_mention)
                id_col = column_to_index[column]
                task_id = self._build_task_id(base_filename, id_row, id_col)
                tasks.append(
                    {
                        "task_id": task_id,
                        "row_id": id_row,
                        "column": column,
                        "table": table_list,
                        "column_name": column,
                        "entity_mention": entity_mention,
                        "candidates": candidates,
                    }
                )

        batches: list[tuple[int, list[dict[str, object]]]] = []
        for idx in range(0, len(tasks), self.max_tasks_per_prompt):
            task_batch = tasks[idx : idx + self.max_tasks_per_prompt]
            if not task_batch:
                continue
            batches.append((idx // self.max_tasks_per_prompt, task_batch))

        semaphore = asyncio.Semaphore(self.max_parallel_prompts)

        async def run_batch(batch_index: int, task_batch: list[dict[str, object]]):
            async with semaphore:
                answer_format = self._build_multitask_answer_format(len(task_batch))
                prompt = self.prompt_generator.generate_multi_prompt(
                    task_batch,
                    answer_format=answer_format,
                    compact=self.compact_candidates,
                    format_candidates=self.format_candidates,
                )
                expected_task_ids = [task["task_id"] for task in task_batch]

                prompt_payload = {
                    "prompt": prompt,
                    "task_ids": expected_task_ids,
                    "batch_index": batch_index,
                    "task_count": len(task_batch),
                    "type": "batch",
                }
                response = None
                try:
                    response = await asyncio.to_thread(self.llm_interaction.chat, prompt)
                    parsed = self._parse_llm_json_array(response, expected_task_ids)
                except Exception as exc:
                    logging.error("LLM interaction failed for multi-task prompt: %s", exc)
                    prompt_payload["response"] = response
                    prompt_payload["error"] = str(exc)
                    self._emit_prompt(prompt_payload)
                    parsed = {
                        task_id: self._default_nil_payload() for task_id in expected_task_ids
                    }
                else:
                    prompt_payload["response"] = response
                    self._emit_prompt(prompt_payload)

                return task_batch, parsed

        batch_results = await asyncio.gather(
            *(run_batch(batch_index, task_batch) for batch_index, task_batch in batches)
        )

        for task_batch, parsed in batch_results:
            for task in task_batch:
                task_id = str(task["task_id"])
                id_row = int(task["row_id"])
                column = str(task["column"])
                candidates = task["candidates"]
                ranking_entries, nil_score, explanation = parsed.get(
                    task_id, self._default_nil_payload()
                )

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

                results_by_row[id_row][f"{column}_llm_answer"] = answer_json
                results_by_row[id_row][f"{column}{self.prediction_suffix}"] = predicted_identifier
                results_by_row[id_row][f"{column}_candidate_ranking"] = candidate_ranking_json

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

        # Create table context for the selected row using header + row only.
        table_list = [chunk.columns.tolist(), sample_row.tolist()]

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
            candidates = self._limit_candidates(candidates, entity_mention)
            prompt = self.prompt_generator.generate_prompt(
                table=table_list,
                table_metadata=None,
                table_summary=None,
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
            self._emit_prompt(
                {
                    "prompt": prompt,
                    "response": response,
                    "type": "sample",
                    "column": col,
                }
            )
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

                if self._result_store is not None:
                    try:
                        cleaned = merged_chunk.where(pd.notnull(merged_chunk), None)
                        records = cleaned.to_dict(orient="records")
                        await self._result_store.record_batch(
                            records,
                            start_seq=self._result_seq,
                        )
                        self._result_seq += len(records)
                    except Exception as exc:
                        logging.warning("Result store failed: %s", exc)

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
        await self._flush_prompt_tasks()
