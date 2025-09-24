from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.core.config import Settings, settings
from app.models.dataset import DatasetTableRecord
from app.models.jobs import JobStatus, PredictionSummary, ResultRow
from app.storage.state import StateStore
from lion_linker.lion_linker import LionLinker
from lion_linker.retrievers import LamapiClient, RetrieverClient

logger = logging.getLogger(__name__)


class NullRetriever(RetrieverClient):
    """Retriever that returns no candidates, useful for dry runs or offline mode."""

    def __init__(self, num_candidates: int = 0):
        super().__init__(endpoint="", num_candidates=num_candidates)

    async def fetch_entities(self, mention: str, session, **kwargs):  # type: ignore[override]
        return []

    async def fetch_multiple_entities(self, mentions: List[str], **kwargs):  # type: ignore[override]
        return {mention: [] for mention in mentions}


@dataclass
class JobPaths:
    input_csv: Path
    output_csv: Path
    result_json: Path


class LinkerRunner:
    def __init__(self, store: StateStore, app_settings: Settings | None = None):
        self.store = store
        self.settings = app_settings or settings

    async def run_job(self, job_id: str) -> None:
        job = await self.store.get_job(job_id)
        table = await self.store.get_table(job.dataset_id, job.table_id)

        lion_config = self._merge_configs(table.lion_config, job.lion_config)
        retriever_config = self._merge_configs(table.retriever_config, job.retriever_config)

        mention_override = self._list_option(lion_config, ["mention_columns", "mentionColumns"])
        mention_columns = self._resolve_mention_columns(table, override=mention_override)
        if not mention_columns:
            raise ValueError("No mention columns available for table")

        dry_run = self._bool_option(lion_config, ["dry_run", "dryRun"], self.settings.dry_run)

        await self.store.update_job(
            job_id,
            message="Preparing data for LionLinker",
            total_rows=len(table.rows),
            updated_at=datetime.now(tz=timezone.utc),
        )

        paths = self._prepare_paths(job.job_id, table)
        df = StateStore.dataframe_from_payload(table)
        await asyncio.to_thread(df.to_csv, paths.input_csv, index=False)

        if dry_run:
            results = self._build_nil_results(table, mention_columns)
            await self._write_results(paths.result_json, results)
            await self.store.set_job_result(
                job_id,
                output_path=None,
                result_path=paths.result_json,
                total_rows=len(results),
                processed_rows=len(results),
            )
            await self.store.update_job(
                job_id,
                status=JobStatus.completed,
                message="Dry run completed",
                updated_at=datetime.now(tz=timezone.utc),
            )
            return

        retriever = self._build_retriever(table, retriever_config)

        chunk_size = self._int_option(
            lion_config, ["chunk_size", "chunkSize"], self.settings.chunk_size
        )
        if chunk_size is None or chunk_size < 1:
            chunk_size = 1

        table_ctx_size = self._int_option(
            lion_config, ["table_ctx_size", "tableCtxSize"], self.settings.table_ctx_size
        )
        if table_ctx_size is None or table_ctx_size < 0:
            table_ctx_size = 0

        model_name = self._get_option(
            lion_config, ["model_name", "modelName"], self.settings.model_name
        )
        model_api_provider = self._get_option(
            lion_config,
            ["model_api_provider", "modelApiProvider"],
            self.settings.model_api_provider,
        )
        model_api_key = self._get_option(
            lion_config, ["model_api_key", "modelApiKey"], self.settings.model_api_key
        )
        ollama_host = self._get_option(
            lion_config, ["ollama_host", "ollamaHost"], self.settings.ollama_host
        )

        format_candidates = self._bool_option(
            lion_config, ["format_candidates", "formatCandidates"], True
        )
        compact_candidates = self._bool_option(
            lion_config, ["compact_candidates", "compactCandidates"], True
        )

        prompt_template = self._get_option(
            lion_config, ["prompt_template", "promptTemplate"], None
        )
        few_shot_examples = self._get_option(
            lion_config,
            ["few_shot_examples_file_path", "fewShotExamplesFilePath"],
            None,
        )
        answer_format = self._get_option(lion_config, ["answer_format", "answerFormat"], None)
        gt_columns = self._list_option(lion_config, ["gt_columns", "gtColumns"], None)

        lion_kwargs: Dict[str, Any] = {
            "model_name": model_name,
            "chunk_size": chunk_size,
            "mention_columns": mention_columns,
            "model_api_provider": model_api_provider,
            "ollama_host": ollama_host,
            "model_api_key": model_api_key,
            "table_ctx_size": table_ctx_size,
            "format_candidates": format_candidates,
            "compact_candidates": compact_candidates,
        }
        if prompt_template:
            lion_kwargs["prompt_template"] = prompt_template
        if few_shot_examples:
            lion_kwargs["few_shot_examples_file_path"] = few_shot_examples
        if answer_format:
            lion_kwargs["answer_format"] = answer_format
        if gt_columns:
            lion_kwargs["gt_columns"] = gt_columns

        await self.store.update_job(
            job_id,
            message="Running LionLinker",
            updated_at=datetime.now(tz=timezone.utc),
        )

        lion = self._build_lion_linker(paths, table, retriever, lion_kwargs)

        await lion.run()

        results = self._parse_results(paths.output_csv, table, mention_columns)
        await self._write_results(paths.result_json, results)
        await self.store.set_job_result(
            job_id,
            output_path=paths.output_csv,
            result_path=paths.result_json,
            total_rows=len(results),
            processed_rows=len(results),
        )
        await self.store.update_job(
            job_id,
            status=JobStatus.completed,
            message="Linking completed",
            updated_at=datetime.now(tz=timezone.utc),
        )

    @staticmethod
    def _merge_configs(*configs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for config in configs:
            if config:
                merged.update(config)
        return merged

    @staticmethod
    def _get_option(config: Optional[Dict[str, Any]], keys: List[str], default: Any = None) -> Any:
        if not config:
            return default
        for key in keys:
            if key in config and config[key] is not None:
                return config[key]
        return default

    def _bool_option(
        self, config: Optional[Dict[str, Any]], keys: List[str], default: Optional[bool] = None
    ) -> Optional[bool]:
        value = self._get_option(config, keys, default)
        if value is None:
            return default
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
        return bool(value)

    def _int_option(
        self, config: Optional[Dict[str, Any]], keys: List[str], default: Optional[int] = None
    ) -> Optional[int]:
        value = self._get_option(config, keys, default)
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _float_option(
        self, config: Optional[Dict[str, Any]], keys: List[str], default: Optional[float] = None
    ) -> Optional[float]:
        value = self._get_option(config, keys, default)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _list_option(
        self,
        config: Optional[Dict[str, Any]],
        keys: List[str],
        default: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        value = self._get_option(config, keys, default)
        if value is None:
            return default
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
            return items or default
        if isinstance(value, (set, tuple)):
            value = list(value)
        if isinstance(value, list):
            items = [str(item) for item in value if item is not None]
            return items or default
        return default

    @staticmethod
    def _normalise_keys(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not config:
            return {}
        normalised: Dict[str, Any] = {}
        for key, value in config.items():
            normalised[key] = value
            snake = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
            snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", snake).lower()
            normalised.setdefault(snake, value)
        return normalised

    @staticmethod
    def _load_class(path: str):
        module_name, class_name = path.rsplit(".", 1)
        module = import_module(module_name)
        return getattr(module, class_name)

    def _build_retriever(
        self,
        table: DatasetTableRecord,
        config: Optional[Dict[str, Any]],
    ) -> RetrieverClient:
        normalised = self._normalise_keys(config)

        class_path = self._get_option(normalised, ["class_path"], None)
        if class_path:
            retriever_cls = self._load_class(str(class_path))
        else:
            retriever_cls = LamapiClient

        endpoint = self._get_option(
            normalised, ["endpoint", "retriever_endpoint", "url"], self.settings.retriever_endpoint
        )
        token = self._get_option(
            normalised, ["token", "retriever_token", "api_key"], self.settings.retriever_token
        )
        num_candidates = self._int_option(
            normalised, ["num_candidates"], self.settings.retriever_num_candidates
        )
        kg_value = self._get_option(normalised, ["kg", "knowledge_graph"], table.kg_reference)
        cache = self._bool_option(normalised, ["cache"], self.settings.retriever_cache)
        max_retries = self._int_option(normalised, ["max_retries"], None)
        backoff_factor = self._float_option(normalised, ["backoff_factor"], None)

        recognised = {
            "class_path",
            "classPath",
            "endpoint",
            "retriever_endpoint",
            "retrieverEndpoint",
            "url",
            "endpoint_url",
            "endpointUrl",
            "token",
            "retriever_token",
            "api_key",
            "apiKey",
            "num_candidates",
            "numCandidates",
            "kg",
            "knowledge_graph",
            "knowledgeGraph",
            "cache",
            "max_retries",
            "maxRetries",
            "backoff_factor",
            "backoffFactor",
        }
        extra_kwargs = {key: value for key, value in normalised.items() if key not in recognised}

        if retriever_cls is LamapiClient and not endpoint:
            return NullRetriever(num_candidates=num_candidates or 0)

        kwargs: Dict[str, Any] = dict(extra_kwargs)
        if endpoint is not None:
            kwargs.setdefault("endpoint", endpoint)
        if token is not None:
            kwargs.setdefault("token", token)
        if num_candidates is not None:
            kwargs.setdefault("num_candidates", num_candidates)
        if kg_value is not None:
            kwargs.setdefault("kg", kg_value)
        if cache is not None:
            kwargs.setdefault("cache", cache)
        if max_retries is not None:
            kwargs.setdefault("max_retries", max_retries)
        if backoff_factor is not None:
            kwargs.setdefault("backoff_factor", backoff_factor)

        try:
            return retriever_cls(**kwargs)
        except TypeError as exc:  # pragma: no cover - dependency errors surface here
            raise ValueError(f"Unable to initialise retriever {retriever_cls}: {exc}") from exc

    def _build_lion_linker(
        self,
        paths: JobPaths,
        table: DatasetTableRecord,
        retriever: RetrieverClient,
        lion_kwargs: Dict[str, Any],
    ) -> LionLinker:
        kwargs = {
            "input_csv": str(paths.input_csv),
            "output_csv": str(paths.output_csv),
            "retriever": retriever,
            "kg_name": table.kg_reference,
        }
        kwargs.update(lion_kwargs)
        print("Building Lion-Linker with kwargs:")
        print(kwargs)
        return LionLinker(**kwargs)

    def _prepare_paths(self, job_id: str, table: DatasetTableRecord) -> JobPaths:
        safe_dataset = self._slugify(table.dataset_name)
        safe_table = self._slugify(table.table_name)
        job_dir = self.settings.workspace_path / safe_dataset / safe_table / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return JobPaths(
            input_csv=job_dir / "input.csv",
            output_csv=job_dir / "output.csv",
            result_json=job_dir / "result.json",
        )

    @staticmethod
    def _slugify(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "table"

    def _resolve_mention_columns(
        self, table: DatasetTableRecord, override: Optional[List[str]] = None
    ) -> List[str]:
        if override:
            columns = [col for col in override if col in table.header]
            if columns:
                return columns
        metadata = table.metadata or {}
        mention_columns = metadata.get("mentionColumns") or metadata.get("mention_columns")
        if isinstance(mention_columns, str):
            mention_columns = [mention_columns]
        if mention_columns:
            columns = [col for col in mention_columns if col in table.header]
            if columns:
                return columns

        if self.settings.default_mention_columns:
            columns = [col for col in self.settings.default_mention_columns if col in table.header]
            if columns:
                return columns

        return [table.header[0]] if table.header else []

    @staticmethod
    def _build_nil_results(
        table: DatasetTableRecord, mention_columns: List[str]
    ) -> List[ResultRow]:
        results: List[ResultRow] = []
        for row in table.rows:
            predictions = [
                PredictionSummary(column=column, answer="ANSWER:NIL", identifier="NIL")
                for column in mention_columns
            ]
            data = [str(value) if value is not None else "" for value in row.data]
            results.append(ResultRow(idRow=row.id_row, data=data, predictions=predictions))
        return results

    async def _write_results(self, path: Path, rows: List[ResultRow]) -> None:
        serialisable = [row.model_dump(mode="json") for row in rows]

        def _write() -> None:
            path.write_text(
                json.dumps(serialisable, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        await asyncio.to_thread(_write)

    def _parse_results(
        self,
        output_csv: Path,
        table: DatasetTableRecord,
        mention_columns: List[str],
    ) -> List[ResultRow]:
        df = pd.read_csv(output_csv)
        if len(df) != len(table.rows):
            logger.warning(
                "Output row count %s does not match original rows %s",
                len(df),
                len(table.rows),
            )
        results: List[ResultRow] = []
        header = table.header
        for row_payload, (_, row_series) in zip(table.rows, df.iterrows()):
            data: List[str] = []
            for column in header:
                value = row_series.get(column)
                if pd.isna(value):
                    data.append("")
                else:
                    data.append(str(value))

            predictions: List[PredictionSummary] = []
            for column in mention_columns:
                answer = row_series.get(f"{column}_llm_answer")
                identifier = row_series.get(f"{column}_pred_id")
                answer_str = answer.strip() if isinstance(answer, str) else None
                identifier_str = identifier.strip() if isinstance(identifier, str) else None
                if answer_str is None and not pd.isna(answer):
                    answer_str = str(answer)
                if identifier_str is None and not pd.isna(identifier):
                    identifier_str = str(identifier)
                predictions.append(
                    PredictionSummary(
                        column=column,
                        answer=answer_str,
                        identifier=identifier_str,
                    )
                )

            results.append(ResultRow(idRow=row_payload.id_row, data=data, predictions=predictions))
        return results
