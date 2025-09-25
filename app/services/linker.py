from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, cast

import pandas as pd

from app.core import linker_defaults
from app.core.config import Settings, settings
from app.models.dataset import DatasetTableRecord
from app.models.jobs import JobRecord, JobStatus, PredictionSummary, ResultRow
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
    DEFAULT_MODEL_NAME = linker_defaults.DEFAULT_MODEL_NAME
    DEFAULT_MODEL_PROVIDER = linker_defaults.DEFAULT_MODEL_PROVIDER
    DEFAULT_CHUNK_SIZE = linker_defaults.DEFAULT_CHUNK_SIZE
    DEFAULT_TABLE_CTX_SIZE = linker_defaults.DEFAULT_TABLE_CTX_SIZE
    DEFAULT_OLLAMA_HOST: Optional[str] = linker_defaults.DEFAULT_OLLAMA_HOST
    DEFAULT_FORMAT_CANDIDATES = linker_defaults.DEFAULT_FORMAT_CANDIDATES
    DEFAULT_COMPACT_CANDIDATES = linker_defaults.DEFAULT_COMPACT_CANDIDATES

    def __init__(self, store: StateStore, app_settings: Settings | None = None):
        self.store = store
        self.settings = app_settings or settings
        self._provider_limits: Dict[str, asyncio.Semaphore] = {}
        self._fallback_provider_limit: Optional[asyncio.Semaphore] = None
        if self.settings.queue_workers > 1:
            self._provider_limits["openrouter"] = asyncio.Semaphore(self.settings.queue_workers)
            self._fallback_provider_limit = asyncio.Semaphore(1)

    async def run_job(self, job_id: str) -> None:
        job = await self.store.get_job(job_id)
        table = await self.store.get_table(job.dataset_id, job.table_id)

        lion_config = self._merge_configs(table.lion_config, job.lion_config)
        retriever_config = self._merge_configs(table.retriever_config, job.retriever_config)

        mention_override = cast(
            Optional[List[str]],
            self._list_option(lion_config, ["mention_columns"]),
        )
        mention_columns = self._resolve_mention_columns(table, override=mention_override)
        if not mention_columns:
            raise ValueError("No mention columns available for table")

        dry_run = self._bool_option(lion_config, ["dry_run", "dryRun"], self.settings.dry_run)

        await self.store.update_job(
            job_id,
            message="Preparing data for LionLinker",
            total_rows=len(job.row_ids) if job.row_ids else len(table.rows),
            updated_at=datetime.now(tz=timezone.utc),
        )

        paths = self._prepare_paths(job.job_id, table)
        df = StateStore.dataframe_from_payload(table)
        df_for_lion = df.drop(columns=["id_row"], errors="ignore")
        await asyncio.to_thread(df_for_lion.to_csv, paths.input_csv, index=False)

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

        retriever, effective_retriever_config = self._build_retriever(table, retriever_config)

        chunk_size = self._int_option(lion_config, ["chunk_size"], self.DEFAULT_CHUNK_SIZE)
        if chunk_size is None or chunk_size < 1:
            chunk_size = self.DEFAULT_CHUNK_SIZE

        table_ctx_size = self._int_option(
            lion_config, ["table_ctx_size"], self.DEFAULT_TABLE_CTX_SIZE
        )
        if table_ctx_size is None or table_ctx_size < 0:
            table_ctx_size = self.DEFAULT_TABLE_CTX_SIZE

        lion_kwargs: Dict[str, Any] = {}

        model_name = self._get_option(lion_config, ["model_name"], self.DEFAULT_MODEL_NAME)
        lion_kwargs["model_name"] = model_name or self.DEFAULT_MODEL_NAME

        lion_kwargs["chunk_size"] = chunk_size
        lion_kwargs["table_ctx_size"] = table_ctx_size
        lion_kwargs["mention_columns"] = mention_columns

        lion_kwargs["model_api_provider"] = self._get_option(
            lion_config,
            ["model_api_provider"],
            self.DEFAULT_MODEL_PROVIDER,
        )
        lion_kwargs["ollama_host"] = self._get_option(
            lion_config,
            ["ollama_host"],
            self.DEFAULT_OLLAMA_HOST,
        )
        lion_kwargs["model_api_key"] = self._get_option(
            lion_config,
            ["model_api_key"],
            None,
        )

        lion_kwargs["format_candidates"] = self._bool_option(
            lion_config,
            ["format_candidates"],
            self.DEFAULT_FORMAT_CANDIDATES,
        )
        lion_kwargs["compact_candidates"] = self._bool_option(
            lion_config,
            ["compact_candidates"],
            self.DEFAULT_COMPACT_CANDIDATES,
        )

        gt_columns = self._list_option(lion_config, ["gt_columns"], None)
        if gt_columns is not None:
            lion_kwargs["gt_columns"] = gt_columns

        if job.row_ids:
            lion_kwargs["target_row_ids"] = job.row_ids

        provider = lion_kwargs.get("model_api_provider") or self.DEFAULT_MODEL_PROVIDER
        limiter = self._provider_limits.get(provider) or self._fallback_provider_limit

        if limiter:
            async with limiter:
                await self._execute_job(
                    job,
                    table,
                    paths,
                    mention_columns,
                    lion_config,
                    lion_kwargs,
                    retriever,
                    effective_retriever_config,
                )
        else:
            await self._execute_job(
                job,
                table,
                paths,
                mention_columns,
                lion_config,
                lion_kwargs,
                retriever,
                effective_retriever_config,
            )

    @staticmethod
    def _run_lion_linker_blocking(lion: LionLinker) -> None:
        asyncio.run(lion.run())

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
    def _load_class(path: str):
        module_name, class_name = path.rsplit(".", 1)
        module = import_module(module_name)
        return getattr(module, class_name)

    async def _execute_job(
        self,
        job: JobRecord,
        table: DatasetTableRecord,
        paths: JobPaths,
        mention_columns: List[str],
        lion_config: Dict[str, Any],
        lion_kwargs: Dict[str, Any],
        retriever: RetrieverClient,
        effective_retriever_config: Dict[str, Any],
    ) -> None:
        job_id = job.job_id

        await self.store.update_job(
            job_id,
            message="Running LionLinker",
            updated_at=datetime.now(tz=timezone.utc),
            lionConfig=dict(lion_kwargs),
            retrieverConfig=effective_retriever_config,
        )

        logger.info("Job %s: launching LionLinker run", job_id)
        lion = self._build_lion_linker(
            paths,
            table,
            retriever,
            lion_kwargs,
            effective_retriever_config,
        )

        await asyncio.to_thread(self._run_lion_linker_blocking, lion)
        logger.info("Job %s: LionLinker run finished", job_id)

        results = self._parse_results(paths.output_csv, table, mention_columns)
        logger.info("Job %s: parsed %d result rows", job_id, len(results))

        prediction_batch_size = self._int_option(
            lion_config,
            ["prediction_batch_size", "predictionBatchSize"],
            self.settings.prediction_batch_rows,
        )
        if prediction_batch_size is None or prediction_batch_size < 1:
            prediction_batch_size = self.settings.prediction_batch_rows

        prediction_batches = await self.store.save_predictions(
            job_id,
            job.dataset_id,
            job.table_id,
            results,
            batch_size=prediction_batch_size,
        )
        logger.info(
            "Job %s: saved predictions (batches=%d, batch_size=%d)",
            job_id,
            prediction_batches,
            prediction_batch_size,
        )

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
            prediction_batches=prediction_batches,
            prediction_batch_size=prediction_batch_size,
        )
        logger.info("Job %s: marked as completed", job_id)

    def _build_retriever(
        self,
        table: DatasetTableRecord,
        config: Optional[Dict[str, Any]],
    ) -> tuple[RetrieverClient, Dict[str, Any]]:
        base_config: Dict[str, Any] = {}
        if table.retriever_config and isinstance(table.retriever_config, dict):
            base_config.update(table.retriever_config)
        else:
            base_config.update(linker_defaults.default_retriever_config())

        effective_config = base_config.copy()
        if config:
            effective_config.update(config)

        class_path = effective_config.get("class_path")
        if class_path:
            retriever_cls = self._load_class(str(class_path))
        else:
            retriever_cls = LamapiClient

        endpoint = effective_config.get("endpoint")
        token = effective_config.get("token")
        num_candidates = self._int_option(effective_config, ["num_candidates"], None)
        kg_value = effective_config.get("kg") or linker_defaults.DEFAULT_KG_NAME
        cache = self._bool_option(effective_config, ["cache"], None)
        max_retries = self._int_option(effective_config, ["max_retries"], None)
        backoff_factor = self._float_option(effective_config, ["backoff_factor"], None)

        recognised = {
            "class_path",
            "endpoint",
            "token",
            "num_candidates",
            "kg",
            "cache",
            "max_retries",
            "backoff_factor",
        }
        extra_kwargs = {
            key: value for key, value in effective_config.items() if key not in recognised
        }

        if retriever_cls is LamapiClient and not endpoint:
            return NullRetriever(num_candidates=num_candidates or 0), effective_config

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
            retriever_instance = retriever_cls(**kwargs)
        except TypeError as exc:  # pragma: no cover - dependency errors surface here
            raise ValueError(f"Unable to initialise retriever {retriever_cls}: {exc}") from exc

        return retriever_instance, effective_config

    def _build_lion_linker(
        self,
        paths: JobPaths,
        table: DatasetTableRecord,
        retriever: RetrieverClient,
        lion_kwargs: Dict[str, Any],
        retriever_config: Optional[Dict[str, Any]],
    ) -> LionLinker:
        kg_name = linker_defaults.DEFAULT_KG_NAME
        if retriever_config and isinstance(retriever_config, dict):
            kg_name = retriever_config.get("kg", kg_name)
        elif table.retriever_config and isinstance(table.retriever_config, dict):
            kg_name = table.retriever_config.get("kg", kg_name)

        merged_kwargs = dict(lion_kwargs)
        merged_kwargs.setdefault("kg_name", kg_name)

        kwargs = {
            "input_csv": str(paths.input_csv),
            "output_csv": str(paths.output_csv),
            "retriever": retriever,
        }
        kwargs.update(merged_kwargs)
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
        raw_mention_columns = metadata.get("mentionColumns") or metadata.get("mention_columns")
        mention_columns_list: List[str] = []
        if isinstance(raw_mention_columns, str):
            mention_columns_list = [raw_mention_columns]
        elif isinstance(raw_mention_columns, Iterable):
            mention_columns_list = [str(col) for col in raw_mention_columns if col is not None]

        if mention_columns_list:
            columns = [col for col in mention_columns_list if col in table.header]
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
        if "id_row" not in df.columns:
            raise ValueError("Result CSV must include an 'id_row' column")

        results: List[ResultRow] = []
        header = table.header
        row_lookup = {row.id_row: row for row in table.rows}

        for _, row_series in df.iterrows():
            try:
                id_row = int(row_series.get("id_row"))
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping row with invalid id_row value: %s", row_series.get("id_row")
                )
                continue

            if id_row not in row_lookup:
                logger.warning(
                    "Skipping row %s from output that does not exist in table %s",
                    id_row,
                    table.table_id,
                )
                continue

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
                if answer_str is None and answer is not None and not pd.isna(answer):
                    answer_str = str(answer)
                if identifier_str is None and identifier is not None and not pd.isna(identifier):
                    identifier_str = str(identifier)
                predictions.append(
                    PredictionSummary(
                        column=column,
                        answer=answer_str,
                        identifier=identifier_str,
                    )
                )

            results.append(ResultRow(idRow=id_row, data=data, predictions=predictions))

        return results
