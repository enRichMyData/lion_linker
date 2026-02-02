from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from app.core import linker_defaults
from app.core.config import Settings, settings
from app.models.queue import JobRecord, JobSelection, JobStatus, JobTable, PredictionEntity, PredictionRecord
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


class LinkerRunner:
    DEFAULT_MODEL_NAME = linker_defaults.DEFAULT_MODEL_NAME
    DEFAULT_MODEL_PROVIDER = linker_defaults.DEFAULT_MODEL_PROVIDER
    DEFAULT_CHUNK_SIZE = linker_defaults.DEFAULT_CHUNK_SIZE
    DEFAULT_OLLAMA_HOST: Optional[str] = linker_defaults.DEFAULT_OLLAMA_HOST
    DEFAULT_OLLAMA_API_KEY: Optional[str] = linker_defaults.DEFAULT_OLLAMA_API_KEY
    DEFAULT_FORMAT_CANDIDATES = linker_defaults.DEFAULT_FORMAT_CANDIDATES
    DEFAULT_COMPACT_CANDIDATES = linker_defaults.DEFAULT_COMPACT_CANDIDATES
    DEFAULT_MAX_PARALLEL_PROMPTS = linker_defaults.DEFAULT_MAX_PARALLEL_PROMPTS

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
        if job.task.upper() != "CEA":
            raise ValueError(f"Unsupported task: {job.task}")

        lion_config, retriever_config = self._split_configs(job.config)
        if job.top_k:
            retriever_config["num_candidates"] = job.top_k
            lion_config["max_candidates_per_task"] = job.top_k
        job_secret = await self.store.get_job_secret(job.job_id)
        if job_secret:
            lion_config["model_api_key"] = job_secret
        store_prompts = self._should_store_prompts(job.config)
        row_range = job.row_range if isinstance(job.row_range, dict) else None
        paths = self._prepare_paths(job.job_id)
        input_csv, header = await self._prepare_input_csv(job, paths, row_range)
        paths.input_csv = input_csv
        mention_columns = self._resolve_mention_columns(
            header,
            job.selection,
            lion_config,
            job.link_columns,
        )
        if not mention_columns:
            raise ValueError("No mention columns available for table")

        dry_run = self._bool_option(lion_config, ["dry_run", "dryRun"], self.settings.dry_run)

        total_rows = self._estimate_total_rows(job, row_range, input_csv)
        total_cells = total_rows * len(mention_columns) if total_rows is not None else None
        await self.store.update_job(
            job.job_id,
            rows_total=total_rows,
            cells_total=total_cells,
            table_id=job.table_id,
        )

        if dry_run:
            processed_rows = set()
            processed_cells = 0

            def _batch_iter():
                nonlocal processed_cells, processed_rows
                for batch in self._iter_nil_predictions(
                    job, mention_columns, retriever_config, input_csv, row_range
                ):
                    for item in batch:
                        processed_cells += 1
                        processed_rows.add(item.row_id)
                    yield batch

            await self.store.save_predictions_batches(job.job_id, _batch_iter())
            await self.store.update_job(
                job.job_id,
                status=JobStatus.done,
                rows_processed=len(processed_rows),
                cells_processed=processed_cells,
                finished_at=datetime.now(tz=timezone.utc),
                error=None,
            )
            return

        retriever, effective_retriever_config = self._build_retriever(retriever_config)

        lion_kwargs = self._build_lion_kwargs(lion_config, mention_columns)
        provider = lion_kwargs.get("model_api_provider") or self.DEFAULT_MODEL_PROVIDER
        limiter = self._provider_limits.get(provider) or self._fallback_provider_limit

        if limiter:
            async with limiter:
                await self._execute_job(
                    job,
                    paths,
                    mention_columns,
                    header,
                    lion_kwargs,
                    retriever,
                    effective_retriever_config,
                    store_prompts=store_prompts,
                )
        else:
            await self._execute_job(
                job,
                paths,
                mention_columns,
                header,
                lion_kwargs,
                retriever,
                effective_retriever_config,
                store_prompts=store_prompts,
            )

    @staticmethod
    def _run_lion_linker_blocking(lion: LionLinker) -> None:
        asyncio.run(lion.run())

    @staticmethod
    def _split_configs(
        config: Optional[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        lion_config = linker_defaults.default_lion_config()
        retriever_config = linker_defaults.default_retriever_config()
        if not config:
            return lion_config, retriever_config

        lion_override = None
        retriever_override = None
        if isinstance(config, dict):
            lion_override = (
                config.get("lion") or config.get("lionConfig") or config.get("lion_config")
            )
            retriever_override = (
                config.get("retriever")
                or config.get("retrieverConfig")
                or config.get("retriever_config")
            )
            if lion_override is None and retriever_override is None:
                lion_override = config

        if isinstance(lion_override, dict):
            lion_config.update(lion_override)

        if isinstance(retriever_override, dict):
            base = linker_defaults.default_retriever_config(retriever_override.get("kg"))
            base.update(retriever_override)
            retriever_config = base

        return lion_config, retriever_config

    @staticmethod
    def _bool_from_value(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
        return bool(value)

    def _should_store_prompts(self, config: Optional[Dict[str, Any]]) -> bool:
        if not config or not isinstance(config, dict):
            return False
        for key in ("store_prompts", "storePrompts"):
            if key in config:
                return bool(self._bool_from_value(config.get(key)))
        debug = config.get("debug")
        if isinstance(debug, dict):
            for key in ("store_prompts", "storePrompts"):
                if key in debug:
                    return bool(self._bool_from_value(debug.get(key)))
        lion = config.get("lion") or config.get("lionConfig") or config.get("lion_config")
        if isinstance(lion, dict):
            for key in ("store_prompts", "storePrompts"):
                if key in lion:
                    return bool(self._bool_from_value(lion.get(key)))
        return False

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

    @staticmethod
    def _dataframe_from_table(table: JobTable, row_range: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        data = []
        rows = table.rows
        if row_range:
            start = int(row_range.get("start", 0))
            limit = int(row_range.get("limit", len(rows)))
            if start < 0:
                start = 0
            if limit < 0:
                limit = 0
            rows = rows[start : start + limit]
        for row in rows:
            row_data = ["" if value is None else str(value) for value in row.data]
            data.append(row_data)
        df = pd.DataFrame(data, columns=table.header)
        df.insert(0, "id_row", [row.row_id for row in rows])
        return df

    async def _prepare_input_csv(
        self,
        job: JobRecord,
        paths: JobPaths,
        row_range: Optional[Dict[str, Any]],
    ) -> tuple[Path, List[str]]:
        if job.table is not None:
            df = self._dataframe_from_table(job.table, row_range)
            await asyncio.to_thread(df.to_csv, paths.input_csv, index=False)
            return paths.input_csv, list(job.table.header)

        input_payload = job.input or {}
        mode = str(input_payload.get("mode") or "")
        input_format = str(input_payload.get("format") or "")
        if input_format and input_format != "text/csv":
            raise ValueError("Only text/csv is supported for non-inline input")

        source_uri = input_payload.get("uri")
        if mode == "upload_id":
            upload_id = input_payload.get("upload_id")
            if not upload_id:
                raise ValueError("upload_id is required for upload input")
            upload_doc = await self.store.get_upload(str(upload_id))
            if not upload_doc or not upload_doc.get("path"):
                raise ValueError("Upload not found or not ready")
            source_uri = upload_doc["path"]

        if not source_uri:
            raise ValueError("Input source is missing")

        source_path = await asyncio.to_thread(self._materialize_source, source_uri, paths.input_csv)
        if row_range:
            sliced_path = paths.input_csv.with_name("input_slice.csv")
            await asyncio.to_thread(
                self._slice_csv,
                source_path,
                sliced_path,
                int(row_range.get("start", 0)),
                int(row_range.get("limit", 0)),
            )
            source_path = sliced_path

        header = await asyncio.to_thread(self._read_csv_header, source_path)
        return source_path, header

    @staticmethod
    def _materialize_source(source_uri: str, target_path: Path) -> Path:
        if source_uri.startswith("http://") or source_uri.startswith("https://"):
            import urllib.request

            with urllib.request.urlopen(source_uri) as response, open(target_path, "wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            return target_path
        if source_uri.startswith("file://"):
            source_uri = source_uri[len("file://") :]
        source_path = Path(source_uri)
        if not source_path.exists():
            raise ValueError("Input URI does not exist")
        if source_path.resolve() != target_path.resolve():
            import shutil

            shutil.copyfile(source_path, target_path)
            return target_path
        return source_path

    @staticmethod
    def _read_csv_header(source_path: Path) -> List[str]:
        import csv

        with open(source_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
        if not header:
            raise ValueError("CSV input is missing a header row")
        return [str(item) for item in header]

    @staticmethod
    def _slice_csv(source_path: Path, target_path: Path, start: int, limit: int) -> None:
        if limit < 1:
            raise ValueError("row_range.limit must be >= 1")
        skiprows = range(1, start + 1) if start > 0 else None
        df = pd.read_csv(source_path, skiprows=skiprows, nrows=limit)
        if "id_row" not in df.columns:
            df.insert(0, "id_row", list(range(start, start + len(df))))
        df.to_csv(target_path, index=False)

    def _estimate_total_rows(
        self,
        job: JobRecord,
        row_range: Optional[Dict[str, Any]],
        input_csv: Path,
    ) -> Optional[int]:
        if row_range:
            try:
                limit = int(row_range.get("limit", 0))
            except (TypeError, ValueError):
                limit = 0
            return limit if limit > 0 else None
        if job.table is not None:
            return len(job.table.rows)
        try:
            with open(input_csv, "r", encoding="utf-8") as handle:
                return sum(1 for _ in handle) - 1
        except Exception:
            return None

    @staticmethod
    def _count_rows_from_batches(batches: List[List[PredictionRecord]]) -> int:
        rows = {item.row_id for batch in batches for item in batch}
        return len(rows)

    def _resolve_mention_columns(
        self,
        header: List[str],
        selection: Optional[JobSelection],
        lion_config: Dict[str, Any],
        link_columns: Optional[List[Any]] = None,
    ) -> List[str]:
        if selection and selection.columns:
            resolved: List[str] = []
            for item in selection.columns:
                if isinstance(item, int):
                    if item < 0 or item >= len(header):
                        raise ValueError(f"Selection column index out of range: {item}")
                    resolved.append(header[item])
                else:
                    column_name = str(item)
                    if column_name not in header:
                        raise ValueError(f"Selection column not found: {column_name}")
                    resolved.append(column_name)
            if resolved:
                return resolved

        if link_columns:
            resolved = []
            for item in link_columns:
                if isinstance(item, int):
                    if item < 0 or item >= len(header):
                        raise ValueError(f"Link column index out of range: {item}")
                    resolved.append(header[item])
                else:
                    column_name = str(item)
                    if column_name not in header:
                        raise ValueError(f"Link column not found: {column_name}")
                    resolved.append(column_name)
            if resolved:
                return resolved

        mention_override = self._list_option(lion_config, ["mention_columns", "mentionColumns"])
        if mention_override:
            columns = [col for col in mention_override if col in header]
            if columns:
                return columns

        return [header[0]] if header else []

    def _build_lion_kwargs(self, lion_config: Dict[str, Any], mention_columns: List[str]) -> Dict[str, Any]:
        chunk_size = self._int_option(lion_config, ["chunk_size"], self.DEFAULT_CHUNK_SIZE)
        if chunk_size is None or chunk_size < 1:
            chunk_size = self.DEFAULT_CHUNK_SIZE

        lion_kwargs: Dict[str, Any] = {}
        lion_kwargs["model_name"] = self._get_option(
            lion_config, ["model_name"], self.DEFAULT_MODEL_NAME
        ) or self.DEFAULT_MODEL_NAME
        lion_kwargs["chunk_size"] = chunk_size
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
            self.DEFAULT_OLLAMA_API_KEY,
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
        max_parallel_prompts = self._int_option(
            lion_config,
            ["max_parallel_prompts", "maxParallelPrompts"],
            self.DEFAULT_MAX_PARALLEL_PROMPTS,
        )
        if max_parallel_prompts is not None:
            lion_kwargs["max_parallel_prompts"] = max_parallel_prompts

        gt_columns = self._list_option(lion_config, ["gt_columns"], None)
        if gt_columns is not None:
            lion_kwargs["gt_columns"] = gt_columns

        recognised = {
            "model_name",
            "model_api_provider",
            "model_api_key",
            "ollama_host",
            "chunk_size",
            "table_ctx_size",
            "tableCtxSize",
            "mention_columns",
            "mentionColumns",
            "format_candidates",
            "compact_candidates",
            "max_parallel_prompts",
            "maxParallelPrompts",
            "gt_columns",
            "dry_run",
            "dryRun",
            "store_prompts",
            "storePrompts",
        }
        extra_kwargs = {
            key: value for key, value in lion_config.items() if key not in recognised
        }
        lion_kwargs.update(extra_kwargs)

        return lion_kwargs

    async def _execute_job(
        self,
        job: JobRecord,
        paths: JobPaths,
        mention_columns: List[str],
        header: List[str],
        lion_kwargs: Dict[str, Any],
        retriever: RetrieverClient,
        effective_retriever_config: Dict[str, Any],
        *,
        store_prompts: bool = False,
    ) -> None:
        logger.info("Job %s: launching LionLinker run", job.job_id)
        prompt_records: List[Dict[str, Any]] = []

        prompt_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        if store_prompts:
            def _capture_prompt(payload: Dict[str, Any]) -> None:
                prompt_text = payload.get("prompt")
                if not isinstance(prompt_text, str) or not prompt_text.strip():
                    return
                prompt_records.append(
                    {
                        "seq": len(prompt_records),
                        "prompt": prompt_text,
                        "response": payload.get("response"),
                        "error": payload.get("error"),
                        "task_ids": payload.get("task_ids"),
                        "batch_index": payload.get("batch_index"),
                        "task_count": payload.get("task_count"),
                        "type": payload.get("type"),
                        "created_at": datetime.now(tz=timezone.utc),
                    }
                )

            prompt_callback = _capture_prompt

        try:
            lion = self._build_lion_linker(
                paths,
                retriever,
                lion_kwargs,
                effective_retriever_config,
                prompt_callback=prompt_callback,
            )
            await asyncio.to_thread(self._run_lion_linker_blocking, lion)
            logger.info("Job %s: LionLinker run finished", job.job_id)

            processed_rows = set()
            processed_cells = 0

            def _batch_iter():
                nonlocal processed_cells, processed_rows
                for batch in self._iter_predictions(
                    job,
                    paths.output_csv,
                    mention_columns,
                    header,
                    effective_retriever_config,
                ):
                    for item in batch:
                        processed_cells += 1
                        processed_rows.add(item.row_id)
                    yield batch

            await self.store.save_predictions_batches(job.job_id, _batch_iter())
            await self.store.update_job(
                job.job_id,
                status=JobStatus.done,
                rows_processed=len(processed_rows),
                cells_processed=processed_cells,
                finished_at=datetime.now(tz=timezone.utc),
                error=None,
            )
            logger.info("Job %s: marked as done", job.job_id)
        finally:
            if store_prompts and prompt_records:
                try:
                    await self.store.save_prompts(job.job_id, prompt_records)
                except Exception as exc:  # pragma: no cover - diagnostics only
                    logger.warning("Job %s: unable to store prompts: %s", job.job_id, exc)

    def _build_retriever(self, config: Optional[Dict[str, Any]]) -> tuple[RetrieverClient, Dict[str, Any]]:
        base_config = linker_defaults.default_retriever_config()
        if config and isinstance(config, dict):
            base_config.update(config)
        effective_config = base_config

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
        retriever: RetrieverClient,
        lion_kwargs: Dict[str, Any],
        retriever_config: Optional[Dict[str, Any]],
        prompt_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> LionLinker:
        kg_name = linker_defaults.DEFAULT_KG_NAME
        if retriever_config and isinstance(retriever_config, dict):
            kg_name = retriever_config.get("kg", kg_name)

        merged_kwargs = dict(lion_kwargs)
        merged_kwargs.setdefault("kg_name", kg_name)

        kwargs = {
            "input_csv": str(paths.input_csv),
            "output_csv": str(paths.output_csv),
            "retriever": retriever,
            "batch_size": 3,
        }
        if prompt_callback is not None:
            kwargs["prompt_callback"] = prompt_callback
        kwargs.update(merged_kwargs)
        return LionLinker(**kwargs)

    def _prepare_paths(self, job_id: str) -> JobPaths:
        job_dir = self.settings.workspace_path / self._slugify(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        return JobPaths(
            input_csv=job_dir / "input.csv",
            output_csv=job_dir / "output.csv",
        )

    @staticmethod
    def _slugify(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "job"

    def _build_nil_predictions(
        self,
        job: JobRecord,
        mention_columns: List[str],
        retriever_config: Dict[str, Any],
    ) -> List[PredictionRecord]:
        batches = list(
            self._iter_nil_predictions(job, mention_columns, retriever_config, None, None)
        )
        predictions: List[PredictionRecord] = []
        for batch in batches:
            predictions.extend(batch)
        return predictions

    def _iter_nil_predictions(
        self,
        job: JobRecord,
        mention_columns: List[str],
        retriever_config: Dict[str, Any],
        input_csv: Optional[Path],
        row_range: Optional[Dict[str, Any]],
    ):
        source = str(retriever_config.get("kg") or linker_defaults.DEFAULT_KG_NAME)
        seq_counter = 0

        if job.table is not None:
            header = list(job.table.header)
            col_index = {col: idx for idx, col in enumerate(header)}
            rows = job.table.rows
            if row_range:
                start = int(row_range.get("start", 0))
                limit = int(row_range.get("limit", len(rows)))
                if start < 0:
                    start = 0
                if limit < 0:
                    limit = 0
                rows = rows[start : start + limit]
            batch: List[PredictionRecord] = []
            for row in rows:
                for column in mention_columns:
                    col_id = col_index.get(column)
                    if col_id is None:
                        continue
                    mention_value = row.data[col_id]
                    mention = "" if mention_value is None else str(mention_value)
                    if not mention.strip():
                        continue
                    batch.append(
                        PredictionRecord(
                            job_id=job.job_id,
                            row_id=row.row_id,
                            col_id=col_id,
                            mention=mention,
                            entity=PredictionEntity(id="NIL", source=source, label=None),
                            score=0.0,
                            status="nil",
                            seq=seq_counter,
                            raw={"dry_run": True},
                        )
                    )
                    seq_counter += 1
                if len(batch) >= 1000:
                    yield batch
                    batch = []
            if batch:
                yield batch
            return

        if input_csv is None:
            return

        header = self._read_csv_header(input_csv)
        col_index = {col: idx for idx, col in enumerate(header)}
        usecols = [col for col in mention_columns if col in header]
        if "id_row" in header:
            usecols = ["id_row"] + usecols
        row_offset = int(row_range.get("start", 0)) if row_range else 0
        for chunk in pd.read_csv(input_csv, chunksize=1000, usecols=usecols):
            batch: List[PredictionRecord] = []
            for idx, row in enumerate(chunk.iterrows(), start=row_offset):
                row = row[1]
                row_id = idx
                if "id_row" in chunk.columns:
                    try:
                        row_id = int(row.get("id_row"))
                    except (TypeError, ValueError):
                        row_id = idx
                for column in mention_columns:
                    col_id = col_index.get(column)
                    if col_id is None or column not in chunk.columns:
                        continue
                    mention_value = row.get(column)
                    mention = "" if mention_value is None else str(mention_value)
                    if not mention.strip():
                        continue
                    batch.append(
                        PredictionRecord(
                            job_id=job.job_id,
                            row_id=row_id,
                            col_id=col_id,
                            mention=mention,
                            entity=PredictionEntity(id="NIL", source=source, label=None),
                            score=0.0,
                            status="nil",
                            seq=seq_counter,
                            raw={"dry_run": True},
                        )
                    )
                    seq_counter += 1
            if batch:
                yield batch
            row_offset += len(chunk)

    @staticmethod
    def _parse_candidate_ranking(raw_value: Any) -> List[Dict[str, Any]]:
        if isinstance(raw_value, str) and raw_value.strip():
            try:
                payload = json.loads(raw_value)
            except json.JSONDecodeError:
                return []
        elif isinstance(raw_value, list):
            payload = raw_value
        else:
            return []

        if isinstance(payload, dict):
            candidates = payload.get("candidate_ranking") or payload.get("candidateRanking") or []
        elif isinstance(payload, list):
            candidates = payload
        else:
            candidates = []
        return [item for item in candidates if isinstance(item, dict)]

    @staticmethod
    def _pick_entity_entry(
        entries: List[Dict[str, Any]], predicted_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if not entries:
            return None
        if predicted_id and predicted_id.strip().upper() == "NIL":
            for entry in entries:
                entry_id = str(entry.get("id", "")).strip()
                if entry_id.upper() == "NIL":
                    return entry
            return None
        for entry in entries:
            if entry.get("match") is True:
                return entry
        if predicted_id:
            predicted_norm = LinkerRunner._normalize_entity_id(predicted_id)
            for entry in entries:
                entry_id = str(entry.get("id", "")).strip()
                if entry_id == predicted_id:
                    return entry
                entry_norm = LinkerRunner._normalize_entity_id(entry_id)
                if predicted_norm and entry_norm and predicted_norm == entry_norm:
                    return entry
        return entries[0]

    @staticmethod
    def _coerce_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None

    @staticmethod
    def _normalize_entity_id(value: Any) -> Optional[str]:
        text = LinkerRunner._coerce_str(value)
        if not text:
            return None
        if "/" in text:
            text = text.rsplit("/", 1)[-1]
        if ":" in text:
            text = text.rsplit(":", 1)[-1]
        return text or None

    @staticmethod
    def _clean_raw_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        if isinstance(value, (dict, list, str, int, float, bool)):
            return value
        return str(value)

    def _parse_predictions(
        self,
        job: JobRecord,
        output_csv: Path,
        mention_columns: List[str],
        header: List[str],
        retriever_config: Dict[str, Any],
    ) -> List[PredictionRecord]:
        predictions: List[PredictionRecord] = []
        for batch in self._iter_predictions(
            job, output_csv, mention_columns, header, retriever_config
        ):
            predictions.extend(batch)
        return predictions

    def _iter_predictions(
        self,
        job: JobRecord,
        output_csv: Path,
        mention_columns: List[str],
        header: List[str],
        retriever_config: Dict[str, Any],
    ):
        col_index = {col: idx for idx, col in enumerate(header)}
        source = str(retriever_config.get("kg") or linker_defaults.DEFAULT_KG_NAME)
        seq_counter = 0

        for chunk in pd.read_csv(output_csv, chunksize=2000):
            if "id_row" not in chunk.columns:
                raise ValueError("Result CSV must include an 'id_row' column")
            batch: List[PredictionRecord] = []
            for _, row_series in chunk.iterrows():
                try:
                    row_id = int(row_series.get("id_row"))
                except (TypeError, ValueError):
                    continue

                for column in mention_columns:
                    col_id = col_index.get(column)
                    if col_id is None or column not in chunk.columns:
                        continue
                    mention_value = row_series.get(column)
                    mention = "" if mention_value is None else str(mention_value)
                    if not mention.strip():
                        continue

                    raw_pred_id = row_series.get(f"{column}_pred_id")
                    predicted_id = None
                    if raw_pred_id is not None and not pd.isna(raw_pred_id):
                        predicted_id = str(raw_pred_id).strip()
                    if predicted_id and predicted_id.lower() == "nan":
                        predicted_id = None

                    raw_candidate_ranking = row_series.get(f"{column}_candidate_ranking")
                    raw_llm_answer = row_series.get(f"{column}_llm_answer")

                    entries = self._parse_candidate_ranking(raw_candidate_ranking)
                    entry = self._pick_entity_entry(entries, predicted_id)

                    entity_id = predicted_id or ""
                    label = None
                    score = None
                    if entry:
                        entry_id = self._coerce_str(entry.get("id"))
                        if entry_id:
                            entity_id = entry_id
                        label = self._coerce_str(entry.get("name") or entry.get("label"))
                        score = self._coerce_float(
                            entry.get("score")
                            if entry.get("score") is not None
                            else entry.get("confidence_score")
                        )

                    if entity_id and entity_id.lower() == "nan":
                        continue
                    if not entity_id:
                        continue

                    status = "linked"
                    if entity_id.upper() == "NIL":
                        status = "nil"

                    raw = {
                        "llm_answer": self._clean_raw_value(raw_llm_answer),
                        "candidate_ranking": self._clean_raw_value(raw_candidate_ranking),
                        "pred_id": self._clean_raw_value(raw_pred_id),
                    }
                    if all(value is None for value in raw.values()):
                        raw = None

                    batch.append(
                        PredictionRecord(
                            job_id=job.job_id,
                            row_id=row_id,
                            col_id=col_id,
                            mention=mention,
                            entity=PredictionEntity(id=entity_id, source=source, label=label),
                            score=score,
                            status=status,
                            seq=seq_counter,
                            raw=raw,
                        )
                    )
                    seq_counter += 1

            if batch:
                yield batch
