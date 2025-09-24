from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

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

        mention_columns = self._resolve_mention_columns(table)
        if not mention_columns:
            raise ValueError("No mention columns available for table")

        await self.store.update_job(
            job_id,
            message="Preparing data for LionLinker",
            total_rows=len(table.rows),
            updated_at=datetime.now(tz=timezone.utc),
        )

        paths = self._prepare_paths(job.job_id, table)
        df = StateStore.dataframe_from_payload(table)
        await asyncio.to_thread(df.to_csv, paths.input_csv, index=False)

        if self.settings.dry_run:
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

        retriever = self._build_retriever(table)
        lion = self._build_lion_linker(paths, table, mention_columns, retriever)

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

    def _build_retriever(self, table: DatasetTableRecord) -> RetrieverClient:
        if not self.settings.retriever_endpoint:
            return NullRetriever()
        return LamapiClient(
            endpoint=self.settings.retriever_endpoint,
            token=self.settings.retriever_token,
            num_candidates=self.settings.retriever_num_candidates,
            kg=table.kg_reference,
            cache=self.settings.retriever_cache,
        )

    def _build_lion_linker(
        self,
        paths: JobPaths,
        table: DatasetTableRecord,
        mention_columns: List[str],
        retriever: RetrieverClient,
    ) -> LionLinker:
        chunk_size = max(1, min(self.settings.chunk_size, max(len(table.rows), 1)))
        return LionLinker(
            input_csv=str(paths.input_csv),
            output_csv=str(paths.output_csv),
            model_name=self.settings.model_name,
            retriever=retriever,
            chunk_size=chunk_size,
            mention_columns=mention_columns,
            model_api_provider=self.settings.model_api_provider,
            ollama_host=self.settings.ollama_host,
            model_api_key=self.settings.model_api_key,
            table_ctx_size=self.settings.table_ctx_size,
            kg_name=table.kg_reference,
        )

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

    def _resolve_mention_columns(self, table: DatasetTableRecord) -> List[str]:
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
