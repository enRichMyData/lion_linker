from __future__ import annotations

import asyncio
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING, UpdateOne

from app.core.config import Settings
from app.core.linker_defaults import default_lion_config, default_retriever_config
from app.models.dataset import DatasetPayload, DatasetResponse, DatasetTableRecord, TableRowRecord
from app.models.jobs import JobEnqueueResponse, JobRecord, JobStatus, ResultRow


def _slice_results(
    rows: List[Dict[str, object]], page: int, per_page: int
) -> Tuple[List[Dict[str, object]], int]:
    if per_page <= 0:
        raise ValueError("per_page must be positive")
    total = len(rows)
    start = (page - 1) * per_page
    end = start + per_page
    return rows[start:end], total


def _dataframe_from_payload(table: DatasetTableRecord) -> pd.DataFrame:
    rows = [row.model_dump(by_alias=True) for row in table.rows]
    df = pd.DataFrame([row["data"] for row in rows], columns=table.header)
    df.insert(0, "id_row", [row["idRow"] for row in rows])
    return df


class StateStore:
    """MongoDB-backed store for datasets, tables, and jobs."""

    def __init__(self, settings: Settings):
        uri = settings.mongo_uri or "mongodb://localhost:27017"
        self._client = AsyncIOMotorClient(uri)
        self._db = self._client[settings.mongo_db]
        prefix = settings.mongo_collection_prefix or "lion"
        self._datasets = self._db[f"{prefix}_datasets"]
        self._tables = self._db[f"{prefix}_tables"]
        self._table_rows = self._db[f"{prefix}_table_rows"]
        self._jobs = self._db[f"{prefix}_jobs"]
        self._lock = asyncio.Lock()
        self._prediction_batch_rows = settings.prediction_batch_rows

    async def ensure_indexes(self) -> None:
        dataset_indexes = await self._datasets.index_information()
        if "dataset_name_lower_1" in dataset_indexes and dataset_indexes[
            "dataset_name_lower_1"
        ].get("unique"):
            await self._datasets.drop_index("dataset_name_lower_1")
        await self._datasets.create_index("dataset_name_lower")

        table_indexes = await self._tables.index_information()
        table_index_name = "dataset_id_1_table_name_lower_1"
        if table_index_name in table_indexes and table_indexes[table_index_name].get("unique"):
            await self._tables.drop_index(table_index_name)
        await self._tables.create_index(
            [("dataset_id", ASCENDING), ("table_name_lower", ASCENDING)]
        )
        await self._jobs.create_index("tableId")
        await self._jobs.create_index("createdAt")
        await self._table_rows.create_index(
            [("tableId", ASCENDING), ("idRow", ASCENDING)], unique=True
        )
        await self._table_rows.create_index("datasetId")

    async def upsert_dataset(self, payload: DatasetPayload) -> DatasetResponse:
        async with self._lock:
            now = datetime.now(tz=timezone.utc)

            dataset_id = uuid.uuid4().hex
            table_id = uuid.uuid4().hex

            await self._datasets.insert_one(
                {
                    "_id": dataset_id,
                    "dataset_id": dataset_id,
                    "dataset_name": payload.dataset_name,
                    "dataset_name_lower": payload.dataset_name.lower(),
                    "created_at": now,
                    "updated_at": now,
                }
            )

            row_documents = [
                {
                    "_id": f"{table_id}:{row.id_row}",
                    "datasetId": dataset_id,
                    "tableId": table_id,
                    "idRow": row.id_row,
                    "data": row.data,
                }
                for row in payload.rows
            ]
            if row_documents:
                await self._table_rows.insert_many(row_documents)

            metadata = dict(payload.metadata or {})
            lion_config_doc = default_lion_config()
            if payload.lion_config:
                lion_config_doc.update(payload.lion_config)
            retriever_kg = None
            if payload.retriever_config:
                retriever_kg = payload.retriever_config.get("kg")
            retriever_config_doc = default_retriever_config(retriever_kg)
            if payload.retriever_config:
                retriever_config_doc.update(payload.retriever_config)
            metadata["lion_config"] = lion_config_doc
            metadata["retriever_config"] = retriever_config_doc

            record_doc = {
                "_id": table_id,
                "dataset_id": dataset_id,
                "dataset_name": payload.dataset_name,
                "dataset_name_lower": payload.dataset_name.lower(),
                "table_id": table_id,
                "table_name": payload.table_name,
                "table_name_lower": payload.table_name.lower(),
                "header": payload.header,
                "metadata": metadata,
                "created_at": now,
                "updated_at": now,
            }

            await self._tables.insert_one(record_doc)

            return DatasetResponse(
                datasetId=dataset_id,
                tableId=table_id,
                datasetName=payload.dataset_name,
                tableName=payload.table_name,
                header=payload.header,
                rowCount=len(payload.rows),
                createdAt=now,
                updatedAt=now,
            )

    async def get_table(self, dataset_id: str, table_id: str) -> DatasetTableRecord:
        doc = await self._tables.find_one({"_id": table_id, "dataset_id": dataset_id})
        if not doc:
            raise KeyError(f"Table not found: dataset={dataset_id}, table={table_id}")
        rows_cursor = self._table_rows.find({"tableId": table_id}).sort("idRow", ASCENDING)
        rows: List[TableRowRecord] = []
        async for row_doc in rows_cursor:
            rows.append(
                TableRowRecord.model_validate(
                    {
                        "idRow": row_doc["idRow"],
                        "data": row_doc.get("data", []),
                        "annotations": row_doc.get("annotations", []),
                    }
                )
            )
        payload = doc.copy()
        payload.pop("_id", None)
        payload["rows"] = rows
        self._attach_configs(payload)
        return DatasetTableRecord.model_validate(payload)

    async def get_table_by_name(
        self, dataset_name: str, table_name: str
    ) -> Optional[DatasetTableRecord]:
        doc = await self._tables.find_one(
            {
                "dataset_name_lower": dataset_name.lower(),
                "table_name_lower": table_name.lower(),
            },
            sort=[("created_at", DESCENDING)],
        )
        if not doc:
            return None
        rows_cursor = self._table_rows.find({"tableId": doc["table_id"]}).sort("idRow", ASCENDING)
        rows: List[TableRowRecord] = []
        async for row_doc in rows_cursor:
            rows.append(
                TableRowRecord.model_validate(
                    {
                        "idRow": row_doc["idRow"],
                        "data": row_doc.get("data", []),
                        "annotations": row_doc.get("annotations", []),
                    }
                )
            )
        payload = doc.copy()
        payload.pop("_id", None)
        payload["rows"] = rows
        self._attach_configs(payload)
        return DatasetTableRecord.model_validate(payload)

    @staticmethod
    def _attach_configs(payload: Dict[str, Any]) -> None:
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        payload["lion_config"] = metadata.get("lion_config")
        payload["retriever_config"] = metadata.get("retriever_config")

    async def create_job(
        self,
        table: DatasetTableRecord,
        token: Optional[str] = None,
        lion_config: Optional[Dict[str, Any]] = None,
        retriever_config: Optional[Dict[str, Any]] = None,
        row_ids: Optional[List[int]] = None,
    ) -> JobEnqueueResponse:
        now = datetime.now(tz=timezone.utc)
        job_id = uuid.uuid4().hex

        base_lion_config: Dict[str, Any] = {}
        if table.lion_config:
            base_lion_config.update(table.lion_config)
        if lion_config:
            base_lion_config.update(lion_config)

        base_retriever_config: Dict[str, Any] = {}
        if table.retriever_config:
            base_retriever_config.update(table.retriever_config)
        if retriever_config:
            base_retriever_config.update(retriever_config)

        row_ids_clean: Optional[List[int]] = None
        if row_ids:
            row_ids_clean = sorted({int(r) for r in row_ids})

        total_rows = len(row_ids_clean) if row_ids_clean else None

        record = JobRecord(
            jobId=job_id,
            datasetId=table.dataset_id,
            tableId=table.table_id,
            status=JobStatus.pending,
            createdAt=now,
            updatedAt=now,
            token=token,
            lionConfig=base_lion_config or None,
            retrieverConfig=base_retriever_config or None,
            rowIds=row_ids_clean,
            totalRows=total_rows,
        )
        doc = record.model_dump(by_alias=True)
        doc["_id"] = job_id
        await self._jobs.insert_one(doc)
        return JobEnqueueResponse(
            jobId=job_id,
            datasetId=table.dataset_id,
            tableId=table.table_id,
            status=record.status,
            createdAt=now,
            rowIds=row_ids_clean,
        )

    async def update_job(self, job_id: str, **updates) -> None:
        doc = await self._jobs.find_one({"_id": job_id})
        if not doc:
            raise KeyError(f"Job not found: {job_id}")
        record = JobRecord.model_validate(doc)
        record = record.model_copy(update=updates)
        updated_doc = record.model_dump(by_alias=True)
        updated_doc["_id"] = job_id
        await self._jobs.replace_one({"_id": job_id}, updated_doc)

    async def get_job(self, job_id: str) -> JobRecord:
        doc = await self._jobs.find_one({"_id": job_id})
        if not doc:
            raise KeyError(f"Job not found: {job_id}")
        return JobRecord.model_validate(doc)

    async def get_latest_job_for_table(self, table_id: str) -> Optional[JobRecord]:
        doc = await self._jobs.find_one({"tableId": table_id}, sort=[("createdAt", DESCENDING)])
        if not doc:
            return None
        return JobRecord.model_validate(doc)

    async def list_jobs_for_table(self, table_id: str) -> List[JobRecord]:
        cursor = self._jobs.find({"tableId": table_id}).sort("createdAt", ASCENDING)
        docs = await cursor.to_list(length=None)
        return [JobRecord.model_validate(doc) for doc in docs]

    async def set_job_result(
        self,
        job_id: str,
        *,
        output_path: Optional[Path],
        result_path: Optional[Path],
        total_rows: int,
        processed_rows: Optional[int],
    ) -> None:
        updates = {
            "output_path": str(output_path) if output_path else None,
            "result_path": str(result_path) if result_path else None,
            "total_rows": total_rows,
            "processed_rows": processed_rows,
            "updated_at": datetime.now(tz=timezone.utc),
        }
        await self.update_job(job_id, **updates)

    async def save_predictions(
        self,
        job_id: str,
        dataset_id: str,
        table_id: str,
        rows: List[ResultRow],
        *,
        lion_config: Optional[Dict[str, Any]] = None,
        retriever_config: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
    ) -> int:
        if batch_size is None or batch_size < 1:
            batch_size = self._prediction_batch_rows

        batch_size = max(1, batch_size)

        operations: List[UpdateOne] = []
        for row in rows:
            annotations = [prediction.model_dump(mode="json") for prediction in row.predictions]
            operations.append(
                UpdateOne(
                    {"tableId": table_id, "idRow": row.idRow},
                    {"$set": {"annotations": annotations}},
                )
            )

        if operations:
            await self._table_rows.bulk_write(operations, ordered=False)

        return math.ceil(len(rows) / batch_size) if rows else 0

    async def get_predictions_page(
        self,
        dataset_id: str,
        table_id: str,
        page: int,
        per_page: int,
        row_ids: Optional[List[int]] = None,
    ) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if page < 1 or per_page < 1:
            return [], None

        collected: List[Dict[str, Any]] = []
        query: Dict[str, Any] = {"datasetId": dataset_id, "tableId": table_id}
        shared_meta: Optional[Dict[str, Any]] = None

        if row_ids:
            row_id_list = sorted({int(rid) for rid in row_ids})
            if not row_id_list:
                return [], None
            query["idRow"] = {"$in": row_id_list}
            cursor = self._table_rows.find(query).sort("idRow", ASCENDING)
            docs_all = await cursor.to_list(length=None)
            start = (page - 1) * per_page
            docs = docs_all[start : start + per_page]
        else:
            skip = (page - 1) * per_page
            cursor = (
                self._table_rows.find(query).sort("idRow", ASCENDING).skip(skip).limit(per_page)
            )
            docs = await cursor.to_list(length=per_page)

        for doc in docs:
            if shared_meta is None:
                shared_meta = doc.get("annotationMeta")
            collected.append(
                {
                    "idRow": doc["idRow"],
                    "data": doc.get("data", []),
                    "predictions": doc.get("annotations", []),
                }
            )

        return collected, shared_meta

    async def close(self) -> None:
        self._client.close()

    @staticmethod
    def slice_results(
        rows: List[Dict[str, object]], page: int, per_page: int
    ) -> Tuple[List[Dict[str, object]], int]:
        return _slice_results(rows, page, per_page)

    @staticmethod
    def dataframe_from_payload(table: DatasetTableRecord) -> pd.DataFrame:
        return _dataframe_from_payload(table)
