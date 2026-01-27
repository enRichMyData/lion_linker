from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, ReturnDocument

from app.core.config import Settings
from app.models.queue import (
    JobCreateRequest,
    JobRecord,
    JobStatus,
    JobTable,
    PredictionRecord,
    PredictionResponse,
    PromptResponse,
)
from app.models.api import JobCreateRequest as JobCreateRequestApi


def _collection_name(prefix: str, name: str) -> str:
    prefix = (prefix or "").strip()
    return f"{prefix}_{name}" if prefix else name


class StateStore:
    """MongoDB-backed store for job queue state and predictions."""

    def __init__(self, settings: Settings):
        uri = settings.mongo_uri or "mongodb://localhost:27017"
        self._client = AsyncIOMotorClient(uri)
        self._db = self._client[settings.mongo_db]
        self._settings = settings
        prefix = settings.mongo_collection_prefix or ""
        self._jobs = self._db[_collection_name(prefix, "jobs")]
        self._predictions = self._db[_collection_name(prefix, "predictions")]
        self._prompts = self._db[_collection_name(prefix, "prompts")]
        self._uploads = self._db[_collection_name(prefix, "uploads")]
        self._secrets = self._db[_collection_name(prefix, "secrets")]

    async def ensure_indexes(self) -> None:
        await self._jobs.create_index("job_id", unique=True, sparse=True)
        await self._jobs.create_index("status")
        await self._jobs.create_index("created_at")
        await self._predictions.create_index("job_id")
        await self._predictions.create_index([("job_id", ASCENDING), ("row_id", ASCENDING)])
        await self._predictions.create_index([("job_id", ASCENDING), ("col_id", ASCENDING)])
        await self._predictions.create_index([("job_id", ASCENDING), ("seq", ASCENDING)])
        await self._predictions.create_index(
            [("job_id", ASCENDING), ("row_id", ASCENDING), ("col_id", ASCENDING)]
        )
        await self._predictions.create_index(
            [("job_id", ASCENDING), ("status", ASCENDING), ("score", ASCENDING)]
        )
        await self._prompts.create_index("job_id")
        await self._prompts.create_index([("job_id", ASCENDING), ("seq", ASCENDING)])
        await self._prompts.create_index("created_at")
        await self._uploads.create_index("upload_id", unique=True, sparse=True)
        await self._uploads.create_index("created_at")
        await self._secrets.create_index("job_id", unique=True, sparse=True)
        await self._secrets.create_index("expires_at", expireAfterSeconds=0)

    async def create_job(self, payload: JobCreateRequest) -> JobRecord:
        now = datetime.now(tz=timezone.utc)
        job_id = uuid.uuid4().hex
        record = JobRecord(
            job_id=job_id,
            status=JobStatus.queued,
            task=payload.task,
            table=payload.table,
            selection=payload.selection,
            config=payload.config,
            created_at=now,
        )
        doc = record.model_dump()
        doc["_id"] = job_id
        await self._jobs.insert_one(doc)
        return record

    async def create_job_v1(
        self,
        payload: JobCreateRequestApi,
        *,
        table: JobTable | None = None,
    ) -> JobRecord:
        now = datetime.now(tz=timezone.utc)
        job_id = uuid.uuid4().hex
        input_payload = payload.input.model_dump(mode="json")
        if payload.input.mode.value == "inline":
            input_payload.pop("table", None)
        row_range = payload.row_range.model_dump() if payload.row_range else None
        record = JobRecord(
            job_id=job_id,
            status=JobStatus.queued,
            task="CEA",
            table=table,
            table_id=payload.table_id,
            input=input_payload,
            link_columns=payload.link_columns,
            row_range=row_range,
            top_k=payload.top_k,
            config=payload.config,
            created_at=now,
        )
        doc = record.model_dump()
        doc["_id"] = job_id
        await self._jobs.insert_one(doc)
        return record

    async def get_job(self, job_id: str) -> JobRecord:
        doc = await self._jobs.find_one({"job_id": job_id})
        if not doc:
            raise KeyError(f"Job not found: {job_id}")
        return JobRecord.model_validate(doc)

    async def update_job(self, job_id: str, **updates: Any) -> None:
        if "status" in updates and isinstance(updates["status"], JobStatus):
            updates["status"] = updates["status"].value
        result = await self._jobs.update_one({"job_id": job_id}, {"$set": updates})
        if result.matched_count == 0:
            raise KeyError(f"Job not found: {job_id}")

    async def claim_next_job(self) -> Optional[JobRecord]:
        now = datetime.now(tz=timezone.utc)
        doc = await self._jobs.find_one_and_update(
            {"status": JobStatus.queued.value},
            {"$set": {"status": JobStatus.running.value, "started_at": now}},
            sort=[("created_at", ASCENDING)],
            return_document=ReturnDocument.AFTER,
        )
        if not doc:
            return None
        return JobRecord.model_validate(doc)

    async def mark_running_jobs_failed(self, reason: str) -> int:
        now = datetime.now(tz=timezone.utc)
        result = await self._jobs.update_many(
            {"status": JobStatus.running.value},
            {"$set": {"status": JobStatus.failed.value, "finished_at": now, "error": reason}},
        )
        return result.modified_count

    async def cancel_job(self, job_id: str) -> bool:
        now = datetime.now(tz=timezone.utc)
        result = await self._jobs.update_one(
            {"job_id": job_id, "status": JobStatus.queued.value},
            {"$set": {"status": JobStatus.cancelled.value, "finished_at": now}},
        )
        return result.modified_count > 0

    async def save_predictions(self, job_id: str, predictions: List[PredictionRecord]) -> None:
        await self._predictions.delete_many({"job_id": job_id})
        if not predictions:
            return
        docs = [prediction.model_dump() for prediction in predictions]
        await self._predictions.insert_many(docs)

    async def save_predictions_batches(
        self, job_id: str, batches
    ) -> None:
        await self._predictions.delete_many({"job_id": job_id})
        for batch in batches:
            if not batch:
                continue
            docs = [prediction.model_dump() for prediction in batch]
            await self._predictions.insert_many(docs)

    async def save_prompts(self, job_id: str, prompts: List[Dict[str, Any]]) -> None:
        await self._prompts.delete_many({"job_id": job_id})
        if not prompts:
            return
        docs: List[Dict[str, Any]] = []
        for idx, prompt in enumerate(prompts):
            doc = dict(prompt)
            doc["job_id"] = job_id
            doc.setdefault("seq", idx)
            doc.setdefault("created_at", datetime.now(tz=timezone.utc))
            docs.append(doc)
        if docs:
            await self._prompts.insert_many(docs)

    async def get_predictions_page(
        self, job_id: str, offset: int, limit: int
    ) -> Tuple[List[PredictionResponse], int]:
        if offset < 0 or limit < 1:
            return [], 0
        query = {"job_id": job_id}
        total = await self._predictions.count_documents(query)
        has_seq = await self._predictions.find_one({"job_id": job_id, "seq": {"$exists": True}})
        if has_seq:
            cursor = (
                self._predictions.find({"job_id": job_id, "seq": {"$gte": offset}})
                .sort([("seq", ASCENDING)])
                .limit(limit)
            )
        else:
            cursor = (
                self._predictions.find(query)
                .sort([("row_id", ASCENDING), ("col_id", ASCENDING)])
                .skip(offset)
                .limit(limit)
            )
        docs = await cursor.to_list(length=limit)
        predictions = [PredictionResponse.model_validate(doc) for doc in docs]
        return predictions, total

    async def get_predictions_after(
        self, job_id: str, after_seq: int | None, limit: int
    ) -> List[PredictionRecord]:
        query: Dict[str, Any] = {"job_id": job_id}
        if after_seq is not None:
            query["seq"] = {"$gt": after_seq}
        cursor = (
            self._predictions.find(query)
            .sort([("seq", ASCENDING)])
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        return [PredictionRecord.model_validate(doc) for doc in docs]

    def iter_predictions(self, job_id: str):
        return self._predictions.find({"job_id": job_id}).sort([("seq", ASCENDING)])

    async def get_prediction_cell(
        self, job_id: str, row_id: int, col_id: int
    ) -> Optional[PredictionRecord]:
        doc = await self._predictions.find_one(
            {"job_id": job_id, "row_id": row_id, "col_id": col_id}
        )
        if not doc:
            return None
        return PredictionRecord.model_validate(doc)

    async def create_upload(self, upload_id: str, payload: Dict[str, Any]) -> None:
        doc = dict(payload)
        doc["upload_id"] = upload_id
        await self._uploads.insert_one(doc)

    async def get_upload(self, upload_id: str) -> Optional[Dict[str, Any]]:
        return await self._uploads.find_one({"upload_id": upload_id})

    async def update_upload(self, upload_id: str, **updates: Any) -> None:
        await self._uploads.update_one({"upload_id": upload_id}, {"$set": updates})

    async def save_job_secret(self, job_id: str, model_api_key: str) -> None:
        now = datetime.now(tz=timezone.utc)
        expires_at = now + timedelta(seconds=self._settings.job_secret_ttl_seconds)
        doc = {
            "job_id": job_id,
            "model_api_key": model_api_key,
            "created_at": now,
            "expires_at": expires_at,
        }
        await self._secrets.update_one({"job_id": job_id}, {"$set": doc}, upsert=True)

    async def get_job_secret(self, job_id: str) -> Optional[str]:
        doc = await self._secrets.find_one({"job_id": job_id})
        if not doc:
            return None
        value = doc.get("model_api_key")
        if not isinstance(value, str):
            return None
        return value

    async def get_prompts_page(
        self, job_id: str, offset: int, limit: int
    ) -> Tuple[List[PromptResponse], int]:
        if offset < 0 or limit < 1:
            return [], 0
        query = {"job_id": job_id}
        total = await self._prompts.count_documents(query)
        cursor = (
            self._prompts.find({"job_id": job_id, "seq": {"$gte": offset}})
            .sort([("seq", ASCENDING)])
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        prompts = [PromptResponse.model_validate(doc) for doc in docs]
        return prompts, total

    async def close(self) -> None:
        self._client.close()
