from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies import get_store, get_task_queue
from app.models.dataset import DatasetPayload, DatasetResponse
from app.models.jobs import (
    JobEnqueueResponse,
    JobInfoResponse,
    JobStatus,
    JobStatusResponse,
    ResultRow,
)
from app.services.task_queue import TaskQueue
from app.storage.state import StateStore

router = APIRouter()


async def _read_rows_from_path(path: Path) -> List[ResultRow]:
    if not path.exists():
        return []

    def _read() -> List[dict]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw

    data = await asyncio.to_thread(_read)
    return [ResultRow.model_validate(item) for item in data]


def _require_snake_case(config: Optional[dict], context: str) -> None:
    if not config:
        return
    for key in config.keys():
        if not key:
            raise HTTPException(status_code=400, detail=f"{context} contains an empty key")
        if key.strip() != key:
            raise HTTPException(
                status_code=400, detail=f"{context} key '{key}' contains leading/trailing spaces"
            )
        if key != key.lower() or any(
            ch for ch in key if not (ch.islower() or ch.isdigit() or ch == "_")
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{context} key '{key}' must be snake_case "
                    "(lowercase letters, digits, underscores)"
                ),
            )


@router.post("/dataset", response_model=List[DatasetResponse])
async def register_dataset(
    payload: List[DatasetPayload],
    store: StateStore = Depends(get_store),
) -> List[DatasetResponse]:
    responses = []
    for item in payload:
        responses.append(await store.upsert_dataset(item))
    return responses


@router.post("/annotate", response_model=List[JobEnqueueResponse])
async def annotate_tables(
    payload: List[DatasetPayload],
    token: Optional[str] = Query(default=None),
    store: StateStore = Depends(get_store),
    queue: TaskQueue = Depends(get_task_queue),
) -> List[JobEnqueueResponse]:
    responses: List[JobEnqueueResponse] = []
    for item in payload:
        _require_snake_case(item.lion_config, "lionConfig")
        _require_snake_case(item.retriever_config, "retrieverConfig")
        dataset_response = await store.upsert_dataset(item)
        table = await store.get_table(dataset_response.datasetId, dataset_response.tableId)
        job = await store.create_job(
            table,
            token=token,
            lion_config=item.lion_config,
            retriever_config=item.retriever_config,
        )
        responses.append(job)
        await queue.enqueue(job.jobId)
    return responses


@router.get("/dataset/{dataset_id}/table/{table_id}", response_model=JobStatusResponse)
async def job_status(
    dataset_id: str,
    table_id: str,
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=50, ge=1, le=500),
    token: Optional[str] = Query(default=None),
    store: StateStore = Depends(get_store),
) -> JobStatusResponse:
    try:
        table = await store.get_table(dataset_id, table_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Dataset or table not found") from exc

    job = await store.get_latest_job_for_table(table.table_id)
    if not job:
        raise HTTPException(status_code=404, detail="No job found for this table")

    if job.token and token and token != job.token:
        raise HTTPException(status_code=403, detail="Token mismatch for requested job")

    total_rows = job.total_rows or len(table.rows)
    message = job.message
    rows: List[ResultRow] = []
    if job.status == JobStatus.completed:
        predictions = await store.get_predictions_page(
            job.dataset_id, job.table_id, page, per_page
        )
        if predictions:
            rows = [ResultRow.model_validate(item) for item in predictions]
        elif job.result_path:
            all_rows = await _read_rows_from_path(Path(job.result_path))
            subset, total_rows = StateStore.slice_results(
                [row.model_dump() for row in all_rows], page, per_page
            )
            rows = [ResultRow.model_validate(item) for item in subset]

    response = JobStatusResponse(
        datasetId=dataset_id,
        tableId=table_id,
        jobId=job.job_id,
        status=job.status,
        page=page,
        perPage=per_page,
        totalRows=total_rows,
        rows=rows,
        message=message,
        updatedAt=job.updated_at,
        predictionBatches=job.prediction_batches,
        predictionBatchSize=job.prediction_batch_size,
    )
    return response


@router.get("/annotate/{job_id}", response_model=JobInfoResponse)
async def annotation_info(
    job_id: str,
    token: Optional[str] = Query(default=None),
    store: StateStore = Depends(get_store),
) -> JobInfoResponse:
    try:
        job = await store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    if job.token and token and token != job.token:
        raise HTTPException(status_code=403, detail="Token mismatch for requested job")

    total_rows = job.total_rows
    if total_rows is None:
        try:
            table = await store.get_table(job.dataset_id, job.table_id)
        except KeyError:
            table = None
        if table is not None:
            total_rows = len(table.rows)

    return JobInfoResponse(
        jobId=job.job_id,
        datasetId=job.dataset_id,
        tableId=job.table_id,
        status=job.status,
        totalRows=total_rows,
        processedRows=job.processed_rows,
        message=job.message,
        updatedAt=job.updated_at,
        lionConfig=job.lion_config,
        retrieverConfig=job.retriever_config,
        predictionBatches=job.prediction_batches,
        predictionBatchSize=job.prediction_batch_size,
    )
