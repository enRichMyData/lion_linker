from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.dependencies import get_settings, get_store, get_task_queue
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
from lion_linker.lion_linker import LionLinker
from lion_linker.retrievers import LamapiClient

router = APIRouter()


async def _read_rows_from_path(path: Path) -> List[ResultRow]:
    if not path.exists():
        return []

    def _read() -> List[dict]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw

    data = await asyncio.to_thread(_read)
    return [ResultRow.model_validate(item) for item in data]


class EntityLinkRequest(BaseModel):
    input_csv: str
    model_name: str = "gemma2:2b"
    output_csv: Optional[str] = "output.csv"
    prompt_file_path: Optional[str] = "lion_linker/prompt/prompt_template.txt"
    chunk_size: int = 64
    mention_columns: Optional[List[str]] = Field(default_factory=lambda: ["title"])
    compact_candidates: bool = True
    model_api_provider: str = "ollama"
    ollama_host: Optional[str] = None
    model_api_key: Optional[str] = None
    gt_columns: Optional[List[str]] = None
    table_ctx_size: int = 1
    format_candidates: bool = True
    num_candidates: int = 20


@router.post("/entity_link")
async def run_entity_link(
    request: EntityLinkRequest,
    settings: Settings = Depends(get_settings),
) -> dict:
    endpoint = settings.retriever_endpoint
    token = settings.retriever_token
    if not endpoint or not token:
        raise HTTPException(
            status_code=500,
            detail="Retriever endpoint/token must be configured via environment variables.",
        )

    retriever = LamapiClient(
        endpoint=endpoint,
        token=token,
        num_candidates=request.num_candidates,
        kg="wikidata",
        cache=settings.retriever_cache,
    )

    mention_columns = request.mention_columns
    if not mention_columns and settings.default_mention_columns:
        mention_columns = settings.default_mention_columns

    ollama_host = request.ollama_host or settings.ollama_host or "http://ollama:11434"

    lion_linker = LionLinker(
        input_csv=request.input_csv,
        model_name=request.model_name,
        retriever=retriever,
        output_csv=request.output_csv,
        prompt_file_path=request.prompt_file_path,
        chunk_size=request.chunk_size,
        mention_columns=mention_columns,
        compact_candidates=request.compact_candidates,
        model_api_provider=request.model_api_provider,
        ollama_host=ollama_host,
        model_api_key=request.model_api_key,
        gt_columns=request.gt_columns,
        table_ctx_size=request.table_ctx_size,
        format_candidates=request.format_candidates,
    )

    await lion_linker.run()

    return {"message": "Entity linking completed", "output_csv": request.output_csv}


@router.post("/dataset", response_model=List[DatasetResponse])
async def register_dataset(
    payload: List[DatasetPayload],
    store: StateStore = Depends(get_store),
) -> List[DatasetResponse]:
    responses = []
    for item in payload:
        responses.append(await store.upsert_dataset(item))
    return responses


@router.post("/createWithArray", response_model=List[JobEnqueueResponse])
async def enqueue_jobs(
    payload: List[DatasetPayload],
    token: Optional[str] = Query(default=None),
    store: StateStore = Depends(get_store),
    queue: TaskQueue = Depends(get_task_queue),
) -> List[JobEnqueueResponse]:
    responses: List[JobEnqueueResponse] = []
    for item in payload:
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
    if job.status == JobStatus.completed and job.result_path:
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
    )
    return response


@router.get("/jobs/{job_id}", response_model=JobInfoResponse)
async def job_info(
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
    )
