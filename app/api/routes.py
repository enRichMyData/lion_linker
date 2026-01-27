from __future__ import annotations

import asyncio
import base64
import copy
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, Security, status
from fastapi.responses import FileResponse, StreamingResponse

from app.core import api_limits
from app.core.config import settings
from app.dependencies import get_llm_api_key, get_store, get_task_queue, require_api_key
from app.models.api import (
    ArtifactsResponse,
    CandidateEntry,
    CandidateResponse,
    CapabilitiesData,
    CapabilitiesResponse,
    DownloadResponse,
    JobCreateRequest,
    JobCreateResponse,
    JobProgress,
    JobStatusResponse,
    ResultsPageResponse,
    UploadCreateRequest,
    UploadCreateResponse,
)
from app.models.queue import JobStatus, JobTable, PredictionRecord, TableRow
from app.services.task_queue import TaskQueue
from app.storage.state import StateStore

router = APIRouter(dependencies=[Depends(require_api_key)])


def _encode_cursor(job_id: str, seq: int) -> str:
    payload = json.dumps({"job_id": job_id, "seq": seq}).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("utf-8").rstrip("=")


def _decode_cursor(cursor: str) -> Optional[Dict[str, Any]]:
    if not cursor:
        return None
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("utf-8"))
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _inline_table_to_job_table(table_payload) -> JobTable:
    rows = []
    for idx, row in enumerate(table_payload.rows):
        cells = ["" if value is None else str(value) for value in row.cells]
        rows.append(TableRow(row_id=idx, data=cells))
    return JobTable(header=table_payload.header, rows=rows)


def _prediction_status(prediction: PredictionRecord) -> str:
    if prediction.status:
        return prediction.status
    entity_id = (prediction.entity.id or "").strip().upper()
    if entity_id == "NIL":
        return "nil"
    return "linked"


def _parse_candidate_ranking(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, str) and raw.strip():
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []
    else:
        payload = raw
    if isinstance(payload, dict):
        candidates = payload.get("candidate_ranking") or payload.get("candidateRanking") or []
    elif isinstance(payload, list):
        candidates = payload
    else:
        candidates = []
    return [item for item in candidates if isinstance(item, dict)]


def _pick_candidate_entry(
    entries: List[Dict[str, Any]],
    predicted_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not entries:
        return None
    for entry in entries:
        if entry.get("match") is True:
            return entry
    if predicted_id:
        predicted_norm = str(predicted_id).strip()
        for entry in entries:
            entry_id = str(entry.get("id", "")).strip()
            if entry_id == predicted_norm:
                return entry
    return entries[0]


def _strip_model_api_key(
    config: Optional[Dict[str, Any]],
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not config or not isinstance(config, dict):
        return None, config

    cleaned = copy.deepcopy(config)
    model_api_key: Optional[str] = None

    def _pop_key(payload: Dict[str, Any]) -> None:
        nonlocal model_api_key
        value = payload.pop("model_api_key", None)
        if isinstance(value, str) and value.strip():
            model_api_key = value.strip()

    _pop_key(cleaned)
    for key in ("lion", "lionConfig", "lion_config"):
        nested = cleaned.get(key)
        if isinstance(nested, dict):
            _pop_key(nested)

    return model_api_key, cleaned


@router.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "status": "healthy", "time": datetime.now(tz=timezone.utc).isoformat()}


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def capabilities() -> CapabilitiesResponse:
    data = CapabilitiesData(
        input_modes=["inline", "uri", "upload_id"],
        supported_formats=["text/csv", "application/json"],
        max_inline_bytes=api_limits.MAX_INLINE_BYTES,
        max_rows_inline=api_limits.MAX_ROWS_INLINE,
        max_top_k=api_limits.MAX_TOP_K,
        max_link_columns=api_limits.MAX_LINK_COLUMNS,
        supports_sse=True,
        supports_exports=True,
        default_top_k=api_limits.DEFAULT_TOP_K,
        default_timeout_seconds=api_limits.DEFAULT_TIMEOUT_SECONDS,
    )
    return CapabilitiesResponse(data=data)


@router.post("/jobs", response_model=JobCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    payload: JobCreateRequest,
    store: StateStore = Depends(get_store),
    queue: TaskQueue = Depends(get_task_queue),
    llm_api_key: Optional[str] = Security(get_llm_api_key),
) -> JobCreateResponse:
    if payload.top_k is None:
        payload.top_k = api_limits.DEFAULT_TOP_K
    if payload.top_k is not None and payload.top_k > api_limits.MAX_TOP_K:
        raise HTTPException(status_code=422, detail="top_k exceeds max_top_k")
    if payload.link_columns and len(payload.link_columns) > api_limits.MAX_LINK_COLUMNS:
        raise HTTPException(status_code=422, detail="link_columns exceeds max_link_columns")

    job_table = None
    if payload.input.mode.value == "inline":
        if payload.input.format.value != "application/json":
            raise HTTPException(status_code=422, detail="inline input must be application/json")
        row_count = len(payload.input.table.rows) if payload.input.table else 0
        if row_count > api_limits.MAX_ROWS_INLINE:
            raise HTTPException(status_code=413, detail="inline input exceeds max_rows_inline")
        encoded = json.dumps(payload.model_dump()).encode("utf-8")
        if len(encoded) > api_limits.MAX_INLINE_BYTES:
            raise HTTPException(status_code=413, detail="inline input exceeds max_inline_bytes")
        job_table = _inline_table_to_job_table(payload.input.table)
    else:
        if payload.input.format.value != "text/csv":
            raise HTTPException(status_code=422, detail="non-inline input must be text/csv")

    model_api_key, cleaned_config = _strip_model_api_key(payload.config)
    if not model_api_key and llm_api_key:
        model_api_key = llm_api_key
    payload_for_store = payload.model_copy(deep=True)
    if cleaned_config is not None or payload.config is not None:
        payload_for_store.config = cleaned_config

    job = await store.create_job_v1(payload_for_store, table=job_table)
    if model_api_key:
        await store.save_job_secret(job.job_id, model_api_key)
    await queue.enqueue(job.job_id)

    return JobCreateResponse(
        job_id=job.job_id,
        table_id=job.table_id,
        status=job.status.value,
        created_at=job.created_at,
        limits={"top_k": payload.top_k},
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str, store: StateStore = Depends(get_store)) -> JobStatusResponse:
    try:
        job = await store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    progress = JobProgress(
        rows_total=job.rows_total,
        rows_processed=job.rows_processed,
        cells_total=job.cells_total,
        cells_processed=job.cells_processed,
    )

    return JobStatusResponse(
        job_id=job.job_id,
        table_id=job.table_id,
        status=job.status.value,
        progress=progress,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
    )


@router.get("/jobs/{job_id}/results", response_model=ResultsPageResponse)
async def job_results(
    job_id: str,
    cursor: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    store: StateStore = Depends(get_store),
) -> ResultsPageResponse:
    try:
        await store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    decoded = _decode_cursor(cursor) if cursor else None
    after_seq = None
    if decoded:
        if decoded.get("job_id") != job_id:
            raise HTTPException(status_code=400, detail="Cursor does not match job_id")
        after_seq = decoded.get("seq")
        if after_seq is not None and not isinstance(after_seq, int):
            raise HTTPException(status_code=400, detail="Invalid cursor")

    page = await store.get_predictions_after(job_id, after_seq, limit + 1)
    results = []
    next_cursor = None
    if len(page) > limit:
        last_item = page[limit - 1]
        next_cursor = _encode_cursor(job_id, int(last_item.seq or 0))
        page = page[:limit]

    for prediction in page:
        raw = prediction.raw or {}
        candidates_raw = _parse_candidate_ranking(raw.get("candidate_ranking"))
        chosen = _pick_candidate_entry(candidates_raw, prediction.entity.id)

        chosen_id = None
        chosen_name = None
        chosen_types = None
        chosen_description = None
        chosen_confidence_label = None
        chosen_confidence_score = None
        chosen_match = None
        if isinstance(chosen, dict):
            chosen_id = str(chosen.get("id", "")) or None
            chosen_name = chosen.get("name") or chosen.get("label")
            chosen_name = str(chosen_name) if chosen_name not in (None, "") else None
            chosen_types = chosen.get("types")
            chosen_description = chosen.get("description")
            chosen_description = (
                str(chosen_description) if chosen_description not in (None, "") else None
            )
            chosen_confidence_label = chosen.get("confidence_label")
            chosen_confidence_score = chosen.get("confidence_score")
            chosen_match = chosen.get("match")

        if chosen_match is None:
            chosen_match = bool(chosen_id and chosen_id == prediction.entity.id)

        results.append(
            {
                "row": prediction.row_id,
                "col": prediction.col_id,
                "cell_id": f"{prediction.row_id}:{prediction.col_id}",
                "mention": prediction.mention,
                "final": {
                    "id": chosen_id or prediction.entity.id,
                    "name": chosen_name or prediction.entity.label,
                    "types": chosen_types,
                    "description": chosen_description,
                    "confidence_label": chosen_confidence_label,
                    "confidence_score": (
                        chosen_confidence_score
                        if chosen_confidence_score is not None
                        else prediction.score
                    ),
                    "match": chosen_match,
                },
            }
        )

    return ResultsPageResponse(
        job_id=job_id,
        cursor=cursor,
        next_cursor=next_cursor,
        results=results,
    )


@router.get(
    "/jobs/{job_id}/cells/{row}/{col}/candidates",
    response_model=CandidateResponse,
)
async def job_cell_candidates(
    job_id: str,
    row: int,
    col: int,
    store: StateStore = Depends(get_store),
) -> CandidateResponse:
    prediction = await store.get_prediction_cell(job_id, row, col)
    if not prediction:
        raise HTTPException(status_code=404, detail="Cell not found")

    raw = prediction.raw or {}
    candidates_raw = _parse_candidate_ranking(raw.get("candidate_ranking"))
    candidates: List[CandidateEntry] = []
    for idx, entry in enumerate(candidates_raw, start=1):
        entry_id = str(entry.get("id", ""))
        name_value = entry.get("name") or entry.get("label")
        name_value = str(name_value) if name_value not in (None, "") else None
        description_value = entry.get("description")
        description_value = (
            str(description_value) if description_value not in (None, "") else None
        )
        candidates.append(
            CandidateEntry(
                rank=idx,
                id=entry_id or None,
                confidence_label=entry.get("confidence_label"),
                confidence_score=entry.get("confidence_score"),
                name=name_value,
                types=entry.get("types"),
                description=description_value,
                entity_id=entry_id,
                label=name_value,
                score=entry.get("confidence_score"),
                features=entry.get("features"),
            )
        )

    return CandidateResponse(
        job_id=job_id,
        row=row,
        col=col,
        cell_id=f"{row}:{col}",
        mention=prediction.mention,
        candidates=candidates,
    )


@router.post("/jobs/{job_id}:cancel", response_model=JobStatusResponse)
async def cancel_job(job_id: str, store: StateStore = Depends(get_store)) -> JobStatusResponse:
    try:
        job = await store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    if job.status != JobStatus.queued:
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job.status.value}; only queued jobs can be cancelled",
        )

    cancelled = await store.cancel_job(job_id)
    if not cancelled:
        raise HTTPException(status_code=409, detail="Job could not be cancelled")

    job = await store.get_job(job_id)
    progress = JobProgress(
        rows_total=job.rows_total,
        rows_processed=job.rows_processed,
        cells_total=job.cells_total,
        cells_processed=job.cells_processed,
    )
    return JobStatusResponse(
        job_id=job.job_id,
        table_id=job.table_id,
        status=job.status.value,
        progress=progress,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
    )


@router.post("/uploads", response_model=UploadCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_upload(
    payload: UploadCreateRequest,
    store: StateStore = Depends(get_store),
) -> UploadCreateResponse:
    upload_id = uuid.uuid4().hex
    expires_at = datetime.now(tz=timezone.utc) + timedelta(hours=1)
    upload_path = str(settings.workspace_path / "uploads" / f"{upload_id}.csv")
    await store.create_upload(
        upload_id,
        {
            "content_type": payload.content_type,
            "content_length": payload.content_length,
            "created_at": datetime.now(tz=timezone.utc),
            "expires_at": expires_at,
            "path": upload_path,
            "status": "pending",
        },
    )
    upload_url = f"/uploads/{upload_id}"
    return UploadCreateResponse(upload_id=upload_id, upload_url=upload_url, expires_at=expires_at)


@router.put("/uploads/{upload_id}")
async def upload_content(
    upload_id: str,
    request: Request,
    store: StateStore = Depends(get_store),
) -> Response:
    upload_doc = await store.get_upload(upload_id)
    if not upload_doc:
        raise HTTPException(status_code=404, detail="Upload not found")

    upload_dir = settings.workspace_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = Path(upload_doc.get("path") or (upload_dir / f"{upload_id}.csv"))
    if not target_path.is_absolute():
        target_path = upload_dir / target_path

    with open(target_path, "wb") as handle:
        async for chunk in request.stream():
            handle.write(chunk)

    await store.update_upload(upload_id, status="ready", path=str(target_path))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, store: StateStore = Depends(get_store)) -> StreamingResponse:
    async def _event_stream():
        while True:
            try:
                job = await store.get_job(job_id)
            except KeyError:
                yield "event: error\ndata: {\"error\":\"Job not found\"}\n\n"
                return

            payload = {
                "job_id": job.job_id,
                "status": job.status.value,
                "rows_processed": job.rows_processed,
                "cells_processed": job.cells_processed,
                "rows_total": job.rows_total,
                "cells_total": job.cells_total,
            }
            yield f"event: progress\ndata: {json.dumps(payload)}\n\n"
            if job.status in {JobStatus.done, JobStatus.failed, JobStatus.cancelled}:
                return
            await asyncio.sleep(1)

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@router.get("/jobs/{job_id}/results/download", response_model=DownloadResponse)
async def job_results_download(
    job_id: str,
    store: StateStore = Depends(get_store),
) -> DownloadResponse:
    try:
        await store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    job_dir = settings.workspace_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    export_path = job_dir / "results.csv"
    if not export_path.exists():
        import csv

        with open(export_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row", "col", "cell_id", "mention", "entity_id", "label", "score", "status"])
            async for doc in store.iter_predictions(job_id):
                prediction = PredictionRecord.model_validate(doc)
                status = _prediction_status(prediction)
                writer.writerow(
                    [
                        prediction.row_id,
                        prediction.col_id,
                        f"{prediction.row_id}:{prediction.col_id}",
                        prediction.mention,
                        prediction.entity.id,
                        prediction.entity.label or "",
                        prediction.score if prediction.score is not None else "",
                        status,
                    ]
                )

    download_url = f"/jobs/{job_id}/artifacts/results.csv"
    expires_at = datetime.now(tz=timezone.utc) + timedelta(hours=1)
    return DownloadResponse(job_id=job_id, download_url=download_url, expires_at=expires_at)


@router.get("/jobs/{job_id}/artifacts", response_model=ArtifactsResponse)
async def job_artifacts(job_id: str, store: StateStore = Depends(get_store)) -> ArtifactsResponse:
    try:
        await store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    job_dir = settings.workspace_path / job_id
    artifacts = []
    if job_dir.exists():
        for path in job_dir.iterdir():
            if path.is_file():
                artifacts.append(
                    {
                        "name": path.name,
                        "type": "export" if path.suffix == ".csv" else "artifact",
                        "size_bytes": path.stat().st_size,
                    }
                )
    return ArtifactsResponse(job_id=job_id, artifacts=artifacts)


@router.get("/jobs/{job_id}/artifacts/{name}")
async def download_artifact(job_id: str, name: str, store: StateStore = Depends(get_store)) -> Response:
    try:
        await store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    path = settings.workspace_path / job_id / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")

    media_type = "application/octet-stream"
    if path.suffix == ".csv":
        media_type = "text/csv"
    elif path.suffix == ".json":
        media_type = "application/json"
    elif path.suffix in {".txt", ".log"}:
        media_type = "text/plain"

    return FileResponse(
        path,
        media_type=media_type,
        filename=path.name,
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )
