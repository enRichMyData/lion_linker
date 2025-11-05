from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"


class JobRecord(BaseModel):
    job_id: str = Field(alias="jobId")
    dataset_id: str = Field(alias="datasetId")
    table_id: str = Field(alias="tableId")
    status: JobStatus
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    token: Optional[str] = None
    message: Optional[str] = None
    output_path: Optional[str] = Field(default=None, alias="outputPath")
    result_path: Optional[str] = Field(default=None, alias="resultPath")
    total_rows: Optional[int] = Field(default=None, alias="totalRows")
    processed_rows: Optional[int] = Field(default=None, alias="processedRows")
    lion_config: Optional[Dict[str, Any]] = Field(default=None, alias="lionConfig")
    retriever_config: Optional[Dict[str, Any]] = Field(default=None, alias="retrieverConfig")
    prediction_batches: Optional[int] = Field(default=None, alias="predictionBatches")
    prediction_batch_size: Optional[int] = Field(default=None, alias="predictionBatchSize")
    row_ids: Optional[List[int]] = Field(default=None, alias="rowIds")

    class Config:
        populate_by_name = True


class PredictionSummary(BaseModel):
    column: str
    answer: Optional[Any] = None
    identifier: Optional[str] = None
    class Config:
        populate_by_name = True


class AnnotationMeta(BaseModel):
    jobId: str
    updatedAt: datetime = Field(alias="updatedAt")
    lionConfig: Optional[Dict[str, Any]] = Field(default=None, alias="lionConfig")
    retrieverConfig: Optional[Dict[str, Any]] = Field(default=None, alias="retrieverConfig")


class ResultRow(BaseModel):
    idRow: int
    data: List[str]
    predictions: List[PredictionSummary]
    annotationMeta: Optional[AnnotationMeta] = None


class JobStatusResponse(BaseModel):
    datasetId: str
    tableId: str
    jobId: str
    status: JobStatus
    page: int
    perPage: int
    totalRows: int
    rows: List[ResultRow]
    message: Optional[str] = None
    updatedAt: datetime
    predictionBatches: Optional[int] = None
    predictionBatchSize: Optional[int] = None
    rowIds: Optional[List[int]] = None


class JobEnqueueResponse(BaseModel):
    jobId: str
    datasetId: str
    tableId: str
    status: JobStatus
    createdAt: datetime
    rowIds: Optional[List[int]] = None


class JobInfoResponse(BaseModel):
    jobId: str
    datasetId: str
    tableId: str
    status: JobStatus
    totalRows: Optional[int] = None
    processedRows: Optional[int] = None
    message: Optional[str] = None
    updatedAt: datetime
    lionConfig: Optional[Dict[str, Any]] = None
    retrieverConfig: Optional[Dict[str, Any]] = None
    predictionBatches: Optional[int] = None
    predictionBatchSize: Optional[int] = None
    rowIds: Optional[List[int]] = None
